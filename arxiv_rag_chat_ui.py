# Preparing a single combined Python file with both classes and Streamlit app logic
import os
import arxiv
import pandas as pd
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from arxiv import UnexpectedEmptyPageError
import re
import fitz

#clean slate
# chroma = chromadb.Client(Settings(anonymized_telemetry=False))
# try:
#     chroma.delete_collection("arxiv_papers")
# except Exception:
#     pass


class ChromaLangChainEmbeddingWrapper:
    def __init__(self, langchain_embedding):
        self.langchain_embedding = langchain_embedding

    def __call__(self, input):
        return self.langchain_embedding.embed_documents(input)

    def embed_query(self, query):
        return self.langchain_embedding.embed_query(query)

    def embed_documents(self, texts):
        return self.langchain_embedding.embed_documents(texts)

    def name(self):
        return "openai-embeddings"

class ArxivRAGSystem:
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.embedding_model = OpenAIEmbeddings()

        # ✅ Use persistent client (folder "chroma_db")
        try:
            self.client = chromadb.PersistentClient(path="chroma_db")
        except AttributeError:
            self.client = chromadb.Client(
                Settings(persist_directory="chroma_db", anonymized_telemetry=False)
            )

        # ❌ do NOT keep a cached collection handle; just remember the name + wrapper
        self.collection_name = "arxiv_papers"
        self._emb_wrapper = ChromaLangChainEmbeddingWrapper(self.embedding_model)

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )

        self.llm = ChatOpenAI(model_name="gpt-4o-mini")

        # retriever attaches by name (no cached handle)
        self.retriever = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        ).as_retriever()

        self.chain = self._build_chain()
        self.chain.output_key = "answer"
        
    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._emb_wrapper
        )

    def _build_chain(self):
        template = (
            "You are a curious researcher reviewing academic papers."
            "Your are particularly on the lookout for new trends, ideas, research, or technologies that might change lives. This may mean interactions with various things like healthcare might be different, technologies might emerge that will change what some professions look like, "
            "You are essentially trying to use the given context, which is the abstracts of recently published research papers, to predict novelties of the world before they occur."
            "For example, if you had seen Attention Is All You Need, you might predict that machine translation, and other related tasks, will greatly improve with the transformer. You might further infer that these sorts of tasks might propogate beyond scientific settings or spam email detection and create tools that drive an incredible demand for computing power."
            "Here is the context:\n\n{context}\n\n"
            "Question: {question}\n\n"
            "Use the context if it's relevant. If the context isn't useful, use your own general knowledge, but make a note as to whether you are using the given context or general knowledge."
            "When there is relevant context, make a specific note of it, reference the specific title and the arxiv id in your text. If you are unable to include the arXiv id, no need for an in text citation, as it should appear in the sources."
            "Provide a thoughtful, structured response."
        )
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        return ConversationalRetrievalChain.from_llm(
        llm=self.llm,
        retriever=self.retriever,
        memory=self.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"})

    def normalize_arxiv_id(self, entry_id: str) -> str:
        """
        Return a canonical arXiv id like '2509.12345v1' or 'cs/0301001v1'
        given either a URL or an ID-like string. Falls back to the input
        if nothing matches.
        """
        import re
        from urllib.parse import urlparse

        s = entry_id.strip()

        # If it's a URL, reduce to just the last path segment (remove /abs/, /pdf/, .pdf, query)
        if s.startswith("http"):
            u = urlparse(s)
            path = u.path  # e.g. '/abs/2509.12345v1' or '/pdf/2509.12345v1.pdf' or '/abs/cs/0301001v1'
            # strip leading '/', split
            parts = [p for p in path.lstrip("/").split("/") if p]
            if parts and parts[0] in ("abs", "pdf"):
                parts = parts[1:]
            if parts:
                s = "/".join(parts)
                # remove trailing .pdf if present
                s = re.sub(r"\.pdf$", "", s, flags=re.IGNORECASE)

        # Try new-style first: 4 digits '.' 4-5 digits with optional version
        m = re.match(r"^(\d{4}\.\d{4,5})(v\d+)?$", s)
        if m:
            core, ver = m.group(1), m.group(2) or ""
            return f"{core}{ver}"

        # Try legacy: category/7digits with optional version (category may have hyphens or dots)
        m = re.match(r"^([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$", s, flags=re.IGNORECASE)
        if m:
            core, ver = m.group(1), m.group(2) or ""
            return f"{core}{ver}"

        # Last-resort: search inside the string (e.g., if extra text around it)
        m = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", s)
        if m:
            return m.group(1)
        m = re.search(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7}(v\d+)?)", s, flags=re.IGNORECASE)
        if m:
            return m.group(1)

        # Couldn’t find anything; return original string so you can see what it was.
        return entry_id


    def fetch_and_embed(self, query="machine learning AND medical", days_back=7, max_results=100):
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=days_back)
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)

        papers = []
        client = arxiv.Client()

        try:
            results_iterator = client.results(search)
            for result in results_iterator:
                if start_date <= result.published <= now:
                    raw = result.entry_id
                    aid = self.normalize_arxiv_id(raw)
                    print(f"[normalize] {raw} -> {aid}")  # <- temporary debug
                    papers.append({
                        "title": result.title.strip(),
                        "abstract": result.summary.strip().replace('\n', ' '),
                        "arxiv_id": aid,
                        "abs_url": result.entry_id,
                        "pdf_url": f"https://arxiv.org/pdf/{aid}.pdf",
                        "published": result.published.isoformat()
                    })

        except UnexpectedEmptyPageError:
            print("Reached empty page of results. Fewer papers returned than requested.")
            return pd.DataFrame()

        df = pd.DataFrame(papers)
        if df.empty:
            print("No new papers found.")
            return df

        chunks = df['abstract'].tolist()
        ids = [row['arxiv_id'] for _, row in df.iterrows()]
        vectors = self.embedding_model.embed_documents(chunks)

        # ✅ re-fetch by name right before writing
        coll = self._get_collection()
        coll.add(
            ids=ids,
            embeddings=vectors,
            documents=chunks,
            metadatas=df.to_dict(orient="records")
        )
        return df



    def query_with_reassembled_context(self, user_question, k=10):
        # --- debug: show retrieval size ---
        print("\n=== query_with_reassembled_context ===")
        print("[QUESTION]", user_question)
        self.retriever.search_kwargs['k'] = k
        retrieved_docs = self.retriever.get_relevant_documents(user_question)
        print("[RETRIEVE] docs:", len(retrieved_docs))

        if not retrieved_docs:
            print("[RETRIEVE] no docs")
            return "No relevant information found."

        # Group chunks by arxiv_id and keep the first title/url we see
        grouped_text = defaultdict(list)
        id2firstmeta = {}
        ordered_ids = []

        for i, doc in enumerate(retrieved_docs):
            meta = doc.metadata or {}
            content = doc.page_content or ""
            aid = self._best_effort_arxiv_id(meta, content)  # <-- robust fallback
            if aid not in id2firstmeta:
                id2firstmeta[aid] = {
                    "title": meta.get("title", "No Title"),
                    "abs_url": meta.get("abs_url", ""),
                    "pdf_url": meta.get("pdf_url", "")
                }
                ordered_ids.append(aid)
            chunk_index = meta.get("chunk_index", 0)
            grouped_text[aid].append((chunk_index, content, meta))


        # Reassemble per-paper context and include the id in the section header
        full_contexts = []
        for aid in ordered_ids:
            chunks = grouped_text[aid]
            chunks_sorted = sorted(chunks, key=lambda x: x[0])
            combined_text = "\n".join([text for _, text, _ in chunks_sorted])
            title = id2firstmeta[aid]["title"]

            # only include the ID if it looks valid
            show_id = bool(aid and aid != "[missing]")
            header = f"[{title} — arXiv:{aid}]" if show_id else f"[{title}]"

            section = f"{header}\n{combined_text}"
            full_contexts.append(section)

        context = "\n\n".join(full_contexts)
        print("[CONTEXT] length:", len(context))
        print("[CONTEXT] head:", context[:300].replace("\n", " "))

        # Ask the LLM
        result = self.chain.invoke({"question": user_question, "context": context})
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)

        # Build a deterministic Sources block so IDs ALWAYS show
        sources_lines = []
        for aid in ordered_ids:
            meta = id2firstmeta[aid]
            title = meta["title"]
            abs_url = meta["abs_url"]
            sources_lines.append(f"- {title} (arXiv:{aid}) — {abs_url}")

        sources_block = "\n\nSources:\n" + "\n".join(sources_lines)
        final = answer + sources_block

        print("[SOURCES] listed:", len(sources_lines))
        return final

    
    def deep_embed_full_paper(self, arxiv_id, chunk_size=1000):
        import requests, fitz, os

        # ✅ use fresh handle
        coll = self._get_collection()

        # ✅ use the fresh handle here (NOT self.collection)
        paper_meta = coll.get(where={"arxiv_id": arxiv_id})
        if not paper_meta['metadatas']:
            print(f"No metadata found for arxiv_id: {arxiv_id}")
            return

        pdf_url = paper_meta['metadatas'][0].get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if not pdf_url:
            print("PDF URL not found in metadata. Skipping.")
            return

        # Download and extract PDF text
        response = requests.get(pdf_url)
        with open("temp_paper.pdf", "wb") as f:
            f.write(response.content)

        doc = fitz.open("temp_paper.pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        doc.close()
        os.remove("temp_paper.pdf")

        # Chunk the text manually (basic version)
        chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

        deep_collection_name = f"arxiv_paper_{arxiv_id.replace('/', '_')}"
        deep_collection = self.client.get_or_create_collection(
            name=deep_collection_name,
            embedding_function=self._emb_wrapper
        )

        ids = [f"{arxiv_id}_deep_chunk_{i}" for i in range(len(chunks))]
        vectors = self.embedding_model.embed_documents(chunks)

        metadatas = [{"arxiv_id": arxiv_id, "chunk_index": i, "source": "full_paper"} for i in range(len(chunks))]
        deep_collection.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)

        print(f"Full paper embedded under collection: {deep_collection_name}")
        return deep_collection_name
    
    def query_deep_paper(self, arxiv_id: str, user_question: str, k: int = 12):
        deep_name = f"arxiv_paper_{arxiv_id.replace('/', '_')}"
        deep_retriever = Chroma(
            client=self.client,
            collection_name=deep_name,
            embedding_function=self.embedding_model
        ).as_retriever(search_kwargs={"k": k})

        docs = deep_retriever.get_relevant_documents(user_question)
        if not docs:
            return "No deep chunks found for that paper."

        combined = "\n".join([(d.page_content or "") for d in docs])
        # Put the ID loudly into the context so the model uses it
        context = f"[USE THIS ARXIV ID: {arxiv_id}]\n{combined}"

        template = (
            "You are a careful reader of a single arXiv paper.\n"
            "Use the ID shown in the first line of the context when you cite the paper (format: (arXiv:ID)).\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer concisely and include the arXiv ID if possible."
        )
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)

        chain = LLMChain(llm=self.llm, prompt=prompt)

        # DEBUG (leave these prints while you test)
        print("DEEP_PROMPT_VARS:", prompt.input_variables)
        print("DEEP_PROMPT_TEXT_HEAD:", prompt.template[:160].replace("\n"," "))

        result = chain.invoke({"context": context, "question": user_question})
        text = result["text"] if isinstance(result, dict) and "text" in result else str(result)
        return text + f"\n\nSource: arXiv:{arxiv_id}"


    
    
    def reset_memory(self):
        self.memory.clear()
        
    def reset_index(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        # recreate and refresh retriever so nothing points to a deleted UUID
        _ = self._get_collection()
        self.retriever = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        ).as_retriever()
    def _best_effort_arxiv_id(self, meta: dict, text: str) -> str:
        aid = meta.get("arxiv_id")
        if aid: return aid

        # try URLs in metadata
        for key in ("abs_url", "pdf_url"):
            url = meta.get(key)
            if url:
                m = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)', url, flags=re.I)
                if m: return m.group(1)

        # try the page content (from your "[ARXIV-ID: ...]" prefix or any arXiv-looking string)
        m = re.search(r'\[ARXIV-ID:\s*([^\]]+)\]', text)
        if m: return m.group(1).strip()
        m = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', text)
        if m: return m.group(1)
        m = re.search(r'([a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)', text, flags=re.I)
        if m: return m.group(1)

        return "[missing]"




# Streamlit App Logic
st.title("ArXiv Research Chatbot")

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = ArxivRAGSystem()

rag_system = st.session_state.rag_system

if st.button("🧹 New Conversation"):
    rag_system.reset_memory()
    st.session_state.chat_history = []
    st.success("Conversation history cleared.")


st.sidebar.header("Fetch New Papers")
topic = st.sidebar.text_input("Topic keywords", value="machine learning AND medical")
days_back = st.sidebar.slider("Days back", 1, 30, 7)
max_results = st.sidebar.slider("Max results", 10, 200, 50)

if st.sidebar.button("Fetch and Embed"):
    with st.spinner("Fetching and embedding papers..."):
        df = rag_system.fetch_and_embed(query=topic, days_back=days_back, max_results=max_results)
    st.sidebar.success(f"Fetched and embedded {len(df)} papers." if not df.empty else "No new papers found.")

st.header("Ask a Question About This Research")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    st.chat_message(message['role']).markdown(message['content'])

user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_system.query_with_reassembled_context(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.markdown(response)

st.sidebar.header("Deep Embed a Full Paper")
arxiv_id = st.sidebar.text_input("ArXiv ID (for deep embedding)", value="")
if st.sidebar.button("Deep Embed Full Paper"):
    if arxiv_id:
        with st.spinner(f"Embedding full paper: {arxiv_id}..."):
            try:
                collection_name = rag_system.deep_embed_full_paper(arxiv_id)
                st.sidebar.success(f"Embedded to: {collection_name}")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    else:
        st.sidebar.warning("Please enter a valid ArXiv ID.")
        
if st.sidebar.button("🔁 Rebuild RAG system"):
    if "rag_system" in st.session_state:
        del st.session_state["rag_system"]
    st.session_state.rag_system = ArxivRAGSystem()
    st.sidebar.success("Rebuilt RAG system.")
if st.sidebar.button("🗑️ Reset index"):
    rag_system.reset_index()
    st.sidebar.success("Vector index dropped & recreated.")
    
st.sidebar.header("Ask Deep-Embedded Paper")
deep_q = st.sidebar.text_input("Your question about that paper", value="What loss does QWD-GAN optimize?")

if st.sidebar.button("🔎 Ask deep paper"):
    if not arxiv_id:
        st.sidebar.warning("Enter an arXiv ID above, then ask your question.")
    elif not deep_q:
        st.sidebar.warning("Enter a question.")
    else:
        # Try querying first; if the deep collection isn't there yet, embed then retry.
        with st.spinner("Querying deep-embedded paper..."):
            ans = rag_system.query_deep_paper(arxiv_id, deep_q)
            if isinstance(ans, str) and ans.startswith("No deep chunks found for that paper"):
                # lazily deep-embed on demand, then query again
                try:
                    rag_system.deep_embed_full_paper(arxiv_id)
                    ans = rag_system.query_deep_paper(arxiv_id, deep_q)
                except Exception as e:
                    st.sidebar.error(f"Deep embed failed: {e}")
                    st.stop()
        st.write(ans)
        st.sidebar.success("Answered.")


