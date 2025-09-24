FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use your requirements file as-is
COPY arxivReqs.txt .
RUN pip install --no-cache-dir -r arxivReqs.txt

# Copy the app
COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN mkdir -p /app/chroma_db
EXPOSE 8501

CMD ["streamlit", "run", "arxiv_rag_chat_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
