FROM python:3.12.0

RUN pip install llama-index-storage-docstore-redis llama-index-vector-stores-redis llama-index-readers-google llama_index docx2txt openai nbconvert schedule

COPY . /app

WORKDIR /app

CMD ["python",load_index.py]






