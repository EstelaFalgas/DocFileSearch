#pip install llama-index-storage-docstore-redis llama-index-vector-stores-redis llama-index-readers-google llama_index docx2txt openai nbconvert schedule python-dotenv

import os
from datetime import datetime
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline, IngestionCache
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.readers.google import GoogleDriveReader
from llama_index.core import StorageContext, VectorStoreIndex

import logging
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#OpenAI API KEY
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OpenAI_key")

# Redis connection details
REDIS_HOST = "localhost"  
REDIS_PORT = 6379

# Google Drive setup
service_account_key_path = os.getenv("service_account_key_path")

folder_id = os.getenv("folder_id")

# Set up Redis document and store, cache and index
docstore = RedisDocumentStore.from_host_and_port(REDIS_HOST, REDIS_PORT, namespace="document_store")
timestamp_cache = RedisCache.from_host_and_port(REDIS_HOST, REDIS_PORT)
cache = IngestionCache(
    cache=RedisCache.from_host_and_port(REDIS_HOST, REDIS_PORT),
    collection="redis_cache",
)
vector_store = RedisVectorStore(redis_url=f"redis://{REDIS_HOST}:{REDIS_PORT}")
index= VectorStoreIndex.from_vector_store(vector_store)
# Initialize Google Drive reader
loader = GoogleDriveReader(service_account_key_path=service_account_key_path)

#clearing logging unwanted info
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

# Load data from Google Drive with caching
def load_doctore():

    documents = []
    for doc in loader.load_data(folder_id=folder_id):
        doc_id = doc.id_  
        last_modified = datetime.strptime(doc.metadata['modified at'], '%Y-%m-%dT%H:%M:%S.%fZ')  

        # Retrieve last seen modification timestamp from cache
        cached_timestamp = timestamp_cache.get(doc_id)
        if cached_timestamp:
            cached_timestamp = datetime.fromisoformat(cached_timestamp)

        # If the document is new or modified, reload it
        if not cached_timestamp or last_modified > cached_timestamp:
            documents.append(doc)
            timestamp_cache.put(doc_id, last_modified.isoformat())  # Update the cache with the latest modification time

    # Update the document store with the new/modified documents
    if documents:
        docstore.add_documents(documents)
        index.refresh(documents)
    else:
        print("No new or modified documents to update.")


load_doctore()

engine= index.as_query_engine()



""" 

import schedule
import time

# Schedule the function to run every day at 00:00
schedule.every().day.at("00:00").do(load_doctore)

# Keep the script running
while True:
    schedule.run_pending()  # Check if there's any pending task to run
    time.sleep(60)  # Sleep for 1 minute before checking again
"""