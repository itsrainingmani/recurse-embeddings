import os
import sys
from pathlib import Path
from pprint import pprint
from dotenv import load_dotenv

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from supabase.client import Client, create_client

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)  # type: ignore
embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434", model="nomic-embed-text"
)

vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

query = "What are some repositories that were built for learning Rust"
matched_docs = vector_store.similarity_search(query)

pprint(matched_docs[0:3])
