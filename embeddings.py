import os
import sys
from pathlib import Path
from pprint import pprint
from dotenv import load_dotenv

from langchain_community.vectorstores import SupabaseVectorStore

# from langchain.text_splitter import MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import JSONLoader
from openai import embeddings
from supabase.client import Client, create_client


# Define the metadata extraction function
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["name"] = record.get("name")
    metadata["owner"] = record.get("owner")
    metadata["createdAt"] = record.get("createdAt")
    metadata["description"] = record.get("description")
    langs: list[str] = record.get("languages")  # type: ignore

    metadata["languages"] = ", ".join(langs)
    metadata["url"] = record.get("url")

    return metadata


load_dotenv()

openai_key_str = os.getenv("OPENAI_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if openai_key_str is None or supabase_url is None or supabase_key is None:
    print("Something fucky happened", file=sys.stderr)
    sys.exit(1)

supabase: Client = create_client(supabase_url, supabase_key)  # type: ignore
# embeddings = OpenAIEmbeddings(openai_api_key=openai_key_str)  # type: ignore
embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434", model="nomic-embed-text"
)

# Only useful if we want to split the markdown document
# headers_to_split_on = [
#     ("#", "Header 1"),
#     ("##", "Header 2"),
#     ("###", "Header 3"),
# ]
# markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=20)

# with open("readme-data.txt", "rt") as readme_file:
#     readme_data = readme_file.read()
#     # md_header_splits = markdown_splitter.split_text(readme_data)

#     # how do i inject context metadata for each "document"
#     docs = markdown_splitter.create_documents([readme_data])

#     print(docs[0:10])
#     print(len(docs))

# Naive Approach
# Process the JSON file using the JSON loader from LangChain | use jq to get the specific json elements that we want
# Each README becomes its own embedding. We still want to attach the "metadata" to the readme to put it inside the embedding
# Once we get each embedding, upload it to Supabase

loader = JSONLoader(
    file_path="../data/repos.json",
    jq_schema=".[]",
    content_key="readme",
    text_content=False,
    metadata_func=metadata_func,
)

docs = loader.load()

vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=500,
)
