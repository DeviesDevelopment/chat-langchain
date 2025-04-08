"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import unicodedata

from parser import langchain_docs_extractor
from dotenv import load_dotenv

import weaviate
from weaviate.collections import Collection
from weaviate.collections.classes.config import Property, CollectionConfig
import weaviate.classes as wvc

from weaviate.classes.init import Auth
from bs4 import BeautifulSoup, SoupStrainer
from langchain_core.documents import Document
from langchain_community.document_loaders import AzureBlobStorageContainerLoader, SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore

load_dotenv(override=True)


WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]


weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)


def check_collection_index():
    if "Sales" not in weaviate_client.collections.list_all():
        weaviate_client.collections.create(
            name="Sales",
            properties=[
                Property(name="text", data_type=wvc.config.DataType.TEXT),
                Property(name="source", data_type=wvc.config.DataType.TEXT),
                Property(name="title", data_type=wvc.config.DataType.TEXT),
            ]
        )


def get_data_vector_store(embeddings: Embeddings):
    check_collection_index()
    return WeaviateVectorStore(
        client=weaviate_client,
        index_name='Sales',
        text_key="text",
        embedding=embeddings,
        attributes=["source", "title"],
    )


def delete_collection():
    try:
        weaviate_client.collections.delete('Sales')
        print(f"Collections deleted successfully")
    except Exception as e:
        print(f"Error deleting index: {e}")


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


def fix_azure_docs_metadata(docs:list[Document]):
    AZURE_BLOB_CONN = os.environ["AZURE_BLOB_CONN"]
    AZURE_BLOB_CONTAINER = os.environ["AZURE_BLOB_CONTAINER"]

    endpoint_key = "BlobEndpoint="
    start_index = AZURE_BLOB_CONN.find(endpoint_key)
    blob_endpoint = AZURE_BLOB_CONN[start_index + len(endpoint_key):]

    for doc in docs:
        raw_source_path = doc.metadata.get('source')

        container_key_index = raw_source_path.find(AZURE_BLOB_CONTAINER)
        folder_path = raw_source_path[container_key_index:]
        
        filename = os.path.basename(folder_path)
        clean_filename = unicodedata.normalize("NFC", filename)

        doc.metadata["source"] = blob_endpoint + folder_path
        doc.metadata["title"] = clean_filename
    return docs


def load_azure_blob_docs():
    AZURE_BLOB_CONN = os.environ["AZURE_BLOB_CONN"]
    AZURE_BLOB_CONTAINER = os.environ["AZURE_BLOB_CONTAINER"]

    azure_blob_docs = AzureBlobStorageContainerLoader(
        conn_str=AZURE_BLOB_CONN,
        container=AZURE_BLOB_CONTAINER,
        prefix="cvs/"
    ).load()

    return fix_azure_docs_metadata(azure_blob_docs)


def load_webpage_docs():
    return SitemapLoader(
        "https://www.devies.se/sitemap.xml",
        filter_urls=["https://www.devies.se/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def ingest_docs():
    embedding = get_embeddings_model()

    # create ElasticSearch instance for data
    data_vector_store = get_data_vector_store(embedding)

    docs_from_webpage = load_webpage_docs()
    logger.info(f"Loaded {len(docs_from_webpage)} docs from sitemap")

    docs_from_webpage = [doc for doc in docs_from_webpage if doc.metadata.get('source') != "https://www.devies.se/news/uppstickaren-devies-rekryterar-patrik-holm-och-expanderar/"]
    logger.info(f"Webpage filter result: {len(docs_from_webpage)} docs from sitemap")

    docs_from_azure_blob = load_azure_blob_docs()
    logger.info(f"Loaded {len(docs_from_azure_blob)} docs from azure blob")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_transformed = text_splitter.split_documents(docs_from_azure_blob + docs_from_webpage)
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # clear index
    delete_collection()

    # create new and populate with new docs
    data_vector_store = get_data_vector_store(embeddings=embedding)
    data_vector_store.add_documents(documents=docs_transformed)

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""


if __name__ == "__main__":
    ingest_docs()
