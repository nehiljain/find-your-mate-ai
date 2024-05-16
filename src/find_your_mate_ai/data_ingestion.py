"""
This file contains the data ingestion pipeline for the find_your_mate_ai project.
It is responsible for ingesting data from the specified directory path
and indexing it using LlamaIndex.
After the data is stored in the vector store, it is ready for querying.
"""

import logging
import sys
from pathlib import Path
import re
import hashlib
from typing import List
from pydantic import BaseModel, Field
import pendulum
import openai
import pandas as pd

from pymongo import MongoClient
from pymongo.server_api import ServerApi


from src.find_your_mate_ai.config import settings
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core.schema import BaseNode
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.program.openai import OpenAIPydanticProgram

from llama_index.core.extractors import PydanticProgramExtractor
from pymongoarrow.monkey import patch_all

patch_all()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class CofounderMetadata(BaseModel):
    """Node metadata."""

    name: str = Field(
        ..., description="Name of the founder in the profile. Usually in headers"
    )
    profile_url: str = Field(
        ..., description="The url of the profile which is the url of the page"
    )
    linkedin_url: str = Field(
        ...,
        description="A linked in profile url attached to the profile. Could be None but very rarely.",
    )
    hobbies: List[str] = Field(
        ...,
        description=("List of hobbies of the founder. ex: Cycling, Reading, Gaming"),
    )
    employement_industries: List[str] = Field(
        ...,
        description=(
            "List of industries the founder has worked in. ex: AI, Real Estate, Travel"
        ),
    )
    location: str = Field(
        ...,
        description=(
            "Location of the founder. Usually in the format location: city, state, country. ex: location: New York, NY, USA"
        ),
    )
    age: int = Field(
        ...,
        description=("Age of the founder. Usually in the format age: x. ex: age: 25"),
    )


EXTRACT_TEMPLATE_STR = """\
Here is the content of a founder looking for a match on YCombinator Founder matching platform:

<content>
{context_str}
</content> \
"""


def get_file_id(file_name: str, file_size: int) -> str:
    """
    Generate a unique identifier for a file based on its name.
    """
    return hashlib.sha256(file_name.encode() + str(file_size).encode()).hexdigest()


class MongoConnection:
    """
    MongoConnection is a class that is responsible for connecting to a MongoDB database and retrieving a storage context.
    """

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        server_api_version: str = "1",
    ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.server_api_version = server_api_version

    def get_mongo_client(self):
        """
        Returns a MongoClient instance.
        """
        return MongoClient(
            self.mongo_uri, server_api=ServerApi(self.server_api_version)
        )

    def get_storage_context(self):
        """
        Returns a StorageContext instance. StorageContext is LlamaIndex's class that contains a docstore.
        """
        return StorageContext.from_defaults(
            docstore=MongoDocumentStore.from_uri(
                uri=self.mongo_uri, db_name=self.db_name, namespace=self.collection_name
            ),
        )


def fetch_all_documents_from_mongodb(
    client: MongoClient, db_name: str, collection_name: str
) -> pd.DataFrame:
    """
    Fetches all metadata from documents stored in a MongoDB collection and returns it as a pandas DataFrame.

    Args:
    client (MongoClient): MongoClient instance.
    db_name (str): Name of the database.
    collection_name (str): Name of the collection where documents are stored.

    Returns:
    pd.DataFrame: DataFrame containing all metadata from the documents.
    """
    database = client[db_name]
    collection = database[collection_name]
    df = collection.find_pandas_all({})
    if df.empty:
        logging.info(
            f"No documents found in MongoDB collection {db_name} {collection_name}"
        )
        return pd.DataFrame()
    metadata_df = df["__data__"].apply(lambda x: x.get("metadata", {})).apply(pd.Series)
    result_df = df.join(metadata_df)
    result_df.drop_duplicates(
        subset=result_df.columns.difference(["_id"]), keep="first", inplace=True
    )
    return result_df


def ingest_profiles_data(
    source_data_path: str,
    output_data_path: str,
    openai_api_key: str,
    mongo_conn: MongoConnection,
) -> List[BaseNode]:
    source_data_path = Path(source_data_path)
    output_data_path = Path(output_data_path)
    if not source_data_path.exists() or not source_data_path.is_dir():
        logging.error("Invalid source data directory path: %s", source_data_path)
        sys.exit(1)
    if not output_data_path.exists() or not output_data_path.is_dir():
        logging.error("Invalid output data directory path: %s", output_data_path)
        sys.exit(1)
    logging.info("Starting data ingestion from %s", source_data_path)

    openai.api_key = openai_api_key

    storage_context = mongo_conn.get_storage_context()
    documents = SimpleDirectoryReader(
        source_data_path,
        required_exts=[".mdx"],
        filename_as_id=True,
    ).load_data()
    logging.info("Loaded %d documents from %s", len(documents), source_data_path)
    documents = documents[:50]
    for doc in documents:
        doc.doc_id = get_file_id(doc.metadata["file_name"], doc.metadata["file_size"])
        doc.excluded_llm_metadata_keys = ["file_name"]
    documents_df = pd.DataFrame([doc.to_dict() for doc in documents])
    metadata_df = documents_df["metadata"].apply(pd.Series)
    documents_df = pd.concat(
        [documents_df.drop(columns=["metadata"]), metadata_df], axis=1
    )
    existing_documents_df = fetch_all_documents_from_mongodb(
        mongo_conn.get_mongo_client(), mongo_conn.db_name, mongo_conn.collection_name
    )
    logging.info("Pre-existing documents in MongoDB: %d", len(existing_documents_df))
    # Extract relevant columns for comparison
    comparison_columns = [
        "file_path",
        "file_name",
        "file_size",
        "creation_date",
        "last_modified_date",
    ]
    if not existing_documents_df.empty:
        existing_documents_df = existing_documents_df[comparison_columns]
        # Filter out documents that already exist in MongoDB
        documents_df = documents_df.merge(
            existing_documents_df, on=comparison_columns, how="left", indicator=True
        )
        documents_df = documents_df[documents_df["_merge"] == "left_only"].drop(
            columns=["_merge"]
        )
    else:
        logging.info("No existing documents found in MongoDB.")
    # Filter documents based on the file_name that exist in document_df
    documents = [
        doc
        for doc in documents
        if doc.metadata["file_name"] in documents_df.file_name.values
    ]
    logging.info("Loaded %d new documents from %s", len(documents), source_data_path)

    # Configuring cache
    cache = IngestionCache(
        cache=RedisCache.from_host_and_port("localhost", 6379),
        collection="redis_cache",
    )

    gpt3 = OpenAI(model="gpt-3.5-turbo")
    openai_program = OpenAIPydanticProgram.from_defaults(
        output_cls=CofounderMetadata,
        prompt_template_str=EXTRACT_TEMPLATE_STR,
        llm=gpt3,
        # extract_template_str=EXTRACT_TEMPLATE_STR
    )
    program_extractor = PydanticProgramExtractor(
        program=openai_program, input_key="context_str", show_progress=True
    )

    # Configure document store

    # Configure ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            program_extractor,
            MarkdownNodeParser(),
            OpenAIEmbedding(),
        ],
        docstore=storage_context.docstore,
        cache=cache,
    )

    pipeline_storage_path = f"{output_data_path}/pipeline_storage"
    try:
        if any(Path(pipeline_storage_path).iterdir()):
            pipeline.load(pipeline_storage_path)
            logging.info("Ingestion pipeline configured")
    except FileNotFoundError:
        logging.info(
            "Pipeline storage directory does not exist or is empty. Skipping pipeline load."
        )
    # Run ingestion pipeline
    nodes = pipeline.run(documents=documents)
    logging.info("Pipeline run finished with %d nodes created and stored", len(nodes))
    pipeline.persist(f"{output_data_path}/pipeline_storage")
    logging.info("Ingestion pipeline completed, persisted at %s", output_data_path)

    storage_context.docstore.add_documents(nodes)
    logging.info("Ingested %d nodes and stored in MongoDB", len(nodes))

    return nodes


def index_profiles_data(
    nodes: List[BaseNode],
    output_data_path: str,
    index_name: str = "profiles_index",
    namespace: str = "find_your_mate_ai",
    pinecone_api_key: str = None,
    pinecone_config: dict = None,
) -> VectorStoreIndex:
    """Indexes the profiles data using a VectorStore.
    Args:
        nodes: List[BaseNode]: List of nodes to index.
        output_data_path: str: Path to store the index.
        index_name: str: Name of the Pinecone index to create or connect.
        namespace: str: Namespace for the Pinecone vector store.
        pinecone_api_key: str: API key for Pinecone, defaults to settings if None.
        pinecone_config: dict: Configuration for Pinecone index creation.
    Returns:
        VectorStoreIndex: Index that can be used for retrieval and querying.
    """
    # Set default Pinecone configuration if none provided
    if pinecone_config is None:
        pinecone_config = {
            "dimension": 1536,  # Assuming dimension size from embeddings
            "metric": "cosine",
            "spec": ServerlessSpec(cloud="aws", region="us-west-2"),
        }

    # Use default API key from settings if not provided
    if pinecone_api_key is None:
        pinecone_api_key = settings.PINECONE_API_KEY

    # Initialize Pinecone
    pinecone = Pinecone(api_key=pinecone_api_key)

    # Create or connect to an existing Pinecone index
    try:
        pinecone.create_index(index_name, **pinecone_config)
    except Exception as e:
        logging.info(f"Index {index_name} already exists or other error: {e}")

    pinecone_index = pinecone.Index(index_name)

    # Setup Pinecone Vector Store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, namespace=namespace
    )

    # Create Vector Store Index
    vector_store_index = VectorStoreIndex(nodes=nodes, vector_store=vector_store)
    logging.info(f"Vector store index created and persisted at {output_data_path}")

    return vector_store_index


def from_qa_dataset_to_df(qa_dataset) -> pd.DataFrame:
    """Converts a qa_dataset to a pandas dataframe

    Args:
        qa_dataset: QADataSet: QA dataset to convert to a pandas dataframe.
    Returns:
        pd.DataFrame: Pandas dataframe containing the QA dataset.
    """
    rows = []
    for query_id, query_text in qa_dataset.queries.items():
        relevant_doc_ids = qa_dataset.relevant_docs[query_id]
        doc_texts = [qa_dataset.corpus[doc_id] for doc_id in relevant_doc_ids]
        separator = "\n" + "=" * 10 + "\n" + "=" * 10 + "\n"
        doc_text_combined = separator.join(doc_texts)
        rows.append(
            {
                "query_id": "query_id",
                "answer_ids": relevant_doc_ids,
                "query_text": query_text.replace('"', "'"),
                "answer_contents": doc_texts,
                "answer_contents_str": doc_text_combined.replace('"', "'"),
            }
        )

    df = pd.DataFrame(rows)
    return df


def generate_synthetic_questions_data(
    nodes: List[BaseNode], output_data_path: str
) -> List[BaseNode]:
    """Generates synthetic questions data using llama_index function"""
    gpt4 = OpenAI(temperature=1, model="gpt-3.5-turbo")
    qa_dataset = generate_question_context_pairs(
        nodes, llm=gpt4, num_questions_per_chunk=2
    )
    qa_df = from_qa_dataset_to_df(qa_dataset)
    current_time = pendulum.now().format("YYYY-MM-DD_HH-mm-ss")
    filename = f"synthetic_questions_label_studio_ready_{current_time}.json"
    filename_slugified = re.sub(r"\W+(?!\.json$)", "_", filename)
    qa_df.to_json(Path(output_data_path, filename_slugified), orient="records")
    logging.info(
        "Synthetic questions data generated and saved to %s",
        Path(output_data_path, filename_slugified),
    )
    return qa_df


def load_nodes_from_mongodb(mongo_uri: str) -> List[BaseNode]:
    """
    Fetches all metadata from documents stored in a MongoDB collection and returns it as a pandas DataFrame.
    """

    storage_context = StorageContext.from_defaults(
        docstore=MongoDocumentStore.from_uri(uri=mongo_uri),
    )
    docs = storage_context.docstore.docs
    logging.info("Number of nodes loaded from MongoDB: %d", len(docs))
    return [node for _, node in docs.items()]


def main():
    nodes = ingest_profiles_data(
        source_data_path=settings.SOURCE_DATA_PATH,
        output_data_path=settings.OUTPUT_DATA_PATH,
        mongo_uri=settings.MONGO_URI,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    logging.info("Ingested %d nodes and stored in MongoDB", len(nodes))
    generate_synthetic_questions_data(nodes, settings.OUTPUT_DATA_PATH)


if __name__ == "__main__":
    main()
