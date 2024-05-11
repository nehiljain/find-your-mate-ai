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


from config import settings
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
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


def fetch_all_documents_from_mongodb(
    mongo_uri: str, db_name: str, collection_name: str
) -> pd.DataFrame:
    """
    Fetches all metadata from documents stored in a MongoDB collection and returns it as a pandas DataFrame.

    Args:
    mongo_uri (str): MongoDB URI for connecting to the database.
    db_name (str): Name of the database.
    collection_name (str): Name of the collection where documents are stored.

    Returns:
    pd.DataFrame: DataFrame containing all metadata from the documents.
    """
    # Create a new client and connect to the server with MongoDB versioning
    client = MongoClient(settings.MONGO_URI, server_api=ServerApi("1"))

    # Access the specific database and collection
    database = client[db_name]
    collection = database[collection_name]

    # Use pymongoarrow to fetch all documents as a pandas DataFrame
    df = collection.find_pandas_all({})

    if df.empty:
        logging.info("No documents found in MongoDB collection.")
        return pd.DataFrame()
    metadata_df = df["__data__"].apply(lambda x: x.get("metadata", {})).apply(pd.Series)

    # Join the metadata columns back to the original DataFrame
    result_df = df.join(metadata_df)
    result_df = result_df[
        [
            "_id",
            "file_path",
            "file_name",
            "file_size",
            "creation_date",
            "last_modified_date",
        ]
    ]
    logging.info("Fetched %d documents from MongoDB", len(result_df))
    logging.info("Columns in the DataFrame: %s", result_df.columns)
    # Remove duplicate rows from the DataFrame
    result_df.drop_duplicates(
        subset=result_df.columns.difference(["_id"]), keep="first", inplace=True
    )
    logging.info(
        "Dropped %d duplicate rows from the DataFrame", len(df) - len(result_df)
    )
    # Clean up by closing the MongoDB client
    client.close()

    return result_df


# TODO: This is a costly function so we should move it to a proper orchestrator with incremental loads
def ingest_profiles_data(
    source_data_path: str, output_data_path: str
) -> List[BaseNode]:
    """Ingests data from the specified directory path."""
    source_data_path = Path(source_data_path)
    output_data_path = Path(output_data_path)
    if not source_data_path.exists() or not source_data_path.is_dir():
        logging.error("Invalid source data directory path: %s", source_data_path)
        sys.exit(1)
    if not output_data_path.exists() or not output_data_path.is_dir():
        logging.error("Invalid output data directory path: %s", output_data_path)
        sys.exit(1)
    logging.info("Starting data ingestion from %s", source_data_path)

    # Configure OpenAI API key
    openai.api_key = settings.OPENAI_API_KEY
    logging.info("OpenAI API key configured")

    MONGO_URI = settings.MONGO_URI
    storage_context = StorageContext.from_defaults(
        docstore=MongoDocumentStore.from_uri(uri=MONGO_URI),
    )
    # Load documents from the specified directory
    documents = SimpleDirectoryReader(
        source_data_path, required_exts=[".mdx"], filename_as_id=True
    ).load_data()
    documents = documents[:50]
    for doc in documents:
        doc.doc_id = get_file_id(doc.metadata["file_name"], doc.metadata["file_size"])
        doc.excluded_llm_metadata_keys = ["file_name"]
    documents_df = pd.DataFrame([doc.to_dict() for doc in documents])
    metadata_df = documents_df["metadata"].apply(pd.Series)
    documents_df = pd.concat(
        [documents_df.drop(columns=["metadata"]), metadata_df], axis=1
    )
    db_name = "db_docstore"
    collection_name = "docstore/data"
    existing_documents_df = fetch_all_documents_from_mongodb(
        MONGO_URI, db_name, collection_name
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

    gpt3 = OpenAI(model="gpt-4")
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


# TODO: Needs to be implmeneted to use Mongo Vector Store Index
def index_profiles_data(
    nodes: List[BaseNode], output_data_path: str
) -> VectorStoreIndex:
    """Indexes the profiles data using a VectorStore.
    Args:
        nodes: List[BaseNode]: List of nodes to index.
        output_data_path: str: Path to store the index.
    Returns:
        VectorStoreIndex: Index that can be used for retrieval and querying.
    """
    # Create vector store index
    pass


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
    )
    logging.info("Ingested %d nodes and stored in MongoDB", len(nodes))
    generate_synthetic_questions_data(nodes, settings.OUTPUT_DATA_PATH)


if __name__ == "__main__":
    main()
