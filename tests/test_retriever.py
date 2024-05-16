from dynaconf import settings
from src.find_your_mate_ai.data_ingestion import (
    MongoConnection,
    BaseNode,
    ingest_profiles_data,
    index_profiles_data,
    ServerlessSpec,
    create_pinecone_index,
    create_auto_retriever,
    OpenAI,
)
import tempfile
import pytest
import uuid
import logging


@pytest.fixture(scope="session")
def test_settings():
    with settings.using_env("test"):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set the OUTPUT_PATH to the temporary directory
            settings.set("OUTPUT_PATH", temp_dir)
            print(settings.to_dict())
            yield settings


# Session-scoped MongoDB setup
@pytest.fixture(scope="session")
def mongo_connection(test_settings):
    unique_collection_name = f"test_collection_{uuid.uuid4()}"
    mongo_connection = MongoConnection(
        mongo_uri=test_settings.MONGO_URI,
        db_name=test_settings.MONGODB_NAME,
        collection_name=unique_collection_name,
    )
    try:
        yield mongo_connection
    finally:
        # Cleanup: drop the collection after the test
        client = mongo_connection.get_mongo_client()
        logging.info(
            f"Dropping collection {unique_collection_name} from database {test_settings.MONGODB_NAME}"
        )
        db = client[test_settings.MONGODB_NAME]
        for collection_name in db.list_collection_names():
            if unique_collection_name in collection_name:
                db.drop_collection(collection_name)
                # TODO: Have a util function to clean up any test_collection_<uuid> collections if the script fails or exits early
                logging.info(
                    f"Collection {collection_name} dropped from database {test_settings.MONGODB_NAME}"
                )
        client.close()


# Session-scoped Pinecone setup
@pytest.fixture(scope="session")
def pinecone_index(test_settings):
    pinecone_config = {
        "dimension": 1536,  # Assuming dimension size from embeddings
        "metric": "cosine",
        "spec": ServerlessSpec(cloud="aws", region="us-west-2"),
    }
    index_name = f"test_idx_{uuid.uuid4().hex[:8]}"

    # Initialize Pinecone
    pinecone_index, pc, index_name = create_pinecone_index(
        index_name=index_name,
        pinecone_api_key=test_settings.PINECONE_API_KEY,
        pinecone_config=pinecone_config,
    )
    try:
        yield pinecone_index
    finally:
        # Teardown Pinecone index
        try:
            pc.delete_index(name=index_name)
            logging.info(f"Successfully deleted Pinecone index: {index_name}")
        except Exception as e:
            logging.error(f"Error deleting Pinecone index: {e}")
            pytest.fail(f"Failed to tear down Pinecone index: {e}")


@pytest.fixture(scope="module")
def ingested_data(mongo_connection, test_settings, pinecone_index):
    nodes = ingest_profiles_data(
        source_data_path=test_settings.SOURCE_DATA_PATH,
        output_data_path=test_settings.OUTPUT_DATA_PATH,
        openai_api_key=test_settings.OPENAI_API_KEY,
        mongo_conn=mongo_connection,
    )
    assert len(nodes) == 15, "No nodes were created during ingestion"
    assert isinstance(
        nodes[0], BaseNode
    ), "The created node is not an instance of BaseNode"

    vector_store_index = index_profiles_data(
        nodes=nodes,
        pinecone_index=pinecone_index,
        namespace="test-find-your-mate-ai",
    )
    assert vector_store_index is not None, "Failed to create Vector Store Index"
    yield nodes, vector_store_index


def test_retriever_with_filter(ingested_data):
    nodes, vector_store_index = ingested_data
    retriever = create_auto_retriever(
        vector_store_index=vector_store_index, llm=OpenAI(model="gpt-4-turbo")
    )
    filtered_nodes = retriever.retrieve("Who are some founders that are 30 or below?")
    assert len(filtered_nodes) == 2, "Found the two relevant nodes"
    expected_file_names = {"ethan-nguyen.mdx", "avery-thompson.mdx"}
    actual_file_names = {node.metadata.get("file_name") for node in filtered_nodes}
    assert (
        actual_file_names == expected_file_names
    ), f"Expected file names {expected_file_names}, but got {actual_file_names}"


def test_retriever_no_filter_found(ingested_data):
    _, vector_store_index = ingested_data
    retriever = create_auto_retriever(
        vector_store_index=vector_store_index, llm=OpenAI(model="gpt-4-turbo")
    )
    filtered_nodes = retriever.retrieve("Who are some founders from indian descent?")
    assert len(filtered_nodes) == 2, "Found the two relevant nodes"
    for node in filtered_nodes:
        assert (
            node.metadata.get("file_name") == "arjun-patel.mdx"
        ), f"Node {node.id} is not correctly identified"
