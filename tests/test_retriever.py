from dynaconf import settings
from src.find_your_mate_ai.data_ingestion import (
    MongoConnection,
    BaseNode,
    ingest_profiles_data,
)
import tempfile
import pytest
import uuid
import logging


@pytest.fixture
def test_settings():
    with settings.using_env("test"):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set the OUTPUT_PATH to the temporary directory
            settings.set("OUTPUT_PATH", temp_dir)
            yield settings


@pytest.fixture
def mongo_connection_mock(test_settings):
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


def test_ingest_profiles_data(mongo_connection_mock, test_settings):
    nodes = ingest_profiles_data(
        source_data_path=test_settings.SOURCE_DATA_PATH,
        output_data_path=test_settings.OUTPUT_DATA_PATH,
        openai_api_key=test_settings.OPENAI_API_KEY,
        mongo_conn=mongo_connection_mock,
    )
    assert len(nodes) == 15, "No nodes were created during ingestion"
    assert isinstance(
        nodes[0], BaseNode
    ), "The created node is not an instance of BaseNode"
