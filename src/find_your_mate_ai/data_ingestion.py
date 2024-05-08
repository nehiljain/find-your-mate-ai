"""
This file contains the data ingestion pipeline for the find_your_mate_ai project.
It is responsible for ingesting data from the specified directory path
and indexing it using LlamaIndex.
After the data is stored in the vector store, it is ready for querying.
"""
import logging
import sys
from pathlib import Path

from typing import List, Tuple
from pydantic import BaseModel, Field
import typer
from dynaconf import settings
import openai
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import BaseNode
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.extractors import PydanticProgramExtractor
from llama_index.llms.openai import OpenAI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



class CofounderMetadata(BaseModel):
    """Node metadata."""

    name: str = Field(
        ..., description="Name of the founder in the profile. Usually in headers"
    )
    linkedin_url: str = Field(
        ..., description="A linked in profile url attached to the profile. Could be None but very rarely."
    )
    hobbies: List[str] = Field(
        ...,
        description=(
            "List of hobbies of the founder. ex: Cycling, Reading, Gaming"
        ),
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
        description=(
            "Age of the founder. Usually in the format age: x. ex: age: 25"
        ),
    )

EXTRACT_TEMPLATE_STR = """\
Here is the content of a founder looking for a match on YCombinator Founder matching platform:
<content>
{context_str}
</content>
Given the contextual information, extract out a {class_name} object.\
"""

def ingest_and_index_data(source_data_path: str, output_data_path: str) -> List[BaseNode]:
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


    # Load documents from the specified directory
    documents = SimpleDirectoryReader(source_data_path).load_data()
    logging.info("Loaded %d documents from %s", len(documents), source_data_path)

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
    )
    program_extractor = PydanticProgramExtractor(
        program=openai_program, input_key="input", show_progress=True
    )

    # Configure document store

    # Configure ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            program_extractor,
            MarkdownNodeParser(),
            OpenAIEmbedding(),
        ],
    )
    pipeline.load(f"{output_data_path}/pipeline_storage")
    logging.info("Ingestion pipeline configured")

    # Run ingestion pipeline
    nodes = pipeline.run(documents=documents)
    logging.info("Pipeline run finished with %d nodes created and stored", len(nodes))
    pipeline.persist(f"{output_data_path}/pipeline_storage")
    logging.info("Ingestion pipeline completed, persisted at %s", output_data_path)

    return nodes


if __name__ == "__main__":
    typer.run(ingest_and_index_data)
