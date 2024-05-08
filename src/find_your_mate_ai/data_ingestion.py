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

from typing import List
from pydantic import BaseModel, Field
import pendulum
import typer
from dynaconf import settings
import openai
import pandas as pd

from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core.schema import BaseNode
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.extractors import PydanticProgramExtractor


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

def ingest_profiles_data(source_data_path: str, output_data_path: str) -> List[BaseNode]:
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

def index_profiles_data(nodes: List[BaseNode], output_data_path: str) -> VectorStoreIndex:
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
        separator = '\n' + '='*10 + '\n' + '='* 10 + '\n'
        doc_text_combined = separator.join(doc_texts)
        rows.append({
            'query_id': 'query_id',
            'answer_ids': relevant_doc_ids,
            'query_text': query_text.replace('"', "'"),
            'answer_contents': doc_texts,
            'answer_contents_str': doc_text_combined.replace('"', "'")
        })

    df = pd.DataFrame(rows)
    return df

def generate_synthetic_questions_data(nodes: List[BaseNode],
                                      output_data_path: str) -> List[BaseNode]:
    """Generates synthetic questions data using llama_index function"""
    gpt4 = OpenAI(temperature=1, model="gpt-4")
    qa_dataset = generate_question_context_pairs(
        nodes, llm=gpt4, num_questions_per_chunk=2
    )
    qa_df = from_qa_dataset_to_df(qa_dataset)
    current_time = pendulum.now().format("YYYY-MM-DD_HH-mm-ss")
    filename = f"synthetic_questions_label_studio_ready_{current_time}.json"
    filename_slugified = re.sub(r'\W+', '_', filename)
    qa_df.to_json(Path(output_data_path, filename_slugified), orient="records")
    logging.info("Synthetic questions data generated and saved at %s", output_data_path)
    return qa_df


if __name__ == "__main__":
    typer.run(ingest_profiles_data)
