from autogen_core.tools import FunctionTool
from azure.search.documents.models import QueryType, VectorizableTextQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import os
import logging
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def search_index(queries: list[str]) -> list[dict]:
    TOP = 1

    final_results = {}

    for query in queries:
        vector_query = [
            VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=TOP * 5,
                fields="ChunkEmbedding",
            )
        ]

        credential = AzureKeyCredential(
            os.environ["AIService__AzureSearchOptions__Key"]
        )
        retrieval_fields = ["ChunkId", "Title", "Chunk", "ChunkFigures"]
        with SearchClient(
            endpoint=os.environ["AIService__AzureSearchOptions__Endpoint"],
            index_name="image-processing-index",
            credential=credential,
        ) as search_client:

            results = list(
                search_client.search(
                    top=TOP,
                    semantic_configuration_name="image-processing-semantic-config",
                    search_text=query,
                    select=",".join(retrieval_fields),
                    vector_queries=vector_query,
                    query_type=QueryType.SEMANTIC,
                    query_language="en-GB",
                )
            )

            for result in results:
                if (
                    result["ChunkId"] not in final_results
                    and result["@search.reranker_score"] >= 2.5
                ):
                    chunk_to_store = {
                        "Chunks": [result["Chunk"]],
                        # "Figures": result["ChunkFigures"],
                    }
                    final_results[result["ChunkId"]] = chunk_to_store

            logging.info("Results: %s", results)

    return final_results.values()

def single_term_search_index(query: str) -> list[dict]:
    return search_index([query])

RAT_SEARCH_INDEX_TOOL = FunctionTool(
    search_index,
    description="Search the Azure Search index for the given query.",
)

RAG_SEARCH_INDEX_TOOL = FunctionTool(
    single_term_search_index,
    description="Search the Azure Search index for the given query.",
)