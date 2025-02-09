from autogen_core.tools import FunctionTool
from azure.search.documents.models import QueryType, VectorizableTextQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import os
import logging
from dotenv import load_dotenv, find_dotenv
import json
import base64

load_dotenv(find_dotenv())

FIGURE_AND_CHUNK_PAIRS = {}


def search_index(queries: list[str], top) -> list[dict]:

    final_results = {}

    for query in queries:
        vector_query = [
            VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=top * 5,
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
                    top=top,
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
                        "Title": result["Title"],
                        "Chunk": result["Chunk"],
                    }
                    final_results[result["ChunkId"]] = chunk_to_store

                    if result["ChunkId"] not in FIGURE_AND_CHUNK_PAIRS:
                        FIGURE_AND_CHUNK_PAIRS[result["ChunkId"]] = {}

                    # Store the figures for later
                    for figure in result["ChunkFigures"]:
                        for figure in result["ChunkFigures"]:
                            # Convert the base64 image to a bytes object.
                            image_data = base64.b64decode(figure["Data"])

                            FIGURE_AND_CHUNK_PAIRS[result["ChunkId"]][
                                figure["FigureId"]
                            ] = image_data

            logging.info("Results: %s", results)

    return json.dumps(final_results)


def rag_search_index(search_term: str) -> list[dict]:
    """Search the Azure Search index for the given query."""
    return search_index([search_term], top=4)


def rat_search_index_breadth_first(search_terms: list[str]) -> list[dict]:
    """Search the Azure Search index for the given set of queries."""
    return search_index(search_terms, top=1)


def rat_search_index_depth_first(search_terms: list[str]) -> list[dict]:
    """Search the Azure Search index for the given set of queries."""
    return search_index(search_terms, top=3)


RAT_BREADTH_SEARCH_INDEX_TOOL = FunctionTool(
    rat_search_index_breadth_first,
    description="Search the Azure Search index for the given set of queries. Send a minimum of 3 different search terms to the search index to retrieve the relevant information.",
)

RAT_DEPTH_SEARCH_INDEX_TOOL = FunctionTool(
    rat_search_index_depth_first,
    description="Search the Azure Search index for the given set of queries. Send a minimum of 3 different search terms to the search index to retrieve the relevant information.",
)

RAG_SEARCH_INDEX_TOOL = FunctionTool(
    rag_search_index,
    description="Search the Azure Search index for the given query.",
)
