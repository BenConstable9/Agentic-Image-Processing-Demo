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


class SearchTool:
    def __init__(self, figure_and_chunk_pairs: dict):
        self.figure_and_chunk_pairs = figure_and_chunk_pairs

    def search_index(self, queries: list[str], top) -> list[dict]:

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

                        if result["ChunkId"] not in self.figure_and_chunk_pairs:
                            self.figure_and_chunk_pairs[result["ChunkId"]] = {}

                        # Store the figures for later
                        for figure in result["ChunkFigures"]:
                            for figure in result["ChunkFigures"]:
                                # Convert the base64 image to a bytes object.
                                image_data = base64.b64decode(figure["Data"])

                                self.figure_and_chunk_pairs[result["ChunkId"]][
                                    figure["FigureId"]
                                ] = image_data

                logging.info("Results: %s", results)

        return json.dumps(final_results)

    def rag_search_index(self, search_term: str) -> list[dict]:
        """Search the Azure Search index for the given query."""
        return self.search_index([search_term], top=4)

    def rat_search_index_breadth_first(self, search_terms: list[str]) -> list[dict]:
        """Search the Azure Search index for the given set of queries."""
        return self.search_index(search_terms, top=1)

    def rat_search_index_depth_first(self, search_terms: list[str]) -> list[dict]:
        """Search the Azure Search index for the given set of queries."""
        return self.search_index(search_terms, top=3)

    @property
    def rat_breadth_first_tool(self):
        return FunctionTool(
            self.rat_search_index_breadth_first,
            description="Search the Azure Search index for the given set of queries. Send a minimum of 3 different search terms to the search index to retrieve the relevant information.",
        )

    @property
    def rat_depth_first_tool(self):
        return FunctionTool(
            self.rat_search_index_depth_first,
            description="Search the Azure Search index for the given set of queries. Send a minimum of 3 different search terms to the search index to retrieve the relevant information.",
        )

    @property
    def rag_search_tool(self):
        return FunctionTool(
            self.rag_search_index,
            description="Search the Azure Search index for the given query.",
        )
