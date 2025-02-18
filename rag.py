from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
import os
from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination,
    SourceMatchTermination,
)
from autogen_agentchat.teams import SelectorGroupChat
from models import GPT_4O_MODEL, GPT_4O_MINI_MODEL
from tools import SearchTool
import logging
from visual_agent import VisualAgent


class Rag:
    def __init__(self):
        self.figure_and_chunk_pairs = {}
        self.search_tool = SearchTool(self.figure_and_chunk_pairs)

    @property
    def research_agent(self):
        return AssistantAgent(
            name="research_agent",
            tools=[self.search_tool.rag_search_tool],
            model_client=GPT_4O_MINI_MODEL,
            description="A research agent that can help you find information.",
            system_message="You are a senior research agent specialising in company based research for a financial services company. Take the user's query, formulate a series of search terms to to retrieve the relevant information from the Azure Search index, and return the results to the user. You must execute a tool call to the search index. DO NOT USE YOUR INTERNAL KNOWLEDGE.",
        )

    @property
    def answer_agent(self):
        return VisualAgent(
            name="answer_agent",
            model_client=GPT_4O_MODEL,
            description="An agent that can answer questions.",
            system_message="You are a senior data analyst at a financial services company who specialises in writing data driven insights to user's questions. Take the user's question, and the context from the search results, write response that clear addresses the user's question. The user may want to invest in company, therefore focus on providing data driven insights and a critical mindset to answering the question. Format the answer in Markdown to aid understanding. Only use information from the search results to answer the user's question. Keep responses consise and to the point. Answer in no more than 3 paragraphs. If any of the retrieved figures would be useful for the user then display the figure retrieved from the search index by adding <figure ChunkId='<ChunkId for selected figure>' FigureId='<Original FigureId for selected figure from research tool call>'> to the end of your response. DO not generate the chunk id and figure id, always use the original ones from the search tool call.",
            model_client_stream=True,
            chunk_and_figure_pairs=self.figure_and_chunk_pairs,
        )

    def agent_selector(self, messages):
        """Unified selector for the complete flow."""
        logging.info("Messages: %s", messages)
        current_agent = messages[-1].source if messages else "user"
        decision = None

        # If this is the first message start with user_message_rewrite_agent
        if current_agent == "user":
            decision = "research_agent"
        # Handle transition after query rewriting
        elif current_agent == "research_agent":
            decision = "answer_agent"

        print("Transitioning To: ", decision)

        return decision

    @property
    def group_chat(self):
        return SelectorGroupChat(
            [self.research_agent, self.answer_agent],
            termination_condition=SourceMatchTermination(sources=["answer_agent"])
            | MaxMessageTermination(15),
            model_client=GPT_4O_MINI_MODEL,
            selector_func=self.agent_selector,
        )
