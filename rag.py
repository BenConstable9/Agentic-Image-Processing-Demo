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
from tools import RAG_SEARCH_INDEX_TOOL
import logging


research_agent = AssistantAgent(
    name="research_agent",
    tools=[RAG_SEARCH_INDEX_TOOL],
    model_client=GPT_4O_MINI_MODEL,
    description="A research agent that can help you find information.",
    system_message="You are a senior research agent specialising in company based research for a financial services company. Take the user's query, formulate a series of search terms to to retrieve the relevant information from the Azure Search index, and return the results to the user. You must execute a tool call to the search index. DO NOT USE YOUR INTERNAL KNOWLEDGE.",
)

answer_agent = AssistantAgent(
    name="answer_agent",
    tools=[],
    model_client=GPT_4O_MODEL,
    description="An agent that can answer questions.",
    system_message="You are a senior data analyst at a financial services company who specialises in writing data driven insights to user's questions. Take the user's question, and the context from the search results, write response that clear addresses the user's question. The user may want to invest in company, therefore focus on providing data driven insights and a critical mindset to answering the question. Format the answer in Markdown to aid understanding. Only use information from the search results to answer the user's question. Keep responses consise and to the point. Answer in no more than 3 paragraphs. If any of the retrieved figures would be useful for the user then display the figure retrieved from the search index by adding <figure chunk_id='<CHUNK_ID for selected figure>' figure_id='<FIGURE_ID for selected figure>'> to the end of your response.",
    # model_client_stream=True,
)


def agent_selector(messages):
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


## Group Chat Manager
RAG_GROUP_CHAT = SelectorGroupChat(
    [research_agent, answer_agent],
    termination_condition=SourceMatchTermination(sources=["answer_agent"])
    | MaxMessageTermination(15),
    model_client=GPT_4O_MINI_MODEL,
    selector_func=agent_selector,
)
