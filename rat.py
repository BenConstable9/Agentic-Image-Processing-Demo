from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
import os
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from models import GPT_4O_MODEL, GPT_4O_MINI_MODEL
from tools import RAT_SEARCH_INDEX_TOOL
import logging


research_agent = AssistantAgent(
    name="research_agent",
    tools=[RAT_SEARCH_INDEX_TOOL],
    model_client=GPT_4O_MINI_MODEL,
    description="A research agent that can help you find information.",
    system_message="You are a senior research agent specialising in company based research for a financial services company. Take the user's query, split it into a series of sub-questions and then formulate a series of search terms to to retrieve the relevant information from the Azure Search index. You must execute a tool call to the search index. DO NOT USE YOUR INTERNAL KNOWLEDGE.",
    # model_client_stream=True,
)

answer_agent = AssistantAgent(
    name="answer_agent",
    tools=[],
    model_client=GPT_4O_MODEL,
    description="An agent that can answer questions.",
    system_message="You are a senior data analyst at a financial services company who specialises in writing data driven insights to user's questions. Take the user's question, and the context from the search results, write response that clear addresses the user's question. The user may want to invest in company, therefore focus on providing data driven insights and a critical mindset to answering the question. Format the answer in Markdown to aid understanding. Only use information from the search results to answer the user's question. Answer in no more than 3 paragraphs. To display any of the figures you have found, add <figure id='FIGURE_ID'> to the end of your response.",
    # model_client_stream=True,
)

revise_answer_agent = AssistantAgent(
    name="revise_answer_agent",
    tools=[RAT_SEARCH_INDEX_TOOL],
    model_client=GPT_4O_MODEL,
    description="An agent that can revise answers.",
    system_message="You are a senior data analyst at a financial services company who specialises in improving and revising data driven insights to user's questions. By nature, you are critical and should ALWAYS MAKE IMPROVEMENTS. Review the initial answer provided, and then consider the context of the question and additional information the user might find useful. You MUST request more information through the tool call to improve the answer. Use this to genrate a series of additional search terms to fetch additional information if necessary to improve the answer. The user may want to invest in company, therefore focus on providing data driven insights and a critical mindset to answering the question. **REVISE THE INITIAL ANSWER AND UPDATE IT WITH YOUR ADDITIONAL INSIGHTS TO CREATE A UNIFIED FINAL ANSWER**. Once you are satisfied with the revised answer and that it fully answers the question, return it to the user by ending the conversation with **TERMINATE**. Make sure that the revised answer provides additional insights and recommendations to the user, BUT do not loose any of the context from the initial response. Keep responses consise and to the point. Answer in no more than 3 paragraphs. To display any of the figures you have found, add <figure id='FIGURE_ID'> to the end of your response.",
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
    elif current_agent == "answer_agent":
        decision = "revise_answer_agent"
    elif current_agent == "revise_answer_agent":
        decision = "revise_answer_agent"

    print("Transitioning To: ", decision)

    return decision


## Group Chat Manager
RAT_GROUP_CHAT = SelectorGroupChat(
    [research_agent, answer_agent, revise_answer_agent],
    termination_condition=TextMentionTermination(
        "TERMINATE", sources=["revise_answer_agent"]
    )
    | MaxMessageTermination(15),
    allow_repeated_speaker=True,
    model_client=GPT_4O_MINI_MODEL,
    selector_func=agent_selector,
)
