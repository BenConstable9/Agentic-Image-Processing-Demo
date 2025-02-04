from typing import List, cast
import json
import chainlit as cl
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage, ToolCallRequestEvent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core import CancellationToken
from rag import RAG_GROUP_CHAT
from rat import RAT_GROUP_CHAT

from chainlit.input_widget import Select

@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    # Set the assistant agent in the user session.
    cl.user_session.set("prompt_history", "")  # type: ignore
    cl.user_session.set("rag_agent", RAG_GROUP_CHAT)  # type: ignore
    cl.user_session.set("rat_agent", RAT_GROUP_CHAT)  # type: ignore

    settings = await cl.ChatSettings(
        [
            Select(
                id="Agent",
                label="Agent Choice:",
                values=["RAG Agent", "RAT Agent"],
                initial_index=0,
            )
        ]
    ).send()

    cl.user_session.set("agent", settings["Agent"])  # Store selection in session state

@cl.on_settings_update
async def handle_agent_update(settings):
    value = settings["Agent"]

    cl.user_session.set("agent", settings["Agent"])  # Store selection in session state

    await cl.Message(content=f"Agent set to: {value}").send()


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="How is Shein innovating?",
            message="How is Shein innovating?",
        ),
        cl.Starter(
            label="What are the priority areas for Shein?",
            message="What are the priority areas for Shein?",
        ),
    ]


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    # Get the team from the user session.
    agent = cl.user_session.get("agent")  # type: ignore

    if agent == "RAG Agent":
        team = cast(SelectorGroupChat, cl.user_session.get("rag_agent"))  # type: ignore
    elif agent == "RAT Agent":
        team = cast(SelectorGroupChat, cl.user_session.get("rat_agent")) # type: ignore
    else:
        return


    # Streaming response message.
    streaming_response: cl.Message | None = None
    # Stream the messages from the team.
    async for msg in team.run_stream(
        task=[TextMessage(content=message.content, source="user")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(msg, ToolCallRequestEvent):
            # Handle the tool call request.
            search_terms = msg.content[0].arguments

            # Send the search terms to the user.
            try:
                args = json.loads(search_terms)

                extracted_search_terms = None
                if "query" in args:
                    extracted_search_terms = args["query"]
                elif "queries" in args:
                    extracted_search_terms = args["queries"]
                
                if extracted_search_terms is not None:
                    await cl.Message(content=f"Searching AI Search with...: {extracted_search_terms}").send()
            except json.JSONDecodeError:
                pass
        elif isinstance(msg, ModelClientStreamingChunkEvent):
            # Stream the model client response to the user.
            if streaming_response is None:
                # Start a new streaming response.
                streaming_response = cl.Message(content="", author=msg.source)
            await streaming_response.stream_token(msg.content)
        elif streaming_response is not None:
            # Done streaming the model client response.
            # We can skip the current message as it is just the complete message
            # of the streaming response.
            await streaming_response.send()
            # Reset the streaming response so we won't enter this block again
            # until the next streaming response is complete.
            streaming_response = None
        elif isinstance(msg, TaskResult):
            await cl.Message(content=msg.messages[-1].content).send()
            # Send the task termination message.
            final_message = "Task terminated. "
            if msg.stop_reason:
                final_message += msg.stop_reason
            await cl.Message(content=final_message).send()
        else:
            # Skip all other message types.
            pass