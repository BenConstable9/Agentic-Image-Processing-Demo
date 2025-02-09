from typing import List, cast
import json
import chainlit as cl
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import (
    ModelClientStreamingChunkEvent,
    TextMessage,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
    TextMessage,
)
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core import CancellationToken
from rag import RAG_GROUP_CHAT
from rat import RAT_GROUP_CHAT
import re
from chainlit.input_widget import Select


def remove_markdown_formatting(text):
    # Remove bold and italic markers while keeping the text inside
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)  # **bold** or __bold__
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)  # *italic* or _italic_

    # Remove inline code formatting
    text = re.sub(r"`(.+?)`", r"\1", text)

    # Remove other Markdown characters while keeping content
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # Remove images
    text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)  # Convert links to plain text
    text = re.sub(r"^\#{1,6}\s*", "", text, flags=re.MULTILINE)  # Remove headers
    text = re.sub(r"^\>\s?", "", text, flags=re.MULTILINE)  # Remove blockquotes
    text = re.sub(
        r"^\s*[-+*]\s+", "", text, flags=re.MULTILINE
    )  # Remove unordered list markers
    text = re.sub(
        r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE
    )  # Remove ordered list markers

    return text


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
        cl.Starter(
            label="What is Shein doing to be more sustainable?",
            message="What is Shein doing to be more sustainable?",
        ),
    ]


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    # Get the team from the user session.
    agent = cl.user_session.get("agent")  # type: ignore

    if agent == "RAG Agent":
        team = cast(SelectorGroupChat, cl.user_session.get("rag_agent"))  # type: ignore
    elif agent == "RAT Agent":
        team = cast(SelectorGroupChat, cl.user_session.get("rat_agent"))  # type: ignore
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
                    extracted_search_terms = ", ".join(args["queries"])

                if extracted_search_terms is not None:
                    await cl.Message(
                        content=f"**Research Agent:**\n\nSearching AI Search with: *'{extracted_search_terms}'*"
                    ).send()
            except json.JSONDecodeError:
                pass
        elif isinstance(msg, ToolCallExecutionEvent):
            # Handle the tool call execution.
            ai_search_results = msg.content[0].content
            try:
                results = json.loads(ai_search_results)

                retrieval_message = (
                    "**Research Agent:**\n\nRetrieved the following information:"
                )
                image_retrievals = []
                for chunk_id, result in results.items():
                    first_150_chars = result["Chunk"][:150]

                    retrieval_message += (
                        f"\n\n *{remove_markdown_formatting(first_150_chars)}...* "
                    )

                    if "Figures" in result:
                        for figure in result["Figures"]:
                            first_150_chars_desc = figure["Description"][:150]
                            image = cl.Image(
                                content=figure["Data"],
                                name=remove_markdown_formatting(first_150_chars_desc),
                                display="inline",
                            )
                            image_retrievals.append(image)

                await cl.Message(
                    content=retrieval_message, elements=image_retrievals
                ).send()
            except json.JSONDecodeError as e:
                raise e
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
        elif isinstance(msg, TextMessage):
            author = msg.source

            if author in ["answer_agent", "revise_answer_agent"]:
                printable_author = "**" + author.replace("_", " ").title() + ":**\n\n"

                content = printable_author + msg.content
                await cl.Message(content=content).send()
                # Send the task termination message.
                # final_message = "Task terminated. "
                # if msg.stop_reason:
                #     final_message += msg.stop_reason
                # await cl.Message(content=final_message).send()
        else:
            # Skip all other message types.
            pass
