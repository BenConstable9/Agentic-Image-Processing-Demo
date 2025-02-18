from typing import List
import json
import chainlit as cl
from autogen_agentchat.messages import (
    ModelClientStreamingChunkEvent,
    TextMessage,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
    TextMessage,
)
from autogen_core import CancellationToken
from rag import Rag
from rat import Rat
import re
from chainlit.input_widget import Select
from figure_processing import get_figures_from_chunk


def remove_markdown_formatting(text: str) -> str:
    """Remove Markdown formatting from text.

    Args:
        text (str): Text to remove formatting from.

    Returns:
        str: Text with formatting removed."""
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

    # Replace all new lines with a space
    text = re.sub(r"\n", " ", text)

    return text


@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    """Start the chat and set the assistant agent in the user session."""
    # Set the assistant agent in the user session.

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
async def handle_agent_update(settings: dict):
    """Handle the agent update in settings."""
    cl.user_session.set("agent", settings["Agent"])  # Store selection in session state


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    """Set the starters for the chat.

    Returns:
        List[cl.Starter]: List of starters."""
    return [
        cl.Starter(
            label="What is the approach for sustainability?",
            message="What is the approach for sustainability?",
        ),
        cl.Starter(
            label="What priority areas will have the most impact on the business and it's stakeholders?",
            message="What priority areas will have the most impact on the business and it's stakeholders?",
        ),
        cl.Starter(
            label="How is Shein innovating?",
            message="How is Shein innovating?",
        ),
        cl.Starter(
            label="How does Shein enforce compliance throughput the supply chain?",
            message="How does Shein enforce compliance throughput the supply chain?",
        ),
        cl.Starter(
            label="What is Shein doing to be more sustainable?",
            message="What is Shein doing to be more sustainable?",
        ),
    ]


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    """Handle the chat messages and populate the UI if needed.

    Args:
        message (cl.Message): Message to handle."""
    # Get the team from the user session.
    agent = cl.user_session.get("agent")  # type: ignore

    if agent == "RAG Agent":
        team = Rag()
    elif agent == "RAT Agent":
        team = Rat()
    else:
        return

    # Streaming response message.
    streaming_response: cl.Message | None = None
    # Stream the messages from the team.
    # Rest the team
    last_chunk_is_image = False

    async for msg in team.group_chat.run_stream(
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
                if "search_term" in args:
                    extracted_search_terms = args["search_term"]
                elif "search_terms" in args:
                    extracted_search_terms = ", ".join(args["search_terms"])

                if extracted_search_terms is not None:
                    await cl.Message(
                        content=f"**Research Agent ({agent}):**\n\nSearching AI Search with: *'{extracted_search_terms}'*"
                    ).send()
            except json.JSONDecodeError:
                pass
        elif isinstance(msg, ToolCallExecutionEvent):
            # Handle the tool call execution.
            ai_search_results = msg.content[0].content
            try:
                results = json.loads(ai_search_results)

                retrieval_message = f"**Research Agent ({agent}):**\n\nRetrieved the following information:"
                image_retrievals = []
                for chunk_id, result in results.items():
                    cleaned_text, chunk_image_retrievals = get_figures_from_chunk(
                        team.figure_and_chunk_pairs, result["Chunk"], chunk_id=chunk_id
                    )

                    image_retrievals.extend(chunk_image_retrievals)

                    first_150_chars = cleaned_text[:150]

                    retrieval_message += (
                        f"\n\n {remove_markdown_formatting(first_150_chars)}... "
                    )

                await cl.Message(
                    content=retrieval_message, elements=image_retrievals
                ).send()
            except json.JSONDecodeError:
                pass
        elif isinstance(msg, ModelClientStreamingChunkEvent):
            # Stream the model client response to the user.
            author = msg.source

            if author in ["answer_agent", "revise_answer_agent"]:
                if streaming_response is None:
                    # Start a new streaming response.
                    streaming_response = cl.Message(content="", author=msg.source)

                    # Stream the printable author
                    printable_author = (
                        "**" + author.replace("_", " ").title() + f" ({agent}):**\n\n"
                    )
                    await streaming_response.stream_token(printable_author)

                    await streaming_response.stream_token(msg.content)

                else:
                    if "<" in msg.content:
                        last_chunk_is_image = True
                        # Split content up to <figure
                        content_split = msg.content.split("<")[0]
                        await streaming_response.stream_token(content_split)
                    elif last_chunk_is_image is False:
                        await streaming_response.stream_token(msg.content)
        elif (
            streaming_response is not None and isinstance(msg, TextMessage)
        ) or isinstance(msg, TextMessage):
            author = msg.source

            if author in ["answer_agent", "revise_answer_agent"]:
                printable_author = (
                    "**" + author.replace("_", " ").title() + f" ({agent}):**\n\n"
                )

                clean_text, image_retrievals = get_figures_from_chunk(
                    team.figure_and_chunk_pairs, msg.content
                )
                cleaned_content = printable_author + clean_text

                if streaming_response is not None:
                    last_chunk_is_image = False
                    streaming_response.content = cleaned_content

                    await streaming_response.send()
                    streaming_response = None
                    if len(image_retrievals) > 0:
                        await cl.Message(content="", elements=image_retrievals).send()
                else:
                    await cl.Message(
                        content=cleaned_content, elements=image_retrievals
                    ).send()

        else:
            # Skip all other message types.
            pass
