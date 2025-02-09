from typing import List, cast
import json
import chainlit as cl
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
from tools import FIGURE_AND_CHUNK_PAIRS


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

def get_figures_from_chunk(text:str, chunk_id: str=None) -> tuple[str, list[cl.Image]]:
    """Extract figures from a chunk of text and load them into the assistant.
    
    Args:
        text (str): Text to extract figures from.
        chunk_id (str, optional): Chunk ID to extract figures for. Defaults to None.
        
    Returns:
        tuple[str, list[cl.Image]]: Tuple containing the cleaned text and a list of images."""
    
    if chunk_id is None:
        # Regex pattern to extract figure chunk_id and figure_id
        pattern = r"<figure\s+chunk_id='(.*?)'\s+figure_id='(.*?)'>"

        # Find all matches and convert to dictionary
        figure_dict = {match[1]: match[0] for match in re.findall(pattern, text)}
    else:
        figure_ids = re.findall(r"FigureId='(.*?)'", text)

        figure_dict = {figure_id: chunk_id for figure_id in figure_ids}

    # Replace all figure placeholders with the actual image
    image_retrievals = []

    for figure_id, chunk_id in figure_dict.items():
        if chunk_id in FIGURE_AND_CHUNK_PAIRS and figure_id in FIGURE_AND_CHUNK_PAIRS[chunk_id]:
            image_data = FIGURE_AND_CHUNK_PAIRS[chunk_id][figure_id]
            image = cl.Image(
                content=image_data,
                name=f"Figure {figure_id}",
                display="inline",
            )
            image_retrievals.append(image)

    cleaned_text = re.sub(r"<figure\s+[^>]*>", "", text)

    return cleaned_text, image_retrievals

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
            label="How is Shein innovating?",
            message="How is Shein innovating?",
        ),
        cl.Starter(
            label="How does Shein enforce compliance throughput the supply chain?",
            message="How does Shein enforce compliance throughput the supply chain?",
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
    """Handle the chat messages and populate the UI if needed.
    
    Args:
        message (cl.Message): Message to handle."""
    # Get the team from the user session.
    agent = cl.user_session.get("agent")  # type: ignore

    if agent == "RAG Agent":
        team = RAG_GROUP_CHAT
    elif agent == "RAT Agent":
        team = RAT_GROUP_CHAT
    else:
        return

    # Streaming response message.
    streaming_response: cl.Message | None = None
    # Stream the messages from the team.
    # Rest the team

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

                retrieval_message = (
                    f"**Research Agent ({agent}):**\n\nRetrieved the following information:"
                )
                image_retrievals = []
                for chunk_id, result in results.items():
                    cleaned_text, chunk_image_retrievals = get_figures_from_chunk(result["Chunk"], chunk_id=chunk_id)

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
                printable_author = "**" + author.replace("_", " ").title() + f" ({agent}):**\n\n"

                text = msg.content

                clean_text, image_retrievals = get_figures_from_chunk(text)
                content = printable_author + clean_text
                await cl.Message(content=content, elements=image_retrievals).send()

        else:
            # Skip all other message types.
            pass

    await team.reset()