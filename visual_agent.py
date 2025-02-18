from typing import AsyncGenerator, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import AgentEvent, ChatMessage, MultiModalMessage
from autogen_core import CancellationToken
import json
from autogen_core import Image
from autogen_agentchat.messages import ToolCallExecutionEvent
from figure_processing import get_figures_from_chunk

import logging
class VisualAgent(AssistantAgent):
    def __init__(
        self,
        name: str,
        description: str,
        system_message: str,
        model_client,
        model_client_stream,
        chunk_and_figure_pairs: dict,
    ):
        super().__init__(
            name=name,
            description=description,
            system_message=system_message,
            model_client=model_client,
            model_client_stream=model_client_stream,
        )

        self.chunk_and_figure_pairs = chunk_and_figure_pairs

    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[AgentEvent | Response, None]:
        # Override and insert the multimodal messages into the context here. This is a work around as AutoGen doesn't support multi-modal tool call responses yet.

        multi_modal_messages = []
        for message in messages:
            if isinstance(message, ToolCallExecutionEvent):
                try:
                    ai_search_results = message.content[0].content
                    results = json.loads(ai_search_results)
                    multi_modal_content = []
                    for chunk_id, result in results.items():
                        cleaned_text, chunk_image_retrievals = get_figures_from_chunk(
                            self.chunk_and_figure_pairs,
                            result["Chunk"],
                            chunk_id=chunk_id,
                        )

                        multi_modal_content.append(cleaned_text)

                        for image in chunk_image_retrievals:
                            multi_modal_content.append(Image.from_base64(image))

                    if len(multi_modal_content) > 0:
                        logging.info("Sending multimodal message")
                        logging.info("Sending %i messages", len(multi_modal_content))
                        multi_modal_messages.append(
                            MultiModalMessage(content=multi_modal_content)
                        )
                except json.JSONDecodeError:
                    multi_modal_messages.append(message)

            else:
                multi_modal_messages.append(message)

        async for event in super().on_messages_stream(
            multi_modal_messages, cancellation_token
        ):
            yield event

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass
