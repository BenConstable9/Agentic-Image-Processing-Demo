import chainlit as cl
import re


def get_figures_from_chunk(
    chunk_and_figure_pairs: dict,
    text: str,
    chunk_id: str = None,
    cast_to_chainlit_image: bool = True,
) -> tuple[str, list[cl.Image]]:
    """Extract figures from a chunk of text and load them into the assistant.

    Args:
        text (str): Text to extract figures from.
        chunk_id (str, optional): Chunk ID to extract figures for. Defaults to None.

    Returns:
        tuple[str, list[cl.Image]]: Tuple containing the cleaned text and a list of images.
    """

    if chunk_id is None:
        # Regex pattern to extract figure chunk_id and figure_id
        pattern = r"<figure\s+ChunkId='(.*?)'\s+FigureId='(.*?)'>"

        # Find all matches and convert to dictionary
        figure_dict = {match[1]: match[0] for match in re.findall(pattern, text)}
    else:
        figure_ids = re.findall(r"FigureId='(.*?)'", text)

        figure_dict = {figure_id: chunk_id for figure_id in figure_ids}

    # Replace all figure placeholders with the actual image
    image_retrievals = []

    for figure_id, chunk_id in figure_dict.items():
        if (
            chunk_id in chunk_and_figure_pairs
            and figure_id in chunk_and_figure_pairs[chunk_id]
        ):
            image_data = chunk_and_figure_pairs[chunk_id][figure_id]

            if cast_to_chainlit_image:
                image = cl.Image(
                    content=image_data,
                    name=f"Figure {figure_id}",
                    display="inline",
                )
            else:
                image = image_data
            image_retrievals.append(image)

    cleaned_text = re.sub(r"<figure\s+[^>]*>", "", text)

    return cleaned_text, image_retrievals
