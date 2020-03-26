from typing import Generator, Union

import numpy
from PIL import Image


def ale_to_ram(ale) -> numpy.ndarray:
    """Return the ram of the ale emulator."""
    ram_size = ale.getRAMSize()
    ram = numpy.zeros(ram_size, dtype=numpy.uint8)
    ale.getRAM(ram)
    return ram


def resize_frame(
    frame: numpy.ndarray, width: int, height: int, mode: str = "RGB"
) -> numpy.ndarray:
    """
    Use PIL to resize an RGB frame to an specified height and width.

    Args:
        frame: Target numpy array representing the image that will be resized.
        width: Width of the resized image.
        height: Height of the resized image.
        mode: Passed to Image.convert.

    Returns:
        The resized frame that matches the provided width and height.

    """
    frame = Image.fromarray(frame)
    frame = frame.convert(mode).resize(size=(width, height))
    return numpy.array(frame)


def split_similar_chunks(
    vector: Union[list, numpy.ndarray], n_chunks: int
) -> Generator[Union[list, numpy.ndarray], None, None]:
    """
    Split an indexable object into similar chunks.

    Args:
        vector: Target object to be split.
        n_chunks: Number of similar chunks.

    Returns:
        Generator that returns the chunks created after splitting the target object.

    """
    chunk_size = int(numpy.ceil(len(vector) / n_chunks))
    for i in range(0, len(vector), chunk_size):
        yield vector[i : i + chunk_size]
