import base64
import uuid
from io import BytesIO
from typing import Literal, Sequence

import httpx
from agents import TResponseInputItem
from openai.types.responses import EasyInputMessageParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputContentParam,
)
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from PIL import Image

ROLE = Literal["user", "assistant", "system", "developer"]
Content = str | Image.Image | httpx.URL


def pil2base64(image: Image.Image) -> str:
    """
    Convert a PIL image to a base64-encoded string.
    """
    import io

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def bytes2pil_image(data: bytes) -> Image.Image:
    """
    Convert bytes data to a PIL Image.

    Args:
        data (bytes): The image data in bytes format.

    Returns:
        Image.Image: The converted PIL Image.
    """

    return Image.open(BytesIO(data))


def pil2bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """
    Convert a PIL Image to bytes.

    Args:
        image (Image.Image): The PIL Image to convert.
        format (str): The format to save the image in. Default is "PNG".

    Returns:
        bytes: The image data in bytes format.
    """

    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def content2param(
    item: Content,
) -> ResponseInputContentParam:
    """コンテンツをResponseInputParamに変換するヘルパー関数

    Args:
        item: 文字列、PIL画像、またはhttpx.URLのいずれか

    Returns:
        TResponseInputItem: ResponseInputTextParamまたはResponseInputImageParam
    """
    if isinstance(item, str):
        return ResponseInputTextParam(type="input_text", text=item)
    elif isinstance(item, Image.Image):
        # 画像をBase64データURLに変換
        base64_data = pil2base64(item)
        return ResponseInputImageParam(
            type="input_image",
            detail="high",
            image_url=f"data:image/png;base64,{base64_data}",
        )
    elif isinstance(item, httpx.URL):
        return ResponseInputImageParam(
            type="input_image",
            detail="high",
            image_url=str(item),
        )
    else:
        raise ValueError(f"Unsupported item type: {type(item)}")


def contents2params(
    role: ROLE,
    items: Sequence[Content],
) -> list[TResponseInputItem]:
    params: list[ResponseInputContentParam] = []
    for item in items:
        params.append(content2param(item))
    if not params:
        raise ValueError("At least one item must be provided")
    return [
        EasyInputMessageParam(
            role=role,
            type="message",
            content=params,
        )
    ]  # type: ignore


def pil2bytesio(
    image: Image.Image, format: str = "PNG", filename: str | None = None
) -> BytesIO:
    if filename is None:
        filename = uuid.uuid4().hex + "." + format.lower()
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)  # Reset the buffer position to the beginning
    buffer.name = filename  # Set the name attribute for identification
    return buffer
