"""SageMaker PyTorch inference handlers for binary dental segmentation."""
import io
import os
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from model import ResUNet

IMAGE_SIZE = 640


def resize_with_padding(
    image,
    target_size=IMAGE_SIZE,
    is_mask=False
):
    width, height = image.size

    scale = min(
        target_size / width,
        target_size / height
    )

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    interpolation = (
        Image.Resampling.NEAREST
        if is_mask
        else Image.Resampling.BILINEAR
    )

    image = image.resize(
        (new_width, new_height),
        interpolation
    )

    canvas = Image.new(
        "L",
        (target_size, target_size),
        color=0
    )

    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2

    canvas.paste(
        image,
        (left, top)
    )

    return canvas

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(model_dir, "resunet_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ResUNet()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    content_type = request_content_type.split(";", 1)[0].strip().lower()
    if content_type not in ("image/jpeg", "image/png"):
        raise ValueError("Content-Type must be image/jpeg or image/png")
    if isinstance(request_body, str):
        request_body = request_body.encode("latin-1")
    elif isinstance(request_body, (bytes, bytearray, memoryview)):
        request_body = bytes(request_body)
    else:
        raise ValueError(
            f"Unsupported request body type: {type(request_body).__name__}"
        )
    try:
        with Image.open(io.BytesIO(request_body)) as source:
            image = source.convert("L")
            image = resize_with_padding(image, target_size=IMAGE_SIZE, is_mask=False)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image") from exc
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def predict_fn(input_data, model):
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(input_data.to(device))
        mask = (torch.sigmoid(logits) >= 0.5).to(torch.uint8)
    return mask.squeeze(0).squeeze(0).cpu().numpy() * 255


def output_fn(prediction, accept):
    accepted = (accept or "image/png").split(";", 1)[0].strip().lower()
    if accepted not in ("image/png", "*/*"):
        raise ValueError("Accept must be image/png")
    buffer = io.BytesIO()
    Image.fromarray(prediction.astype(np.uint8), mode="L").save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"
