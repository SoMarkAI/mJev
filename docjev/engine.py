"""Document-only API over the pinned native HF runtime."""

from pathlib import Path
from PIL import Image
from .io import DEFAULT_CONTEXT
from mjev.families import VL, detect
from mjev.hf import HFMJevEngine
from mjev.prompt import PromptBuilder


class DocJevEngine(HFMJevEngine):
    def __init__(
        self,
        model_path,
        *,
        numerics="native",
        device_map="auto",
        dtype="bfloat16",
        max_input_tokens=4000,
        min_pixels=3136,
        max_pixels=501760,
    ):
        if detect(model_path) != VL:
            raise ValueError("DocJev v0.1 requires a Qwen3-VL checkpoint")
        validate_limits(max_input_tokens, min_pixels, max_pixels)
        super().__init__(
            model_path,
            numerics=numerics,
            device_map=device_map,
            dtype=dtype,
            max_input_tokens=max_input_tokens,
        )
        self.builder.processor.image_processor.size = {
            "shortest_edge": min_pixels,
            "longest_edge": max_pixels,
        }

    def prepare(
        self,
        media,
        question,
        candidates,
        *,
        modality="image",
        context=DEFAULT_CONTEXT,
        video_options=None,
        decoded=None,
    ):
        if modality != "image" or video_options:
            raise ValueError("DocJev v0.1 supports document images only")
        # Match the validated RLCD pipeline exactly: PIL RGB, no silent EXIF rotation.
        if decoded is None:
            with Image.open(Path(media)) as image:
                decoded = {"images": [image.convert("RGB")]}
        return super().prepare(
            media, question, candidates, modality="image", context=context, decoded=decoded
        )

    def score_document(self, document, *, mode="causal", prefix_cache=False, batch_size=1):
        return self.score_many(
            document["image"],
            document["questions"],
            context=document["context"],
            mode=mode,
            projection="full",
            use_prefix_cache=prefix_cache,
            batch_size=batch_size,
        )


def validate_limits(tokens, minimum, maximum):
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in (tokens, minimum, maximum)
        )
        or minimum > maximum
    ):
        raise ValueError("Input limits must be positive integers; min_pixels <= max_pixels")


def preflight(model_path, document, *, max_input_tokens=4000, min_pixels=3136, max_pixels=501760):
    """Check the actual processor/template/token labels without loading model weights."""
    validate_limits(max_input_tokens, min_pixels, max_pixels)
    if detect(model_path) != VL:
        raise ValueError("DocJev v0.1 requires a Qwen3-VL checkpoint")
    engine = DocJevEngine.__new__(DocJevEngine)
    engine.builder = PromptBuilder(model_path)
    engine.max_input_tokens = max_input_tokens
    engine.builder.processor.image_processor.size = {
        "shortest_edge": min_pixels,
        "longest_edge": max_pixels,
    }
    results, decoded = [], None
    for question in document["questions"]:
        _, spec, decoded = engine.prepare(
            document["image"],
            question["question"],
            question["candidates"],
            context=document["context"],
            decoded=decoded,
        )
        results.append(
            {key: spec[key] for key in ("prompt_tokens", "spans", "labels", "label_ids")}
        )
    return results
