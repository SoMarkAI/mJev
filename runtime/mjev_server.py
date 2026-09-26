#!/usr/bin/env python3
"""Native-template Qwen3-Omni multi-question API.

The causal path defaults to native concurrent question rows.  Masked mode
crosses the frontend/EngineCore boundary once and is expanded by the scheduler
into media -> question -> candidate branches followed by Tree-KV decision
rows.  Both paths use the checkpoint's native template and compact label-only
projection.
"""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any

import uvloop
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.utils.argparse_utils import FlexibleArgumentParser

from mjev_contract import (
    LabelSet,
    MAX_QUESTIONS,
    MIN_QUESTIONS,
    MODEL_NAME,
    VALID_MEDIA_TYPES,
    VALID_MODES,
    build_question_text,
    common_prefix_length,
    sha256_bytes,
    validate_single_token_labels,
)
from mjev_mask import MaskedQuestionLayout, build_masked_layout
from mjev_scoring import score_answer


MAX_MEDIA_BYTES = int(os.environ.get("MJEV_MAX_MEDIA_BYTES", str(512 * 1024 * 1024)))


class MediaItem(BaseModel):
    type: str
    url: str = Field(min_length=1)
    uuid: str | None = None

    @model_validator(mode="after")
    def validate_type(self) -> "MediaItem":
        if self.type not in VALID_MEDIA_TYPES:
            raise ValueError(f"media type must be one of {sorted(VALID_MEDIA_TYPES)}")
        return self


class Question(BaseModel):
    id: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] = Field(min_length=2, max_length=26)

    @model_validator(mode="after")
    def validate_labels(self) -> "Question":
        # build_question_text owns the exact shared validation contract.
        build_question_text(self.instructions, self.criteria)
        return self


class MultiQuestionRequest(BaseModel):
    model: str = MODEL_NAME
    mode: str = "causal"
    use_audio_in_video: bool = False
    media: list[MediaItem] = Field(default_factory=list, max_length=3)
    questions: list[Question] = Field(min_length=MIN_QUESTIONS, max_length=MAX_QUESTIONS)

    @model_validator(mode="after")
    def validate_request(self) -> "MultiQuestionRequest":
        if self.model != MODEL_NAME:
            raise ValueError(f"model must be {MODEL_NAME}")
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        media_types = [item.type for item in self.media]
        if len(media_types) != len(set(media_types)):
            raise ValueError("at most one image, video and audio are allowed")
        ids = [question.id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        return self


class UUIDHashRegistry:
    """Small bounded registry preventing stale UUID/content aliasing."""

    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = capacity
        self.values: OrderedDict[str, str] = OrderedDict()
        self.lock = asyncio.Lock()

    async def check(self, stable_id: str, digest: str) -> None:
        async with self.lock:
            previous = self.values.get(stable_id)
            if previous is not None and previous != digest:
                raise HTTPException(409, f"media UUID {stable_id!r} refers to changed content")
            self.values[stable_id] = digest
            self.values.move_to_end(stable_id)
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)


UUID_REGISTRY = UUIDHashRegistry()
CAUSAL_TOPOLOGY = os.environ.get("MJEV_CAUSAL_TOPOLOGY", "B").upper()
if CAUSAL_TOPOLOGY not in {"A", "B"}:
    raise RuntimeError("MJEV_CAUSAL_TOPOLOGY must be A or B")
KV_BLOCK_SIZE = int(os.environ.get("MJEV_BLOCK_SIZE", "16"))
if KV_BLOCK_SIZE not in {16, 32}:
    raise RuntimeError("MJEV_BLOCK_SIZE must be 16 or 32")


def _read_media(url: str) -> tuple[bytes, str, bool]:
    from mjev_media import read_media
    roots = [value for value in os.environ.get('MJEV_ALLOWED_MEDIA_ROOTS', '/data').split(os.pathsep) if value]
    hosts = [value.strip().lower() for value in os.environ.get('MJEV_ALLOWED_MEDIA_HOSTS', '').split(',') if value.strip()]
    return read_media(url, limit=MAX_MEDIA_BYTES, roots=roots, allowed_hosts=hosts)


async def normalize_media(items: list[MediaItem]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from mjev_media import MediaTooLarge
    started = time.perf_counter()
    content: list[dict[str, Any]] = []
    total_bytes = 0
    for item in items:
        try:
            raw, mime, remote = await asyncio.to_thread(_read_media, item.url)
        except MediaTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(422, f"cannot read {item.type}: {exc}") from exc
        if len(raw) > MAX_MEDIA_BYTES:
            raise HTTPException(413, f"{item.type} exceeds {MAX_MEDIA_BYTES} bytes")
        digest = sha256_bytes(raw)
        stable_id = item.uuid or digest
        await UUID_REGISTRY.check(stable_id, digest)
        # Pass the exact validated bytes to vLLM. Never reopen a local path or
        # refetch a URL after hashing and checking it (TOCTOU / UUID mismatch).
        url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        media_part = {"type": item.type, item.type: {"url": url, "uuid": stable_id}}
        # Qwen3-Omni's native template recognizes video content through the
        # ``video`` field (its template intentionally does not treat
        # ``video_url`` as a video placeholder).  Keep the OpenAI-compatible
        # ``video_url`` field for vLLM's media parser and add the alias only
        # for template rendering; this avoids changing the public API or
        # introducing a second media item/fetch.
        if item.type == "video_url":
            media_part["video"] = {"url": url, "uuid": stable_id}
        content.append(media_part)
        total_bytes += len(raw)
    return content, {
        "read_ms": round((time.perf_counter() - started) * 1000, 3),
        "bytes": total_bytes,
    }


def _error_detail(error: ErrorResponse) -> str:
    value = getattr(error, "error", None)
    return str(getattr(value, "message", value or error))


def _extract_raw_scores(output: Any, token_ids: tuple[int, ...]) -> dict[int, float]:
    """Decode raw selected logits carried in vLLM's compact logprob envelope."""
    steps = getattr(output, "logprobs", None)
    if not steps:
        raise RuntimeError("selective sampler returned no score envelope")
    step = steps[0]
    scores: dict[int, float] = {}
    if isinstance(step, dict):
        for token_id in token_ids:
            value = step.get(token_id)
            if value is None:
                continue
            scores[token_id] = float(getattr(value, "logprob", value))
    else:
        ids = getattr(step, "logprob_token_ids", None)
        values = getattr(step, "logprobs", None)
        if ids is not None and values is not None:
            for token_id, value in zip(ids, values):
                if int(token_id) in token_ids:
                    scores[int(token_id)] = float(value)
    missing = set(token_ids) - set(scores)
    if missing:
        raise RuntimeError(f"selective sampler omitted label tokens {sorted(missing)}")
    return scores


def attach_router(app: FastAPI) -> None:
    from mjev_vllm_patch import require_installed
    require_installed()
    router = APIRouter()

    @router.get('/v1/mjev/health')
    async def mjev_health():
        try:
            return {'runtime_hooks': require_installed()}
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc


    @router.post("/v1/mjev/multi_question")
    async def multi_question(payload: MultiQuestionRequest, raw_request: Request):
        require_installed()
        request_started = time.perf_counter()
        handler = raw_request.app.state.openai_serving_chat
        if handler is None:
            raise HTTPException(501, "model does not support chat completions")
        model_error = await handler._check_model(
            ChatCompletionRequest(model=payload.model, messages=[])
        )
        if model_error is not None:
            raise HTTPException(model_error.error.code, _error_detail(model_error))

        media_content, media_stats = await normalize_media(payload.media)
        renderer = handler.online_renderer
        tokenizer = renderer.renderer.tokenizer
        checkpoint_labels = validate_single_token_labels(tokenizer)
        group = uuid.uuid4().hex

        chat_requests: list[ChatCompletionRequest] = []
        label_sets: list[LabelSet] = []
        for question in payload.questions:
            messages = [
                {
                    "role": "user",
                    "content": [
                        *media_content,
                        {"type": "text", "text": build_question_text(question.instructions, question.criteria)},
                    ],
                }
            ]
            label_set = LabelSet.for_criteria(question.criteria, checkpoint_labels)
            label_sets.append(label_set)
            chat_requests.append(
                ChatCompletionRequest(
                    model=payload.model,
                    messages=messages,
                    mm_processor_kwargs={"use_audio_in_video": payload.use_audio_in_video},
                    chat_template_kwargs={"use_audio_in_video": payload.use_audio_in_video},
                    temperature=0,
                    seed=0,
                    max_tokens=1,
                    allowed_token_ids=list(label_set.token_ids),
                    logprobs=True,
                    top_logprobs=min(len(label_set.token_ids), 20),
                    cache_salt=f"mjev:{group}",
                )
            )
        batch_label_token_ids = sorted(
            {token_id for label_set in label_sets for token_id in label_set.token_ids}
        )

        async def preprocess_one(index: int):
            request = chat_requests[index]
            return await renderer.preprocess_chat(
                request,
                request.messages,
                default_template=renderer.chat_template,
                default_template_content_format=renderer.chat_template_content_format,
                default_template_kwargs=renderer.default_chat_template_kwargs,
                tool_dicts=None,
                parser=renderer.parser,
                skip_mm_cache=False,
            )

        preprocess_started = time.perf_counter()
        # Row zero fills the processor/media cache. The remaining rows then
        # resolve the same stable media hashes without a cache stampede.
        first = await preprocess_one(0)
        rest = await asyncio.gather(
            *(preprocess_one(index) for index in range(1, len(chat_requests)))
        )
        processed = [first, *rest]
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

        lora_request = handler._maybe_get_adapters(
            chat_requests[0], supports_default_mm_loras=True
        )
        data_parallel_rank = handler._get_data_parallel_rank(raw_request)
        trace_headers = await handler._get_trace_headers(raw_request.headers)

        async def run_row(index: int) -> tuple[Any, float]:
            request = chat_requests[index]
            _, prompts = processed[index]
            engine_prompt = prompts[0]
            sampling_params = request.to_sampling_params(
                1, handler.default_sampling_params
            )
            # One sampled column plus at most 26 labels. This remains a tiny
            # compact envelope and avoids ragged top-logprob truncation.
            sampling_params.logprobs = 27
            sampling_params.logprob_token_ids = None
            encoded_ids = "_".join(
                str(token_id) for token_id in label_sets[index].token_ids
            )
            request_id = f"mjev-label-{encoded_ids}--{group}-{index}"
            started = time.perf_counter()
            generator = handler.engine_client.generate(
                engine_prompt,
                sampling_params,
                request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=0,
                data_parallel_rank=data_parallel_rank,
                reasoning_ended=None,
            )
            final = None
            try:
                async for result in generator:
                    final = result
                    if await raw_request.is_disconnected():
                        await generator.aclose()
                        raise HTTPException(408, "client disconnected")
            except asyncio.CancelledError:
                raise HTTPException(408, "client disconnected") from None
            if final is None or not final.outputs:
                raise RuntimeError("EngineCore returned no output")
            return final.outputs[0], (time.perf_counter() - started) * 1000

        engine_started = time.perf_counter()
        answers = []
        row_ms: list[float] | None = None
        fork_tokens: int | None = None
        tree_kv_stats: dict[str, Any] | None = None
        effective_topology = "tree-kv" if payload.mode == "masked" else (
            "B-single"
            if CAUSAL_TOPOLOGY == "A" and len(chat_requests) == 1
            else CAUSAL_TOPOLOGY
        )
        try:
            if payload.mode == "masked":
                layouts: list[MaskedQuestionLayout] = []
                for question, (_, prompts) in zip(payload.questions, processed):
                    prompt = prompts[0]
                    values = (
                        prompt.get("prompt_token_ids")
                        if isinstance(prompt, dict)
                        else getattr(prompt, "prompt_token_ids", None)
                    )
                    if not isinstance(values, list):
                        raise RuntimeError("renderer did not return prompt_token_ids")
                    layouts.append(
                        build_masked_layout(
                            values,
                            tokenizer,
                            question.instructions,
                            question.criteria,
                        )
                    )

                compact_roots = [layout.compact_candidate(0) for layout in layouts]
                root_prefix = (
                    common_prefix_length(compact_roots)
                    if len(compact_roots) > 1
                    else 0
                )
                if len(compact_roots) > 1:
                    root_prefix = min(
                        root_prefix,
                        min(layout.plan.common.end for layout in layouts) - 1,
                    )
                    if root_prefix <= 0:
                        raise RuntimeError("masked questions have no shared media prefix")
                label_rows = [list(label_set.token_ids) for label_set in label_sets]
                internal_request = chat_requests[0].model_copy(
                    update={
                        "max_tokens": 1,
                        "allowed_token_ids": sorted(
                            {token for row in label_rows for token in row}
                        ),
                        "vllm_xargs": {
                            "mjev_protocol": "mjev-masked-v1",
                            "questions": [layout.to_payload() for layout in layouts],
                            "label_token_ids": label_rows,
                            "all_label_token_ids": batch_label_token_ids,
                            "root_prefix_tokens": root_prefix,
                            "block_size": KV_BLOCK_SIZE,
                        },
                    }
                )
                sampling_params = internal_request.to_sampling_params(
                    1, handler.default_sampling_params
                )
                sampling_params.logprobs = 27
                sampling_params.logprob_token_ids = None
                engine_prompt = processed[0][1][0]
                if not isinstance(engine_prompt, dict):
                    raise RuntimeError("masked mode requires a dictionary engine prompt")
                engine_prompt = dict(engine_prompt)
                engine_prompt["prompt_token_ids"] = compact_roots[0]
                encoded = "_".join(str(value) for value in batch_label_token_ids)
                generator = handler.engine_client.generate(
                    engine_prompt,
                    sampling_params,
                    f"mjev-label-{encoded}--masked-{group}",
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=0,
                    data_parallel_rank=data_parallel_rank,
                    reasoning_ended=None,
                )
                final = None
                async for result in generator:
                    final = result
                    if await raw_request.is_disconnected():
                        await generator.aclose()
                        raise HTTPException(408, "client disconnected")
                if final is None or not final.outputs:
                    raise RuntimeError("masked EngineCore request returned no output")
                output = final.outputs[0]
                stop_reason = getattr(output, "stop_reason", None)
                prefix = "mjev-masked-scores:"
                if not isinstance(stop_reason, str) or not stop_reason.startswith(prefix):
                    raise RuntimeError("masked EngineCore request returned no Tree-KV payload")
                packed_payload = json.loads(stop_reason.removeprefix(prefix))
                packed_scores = packed_payload["scores"]
                tree_kv_stats = packed_payload.get("tree_kv")
                if len(packed_scores) != len(payload.questions):
                    raise RuntimeError("masked EngineCore score count mismatch")
                for question, label_set, packed in zip(
                    payload.questions, label_sets, packed_scores
                ):
                    raw_by_token = {
                        int(token_id): float(value)
                        for token_id, value in packed.items()
                    }
                    scores = {
                        label: raw_by_token[token_id]
                        for label, token_id in zip(
                            label_set.labels, label_set.token_ids
                        )
                    }
                    answer = max(scores, key=scores.__getitem__)
                    answers.append(
                        {"id": question.id, "answer": answer, "scores": scores}
                    )
                fork_tokens = root_prefix
            elif effective_topology != "A":
                row_results = await asyncio.gather(
                    *(run_row(index) for index in range(len(chat_requests)))
                )
                row_ms = []
                for question, label_set, (output, latency) in zip(
                    payload.questions, label_sets, row_results
                ):
                    raw_by_token = _extract_raw_scores(output, label_set.token_ids)
                    scores = {
                        label: raw_by_token[token_id]
                        for label, token_id in zip(label_set.labels, label_set.token_ids)
                    }
                    answer = max(scores, key=scores.__getitem__)
                    answers.append({"id": question.id, "answer": answer, "scores": scores})
                    row_ms.append(round(latency, 3))
            else:
                token_rows = []
                for _, prompts in processed:
                    prompt = prompts[0]
                    values = prompt.get("prompt_token_ids") if isinstance(prompt, dict) else getattr(prompt, "prompt_token_ids", None)
                    if not isinstance(values, list):
                        raise RuntimeError("renderer did not return prompt_token_ids")
                    token_rows.append(values)
                label_rows = [list(label_set.token_ids) for label_set in label_sets]
                common_tokens = common_prefix_length(token_rows)
                fork_tokens = common_tokens // KV_BLOCK_SIZE * KV_BLOCK_SIZE
                if fork_tokens <= 0 or fork_tokens >= min(map(len, token_rows)):
                    raise RuntimeError("question rows have no safe aligned common prefix")
                internal_request = chat_requests[0].model_copy(
                    update={
                        "max_tokens": len(chat_requests),
                        "allowed_token_ids": sorted({token for row in label_rows for token in row}),
                        "vllm_xargs": {
                            "mjev_protocol": "mjev-causal-a-v1",
                            "prompt_token_ids": token_rows,
                            "label_token_ids": label_rows,
                            "all_label_token_ids": batch_label_token_ids,
                            "fork_prefix_tokens": fork_tokens,
                            "block_size": KV_BLOCK_SIZE,
                        },
                    }
                )
                sampling_params = internal_request.to_sampling_params(
                    1, handler.default_sampling_params
                )
                # The parent Request is instantiated before scheduler-side
                # row expansion, so reserve the sampled-token slot here too.
                sampling_params.logprobs = 27
                sampling_params.logprob_token_ids = None
                first_encoded = "_".join(
                    str(value) for value in batch_label_token_ids
                )
                generator = handler.engine_client.generate(
                    processed[0][1][0],
                    sampling_params,
                    f"mjev-label-{first_encoded}--a-{group}",
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=0,
                    data_parallel_rank=data_parallel_rank,
                    reasoning_ended=None,
                )
                final = None
                async for result in generator:
                    final = result
                    if await raw_request.is_disconnected():
                        await generator.aclose()
                        raise HTTPException(408, "client disconnected")
                if final is None or not final.outputs:
                    raise RuntimeError("single EngineCore request returned no output")
                output = final.outputs[0]
                stop_reason = getattr(output, "stop_reason", None)
                if not isinstance(stop_reason, str) or not stop_reason.startswith("mjev-scores:"):
                    raise RuntimeError("single EngineCore request returned no score payload")
                packed_scores = json.loads(stop_reason.removeprefix("mjev-scores:"))
                if len(packed_scores) != len(payload.questions):
                    raise RuntimeError("single EngineCore score count mismatch")
                for question, label_set, packed in zip(payload.questions, label_sets, packed_scores):
                    raw_by_token = {int(token_id): float(value) for token_id, value in packed.items()}
                    scores = {
                        label: raw_by_token[token_id]
                        for label, token_id in zip(label_set.labels, label_set.token_ids)
                    }
                    answer = max(scores, key=scores.__getitem__)
                    answers.append({"id": question.id, "answer": answer, "scores": scores})
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"EngineCore execution failed: {exc}") from exc
        engine_ms = (time.perf_counter() - engine_started) * 1000

        answers = [score_answer(a["id"], a["scores"]) for a in answers]
        total_ms = (time.perf_counter() - request_started) * 1000
        return {
            "answers": answers,
            "timing_ms": {
                "media_preprocess": media_stats["read_ms"],
                "media_encoder": None,
                "common_prefill": None,
                "question_prefill": round(engine_ms, 3),
                "candidate_prefill": None if payload.mode == "masked" else 0.0,
                "kv_merge": None if payload.mode == "masked" else 0.0,
                "label_projection": None,
                "engine_total": round(engine_ms, 3),
                "end_to_end": round(total_ms, 3),
                "native_processor": round(preprocess_ms, 3),
                "question_rows": row_ms,
            },
            "diagnostics": {
                "mode": payload.mode,
                "topology": effective_topology,
                "public_api_requests": 1,
                "engine_requests": (
                    1
                    if payload.mode == "masked" or effective_topology == "A"
                    else len(payload.questions)
                ),
                "shared_kv_prefix_tokens": fork_tokens or 0,
                "candidate_rows": (
                    sum(len(question.criteria) for question in payload.questions)
                    if payload.mode == "masked"
                    else 0
                ),
                "tree_kv": tree_kv_stats,
                "native_chat_template": True,
                "system_prompt": False,
                "score_semantics": "raw_label_logits",
                "media_bytes": media_stats["bytes"],
                "label_token_ids": checkpoint_labels,
            },
        }

    app.include_router(router)


def install_api() -> None:
    if getattr(api_server, "_mjev_api_installed", False):
        return
    original_build_app = api_server.build_app

    def build_app(*args: Any, **kwargs: Any) -> FastAPI:
        app = original_build_app(*args, **kwargs)
        attach_router(app)
        return app

    api_server.build_app = build_app
    api_server._mjev_api_installed = True


def main() -> None:
    # Never rely solely on sitecustomize being found on PYTHONPATH.
    os.environ['MJEV_ENABLE_PATCHES'] = '1'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    runtime = str(Path(__file__).resolve().parent)
    os.environ['PYTHONPATH'] = runtime + os.pathsep + os.environ.get('PYTHONPATH', '')
    from mjev_vllm_patch import install, require_installed
    install()
    require_installed()
    install_api()
    api_server.cli_env_setup()
    parser = FlexibleArgumentParser(description="vLLM server with mJev API")
    parser = api_server.make_arg_parser(parser)
    args = parser.parse_args()
    api_server.validate_parsed_serve_args(args)
    uvloop.run(api_server.run_server(args))


if __name__ == "__main__":
    main()
