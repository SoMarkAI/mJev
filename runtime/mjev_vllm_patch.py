"""Opt-in vLLM patches for compact label-only projection on TP workers.

The patch is intentionally narrow: it activates only when every sampled row
in a scheduler batch has a ``mjev-label-...`` request id. Ordinary vLLM
requests retain the stock full-vocabulary path.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from mjev_mask import MaskedQuestionLayout


_REQUEST_RE = re.compile(r"(?:^|::)mjev-label-(\d+(?:_\d+)*)--")
_TOPOLOGY_A_PROTOCOL = "mjev-causal-a-v1"
_MASKED_PROTOCOL = "mjev-masked-v1"
_MASKED_ROW_PROTOCOL = "mjev-masked-row-v1"


def _request_token_rows(req_ids: list[str | None]) -> list[tuple[int, ...]] | None:
    rows: list[tuple[int, ...]] = []
    for request_id in req_ids:
        if not request_id:
            continue
        match = _REQUEST_RE.search(request_id)
        if match is None:
            return None
        token_ids = tuple(int(value) for value in match.group(1).split("_"))
        if not 2 <= len(token_ids) <= 26 or len(set(token_ids)) != len(token_ids):
            raise RuntimeError("invalid mJev label token metadata")
        rows.append(token_ids)
    return rows or None


def _selective_tp_logits(model: Any, hidden_states: Any, token_ids: tuple[int, ...]):
    """Compute only selected vocabulary rows, then reduce the tiny TP result."""
    import torch
    import torch.nn.functional as functional
    from vllm.distributed import tensor_model_parallel_all_reduce

    language_model = getattr(model, "language_model", model)
    lm_head = getattr(language_model, "lm_head", None)
    processor = getattr(language_model, "logits_processor", None)
    if lm_head is None or processor is None:
        raise RuntimeError("mJev could not locate the language LM head")
    quant_method = getattr(lm_head, "quant_method", None)
    if quant_method is None or type(quant_method).__name__ != "UnquantizedEmbeddingMethod":
        raise RuntimeError("mJev label projection currently requires BF16 weights")

    shard = getattr(lm_head, "shard_indices", None)
    if shard is None:
        raise RuntimeError("mJev LM head has no TP vocabulary shard metadata")
    start = int(shard.org_vocab_start_index)
    end = int(shard.org_vocab_end_index)
    device = hidden_states.device
    result = torch.zeros(
        (hidden_states.shape[0], len(token_ids)),
        dtype=torch.float32,
        device=device,
    )
    owned_columns = [index for index, token_id in enumerate(token_ids) if start <= token_id < end]
    if owned_columns:
        local_ids = torch.tensor(
            [token_ids[index] - start for index in owned_columns],
            dtype=torch.long,
            device=device,
        )
        selected_weight = lm_head.weight.index_select(0, local_ids)
        selected_bias = None
        if getattr(lm_head, "bias", None) is not None:
            selected_bias = lm_head.bias.index_select(0, local_ids).float()
        local_logits = functional.linear(
            hidden_states.float(), selected_weight.float(), selected_bias
        )
        result[:, owned_columns] = local_logits
    if int(getattr(lm_head, "tp_size", 1)) > 1:
        result = tensor_model_parallel_all_reduce(result)
    soft_cap = getattr(processor, "soft_cap", None)
    if soft_cap is not None:
        result = torch.tanh(result / soft_cap) * soft_cap
    scale = float(getattr(processor, "scale", 1.0))
    if scale != 1.0:
        result = result * scale
    return result


def _install_legacy_v1_runner() -> None:
    """Patch the runner selected by the supported vLLM 0.25.1 runtime."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_mjev_labels_installed", False):
        return
    original_execute_model = GPUModelRunner.execute_model
    original_sample = GPUModelRunner._sample

    def execute_model(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any):
        request_rows = _request_token_rows(list(scheduler_output.num_scheduled_tokens))
        if request_rows is None:
            return original_execute_model(self, scheduler_output, *args, **kwargs)
        original_compute_logits = self.model.compute_logits

        def compact_logits(hidden_states: Any):
            # _update_states() may reorder scheduled requests into InputBatch
            # order before logits are gathered. Read that final order here,
            # immediately before projection, so score rows cannot be assigned
            # to the wrong ragged question.
            current_rows = _request_token_rows(
                list(self.input_batch.req_ids[: self.input_batch.num_reqs])
            )
            if current_rows is None or len(current_rows) != hidden_states.shape[0]:
                raise RuntimeError("mJev compact row map differs from hidden states")
            current_union = tuple(
                dict.fromkeys(token for row in current_rows for token in row)
            )
            self._mjev_pending_rows = current_rows
            return _selective_tp_logits(self.model, hidden_states, current_union)

        self.model.compute_logits = compact_logits
        # Preserve the scheduler's exact sampled-row order until _sample().
        # input_batch.req_ids also contains unscheduled active requests and is
        # therefore not a valid row map after explicit prefix branching.
        try:
            return original_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            self.model.compute_logits = original_compute_logits

    def compact_sample(self: Any, logits: Any, spec_decode_metadata: Any):
        rows = getattr(self, "_mjev_pending_rows", None)
        if rows is None:
            return original_sample(self, logits, spec_decode_metadata)
        self._mjev_pending_rows = None
        union = tuple(dict.fromkeys(token for row in rows for token in row))
        if spec_decode_metadata is not None:
            raise RuntimeError("mJev label scoring does not support speculative decoding")

        import torch
        from vllm.v1.outputs import LogprobsTensors, SamplerOutput

        if logits.ndim != 2 or logits.shape[1] != len(union):
            raise RuntimeError("unexpected compact label-logit shape")
        if logits.shape[0] != len(rows):
            raise RuntimeError("compact label rows do not match active requests")
        union_tensor = torch.tensor(union, dtype=torch.int32, device=logits.device)
        union_column = {token_id: index for index, token_id in enumerate(union)}
        allowed = torch.zeros_like(logits, dtype=torch.bool)
        for row_index, token_row in enumerate(rows):
            columns = torch.tensor(
                [union_column[token_id] for token_id in token_row],
                dtype=torch.long,
                device=logits.device,
            )
            allowed[row_index, columns] = True
        selected_column = logits.masked_fill(~allowed, float("-inf")).argmax(dim=-1)
        sampled = union_tensor[selected_column]
        selected_values = logits.gather(1, selected_column[:, None])

        # LogprobsTensors is used only as a compact transport envelope here.
        # The custom API interprets these K values as raw logits, as required.
        envelope_width = 27
        output_token_ids = sampled[:, None].expand(-1, envelope_width).clone()
        output_values = selected_values.expand(-1, envelope_width).clone()
        output_token_ids[:, 0] = sampled
        output_values[:, 0] = selected_values[:, 0]
        for row_index, token_row in enumerate(rows):
            count = len(token_row)
            columns = torch.tensor(
                [union_column[token_id] for token_id in token_row],
                dtype=torch.long,
                device=logits.device,
            )
            output_token_ids[row_index, 1 : count + 1] = torch.tensor(
                token_row, dtype=torch.int32, device=logits.device
            )
            output_values[row_index, 1 : count + 1] = logits[row_index].index_select(
                0, columns
            ).float()
        ranks = ((logits >= selected_values) & allowed).sum(dim=-1).to(torch.int32)
        return SamplerOutput(
            sampled_token_ids=sampled[:, None],
            logprobs_tensors=LogprobsTensors(
                logprob_token_ids=output_token_ids,
                logprobs=output_values,
                selected_token_ranks=ranks,
                cu_num_generated_tokens=None,
            ),
        )

    GPUModelRunner.execute_model = execute_model
    GPUModelRunner._sample = compact_sample
    GPUModelRunner._mjev_labels_installed = True


def _extra(request: Any) -> dict[str, Any]:
    params = getattr(request, "sampling_params", None)
    value = getattr(params, "extra_args", None)
    return value if isinstance(value, dict) else {}


def _topology_a_meta(
    request: Any,
) -> tuple[list[list[int]], list[list[int]], list[int], int] | None:
    extra = _extra(request)
    if extra.get("mjev_protocol") != _TOPOLOGY_A_PROTOCOL:
        return None
    token_rows = extra.get("prompt_token_ids")
    label_rows = extra.get("label_token_ids")
    all_labels = extra.get("all_label_token_ids")
    prefix = extra.get("fork_prefix_tokens")
    block_size = extra.get("block_size")
    valid = (
        isinstance(token_rows, list)
        and 1 <= len(token_rows) <= 6
        and all(
            isinstance(row, list)
            and row
            and all(isinstance(token, int) and not isinstance(token, bool) for token in row)
            for row in token_rows
        )
        and isinstance(label_rows, list)
        and len(label_rows) == len(token_rows)
        and all(
            isinstance(row, list)
            and 2 <= len(row) <= 26
            and len(set(row)) == len(row)
            and all(isinstance(token, int) and not isinstance(token, bool) for token in row)
            for row in label_rows
        )
        and isinstance(all_labels, list)
        and 2 <= len(all_labels) <= 26
        and len(set(all_labels)) == len(all_labels)
        and all(isinstance(token, int) and not isinstance(token, bool) for token in all_labels)
        and isinstance(prefix, int)
        and not isinstance(prefix, bool)
        and prefix > 0
        and block_size in {16, 32}
        and prefix % block_size == 0
        and prefix < min(map(len, token_rows))
        and all(row[:prefix] == token_rows[0][:prefix] for row in token_rows[1:])
    )
    if not valid or request.prompt_token_ids != token_rows[0]:
        raise ValueError("invalid mJev topology-A metadata")
    return token_rows, label_rows, all_labels, prefix


def _masked_meta(request: Any) -> dict[str, Any] | None:
    extra = _extra(request)
    if extra.get("mjev_protocol") != _MASKED_PROTOCOL:
        return None
    raw_questions = extra.get("questions")
    label_rows = extra.get("label_token_ids")
    all_labels = extra.get("all_label_token_ids")
    root_prefix = extra.get("root_prefix_tokens")
    block_size = extra.get("block_size")
    if (
        not isinstance(raw_questions, list)
        or not 1 <= len(raw_questions) <= 6
        or not isinstance(label_rows, list)
        or len(label_rows) != len(raw_questions)
        or not all(
            isinstance(row, list)
            and 2 <= len(row) <= 26
            and len(set(row)) == len(row)
            and all(isinstance(token, int) and not isinstance(token, bool) for token in row)
            for row in label_rows
        )
        or not isinstance(all_labels, list)
        or not 2 <= len(all_labels) <= 26
        or len(set(all_labels)) != len(all_labels)
        or not all(isinstance(token, int) and not isinstance(token, bool) for token in all_labels)
        or not isinstance(root_prefix, int)
        or isinstance(root_prefix, bool)
        or root_prefix < 0
        or block_size not in {16, 32}
    ):
        raise ValueError("invalid mJev masked metadata")
    layouts = [MaskedQuestionLayout.from_payload(value) for value in raw_questions]
    if any(len(layout.plan.candidates) != len(labels) for layout, labels in zip(layouts, label_rows)):
        raise ValueError("masked candidate spans and label rows differ")
    roots = [layout.compact_candidate(0) for layout in layouts]
    if request.prompt_token_ids != roots[0]:
        raise ValueError("masked root prompt differs from EngineCore prompt")
    if len(layouts) == 1:
        if root_prefix != 0:
            raise ValueError("a single-question masked request must not add a root fork")
    else:
        if root_prefix <= 0 or root_prefix >= min(layout.plan.common.end for layout in layouts):
            raise ValueError("masked root fork must precede every question fork")
        if any(row[:root_prefix] != roots[0][:root_prefix] for row in roots[1:]):
            raise ValueError("masked rows do not share the declared root prefix")
    return {
        "layouts": layouts,
        "label_rows": label_rows,
        "all_labels": all_labels,
        "root_prefix": root_prefix,
        "block_size": block_size,
    }


def _install_topology_a_scheduler() -> None:
    """Expand Q causal question rows after the single frontend/core boundary."""
    from vllm.v1.engine import EngineCoreOutput, FinishReason
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request, RequestStatus

    if getattr(Scheduler, "_mjev_topology_a_installed", False):
        return

    # Request is constructed in EngineCore before Scheduler.add_request.
    if not getattr(Request, "_mjev_metadata_installed", False):
        original_request_init = Request.__init__

        def request_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_request_init(self, *args, **kwargs)
            self._mjev_topology_a = _topology_a_meta(self)

        Request.__init__ = request_init
        Request._mjev_metadata_installed = True

    original_init = Scheduler.__init__
    original_add = Scheduler.add_request
    original_update = Scheduler.update_from_output
    original_finish = Scheduler.finish_requests
    original_try_encoder = Scheduler._try_schedule_encoder_inputs

    def scheduler_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._mjev_a_groups: dict[str, dict[str, Any]] = {}
        self._mjev_a_child_to_parent: dict[str, tuple[str, int]] = {}

    def make_row(
        parent: Any,
        index: int,
        tokens: list[int],
        labels: list[int],
        all_labels: list[int],
    ) -> Any:
        params = parent.sampling_params.clone()
        params.max_tokens = 1
        params.min_tokens = 0
        params.temperature = 0.0
        params.allowed_token_ids = list(labels)
        # One sampled column plus at most 26 labels. This is fixed-width
        # runner metadata only; no model-visible padding is introduced.
        params.logprobs = 27
        params.logprob_token_ids = None
        params.extra_args = None
        encoded = "_".join(str(value) for value in all_labels)
        if index == 0:
            parent.sampling_params = params
            parent.max_tokens = 1
            parent._mjev_topology_a = None
            return parent
        return Request(
            request_id=f"{parent.request_id}::mjev-label-{encoded}--q{index}",
            prompt_token_ids=list(tokens),
            sampling_params=params,
            pooling_params=None,
            client_index=parent.client_index,
            arrival_time=parent.arrival_time,
            prompt_embeds=None,
            prompt_is_token_ids=None,
            mm_features=parent.mm_features,
            lora_request=parent.lora_request,
            cache_salt=parent.cache_salt,
            priority=parent.priority,
            trace_headers=parent.trace_headers,
            block_hasher=parent._block_hasher,
            resumable=False,
        )

    def add_request(self: Any, request: Any) -> None:
        meta = getattr(request, "_mjev_topology_a", None)
        if meta is None:
            original_add(self, request)
            return
        token_rows, label_rows, all_labels, prefix = meta
        parent_id = request.request_id
        row_ids: list[str] = []
        state = {
            "row_ids": row_ids,
            "label_rows": label_rows,
            "scores": {},
            "tokens": {},
            "finished": set(),
            "template": None,
            "client_index": request.client_index,
            "fork_prefix": prefix,
            "forked": False,
        }
        self._mjev_a_groups[parent_id] = state
        for index, (tokens, labels) in enumerate(zip(token_rows, label_rows)):
            row = make_row(request, index, tokens, labels, all_labels)
            row_ids.append(row.request_id)
            self._mjev_a_child_to_parent[row.request_id] = (parent_id, index)
            if index == 0:
                row._mjev_fork_phase = "parent_prefill"
                row._mjev_fork_prefix = prefix
                original_add(self, row)
            else:
                row._mjev_fork_phase = "held"
                self.requests[row.request_id] = row

    def try_schedule_encoder_inputs(
        self: Any,
        request: Any,
        num_computed_tokens: int,
        num_new_tokens: int,
        *args: Any,
        **kwargs: Any,
    ):
        result = original_try_encoder(
            self, request, num_computed_tokens, num_new_tokens, *args, **kwargs
        )
        if getattr(request, "_mjev_fork_phase", None) != "parent_prefill":
            return result
        prefix = request._mjev_fork_prefix
        remaining = prefix - num_computed_tokens
        if remaining <= 0:
            raise RuntimeError("topology A parent reached fork before activation")
        if result[1] <= remaining:
            return result
        return result[0], remaining, result[2], result[3]

    def activate_fork(self: Any, parent_id: str) -> None:
        state = self._mjev_a_groups[parent_id]
        if state["forked"]:
            return
        parent = self.requests[parent_id]
        prefix = state["fork_prefix"]
        if parent.num_computed_tokens != prefix:
            raise RuntimeError("topology A parent missed aligned fork boundary")
        children = [self.requests[row_id] for row_id in state["row_ids"][1:]]
        coordinator = self.kv_cache_manager.coordinator
        block_pool = coordinator.block_pool
        managers = coordinator.single_type_managers
        source_tables = coordinator.get_blocks(parent_id)
        if len(source_tables) != len(managers):
            raise RuntimeError("topology A KV cache group mismatch")
        for manager, source_table in zip(managers, source_tables):
            if prefix % manager.block_size:
                raise RuntimeError("topology A fork is not aligned to every KV page")
            completed = prefix // manager.block_size
            if len(source_table) < completed:
                raise RuntimeError("topology A parent cache lacks prefix pages")
            for child in children:
                table = list(source_table[:completed])
                block_pool.touch([block for block in table if not block.is_null])
                manager.req_to_blocks[child.request_id] = table
                manager.num_cached_block[child.request_id] = completed
        for child in children:
            child.num_computed_tokens = prefix
            child.status = RequestStatus.WAITING
            child._mjev_fork_phase = "ready"
            self._enqueue_waiting_request(child)
        parent._mjev_fork_phase = "ready"
        state["forked"] = True
        print(
            "MJEV_CAUSAL_FORK "
            f"request={parent_id} prefix={prefix} children={len(children)}",
            flush=True,
        )

    @staticmethod
    def raw_scores(output: Any, labels: list[int]) -> dict[int, float]:
        envelope = output.new_logprobs
        if envelope is None or len(envelope.logprob_token_ids) != 1:
            raise RuntimeError("topology A row returned no compact score envelope")
        values: dict[int, float] = {}
        for token_id, value in zip(
            envelope.logprob_token_ids[0].tolist(),
            envelope.logprobs[0].tolist(),
        ):
            if int(token_id) in labels:
                values[int(token_id)] = float(value)
        if set(values) != set(labels):
            raise RuntimeError(
                "topology A row omitted label logits: "
                f"expected={labels} actual={envelope.logprob_token_ids[0].tolist()}"
            )
        return values

    def update_from_output(self: Any, scheduler_output: Any, model_runner_output: Any):
        result = original_update(self, scheduler_output, model_runner_output)
        for request_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(request_id)
            if (
                request is not None
                and getattr(request, "_mjev_fork_phase", None) == "parent_prefill"
                and request.num_computed_tokens == request._mjev_fork_prefix
            ):
                activate_fork(self, request_id)
        internal_ids = set(self._mjev_a_child_to_parent)
        completed: set[str] = set()
        for bundle in result.values():
            kept = []
            for output in bundle.outputs:
                mapping = self._mjev_a_child_to_parent.get(output.request_id)
                if mapping is None:
                    kept.append(output)
                    continue
                parent_id, index = mapping
                state = self._mjev_a_groups[parent_id]
                if output.new_token_ids:
                    state["tokens"][index] = int(output.new_token_ids[0])
                    state["scores"][index] = raw_scores(
                        output, state["label_rows"][index]
                    )
                if index == 0:
                    state["template"] = output
                if output.finished:
                    state["finished"].add(index)
                if len(state["finished"]) == len(state["row_ids"]):
                    expected = set(range(len(state["row_ids"])))
                    if set(state["tokens"]) != expected or set(state["scores"]) != expected:
                        raise RuntimeError("topology A completed with missing rows")
                    template = state["template"] or output
                    payload = [state["scores"][row] for row in range(len(state["row_ids"]))]
                    kept.append(
                        EngineCoreOutput(
                            request_id=parent_id,
                            new_token_ids=[state["tokens"][row] for row in range(len(state["row_ids"]))],
                            finish_reason=FinishReason.LENGTH,
                            stop_reason="mjev-scores:" + json.dumps(payload, separators=(",", ":")),
                            events=template.events,
                            prefill_stats=template.prefill_stats,
                            trace_headers=template.trace_headers,
                            num_nans_in_logits=0,
                        )
                    )
                    completed.add(parent_id)
            bundle.outputs = kept
            if bundle.finished_requests is not None:
                bundle.finished_requests = {
                    request_id for request_id in bundle.finished_requests if request_id not in internal_ids
                }
        for parent_id in completed:
            state = self._mjev_a_groups.pop(parent_id)
            bundle = result.get(state["client_index"])
            if bundle is not None:
                finished = set(bundle.finished_requests or ())
                finished.add(parent_id)
                bundle.finished_requests = finished
            for row_id in state["row_ids"]:
                self._mjev_a_child_to_parent.pop(row_id, None)
        return result

    def finish_requests(self: Any, request_ids: Any, finished_status: Any):
        if isinstance(request_ids, str) and request_ids in self._mjev_a_groups:
            request_ids = list(self._mjev_a_groups[request_ids]["row_ids"])
        return original_finish(self, request_ids, finished_status)

    Scheduler.__init__ = scheduler_init
    Scheduler.add_request = add_request
    Scheduler._try_schedule_encoder_inputs = try_schedule_encoder_inputs
    Scheduler.update_from_output = update_from_output
    Scheduler.finish_requests = finish_requests
    Scheduler._mjev_topology_a_installed = True


def _install_masked_scheduler() -> None:
    """Install one-request hierarchical branches and Tree-KV decision merge."""
    from vllm.v1.engine import EngineCoreOutput, FinishReason
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request, RequestStatus

    if getattr(Scheduler, "_mjev_masked_installed", False):
        return

    # Topology A installs the first Request wrapper.  Stack this wrapper on
    # top so both private protocols remain opt-in and ordinary requests are
    # byte-for-byte stock vLLM objects.
    original_request_init = Request.__init__
    original_num_tokens_with_spec = Request.num_tokens_with_spec.fget
    assert original_num_tokens_with_spec is not None

    def request_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_request_init(self, *args, **kwargs)
        self._mjev_masked = _masked_meta(self)

    Request.__init__ = request_init

    def num_tokens_with_spec(self: Any) -> int:
        value = original_num_tokens_with_spec(self)
        phase = getattr(self, "_mjev_mask_phase", None)
        target = getattr(self, "_mjev_mask_target", None)
        if (
            phase in {"root_prefill", "question_prefill", "candidate_prefill"}
            and isinstance(target, int)
            and self.num_computed_tokens < target
        ):
            return min(value, target)
        return value

    Request.num_tokens_with_spec = property(num_tokens_with_spec)

    original_init = Scheduler.__init__
    original_add = Scheduler.add_request
    original_try_encoder = Scheduler._try_schedule_encoder_inputs
    original_update = Scheduler.update_from_output
    original_finish = Scheduler.finish_requests

    def scheduler_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._mjev_mask_groups: dict[str, dict[str, Any]] = {}
        self._mjev_mask_row_to_parent: dict[str, tuple[str, str, int, int]] = {}

    def row_params(
        parent: Any,
        all_labels: list[int],
        role: str,
        *,
        position_suffix_start: int = 0,
        position_shift: int = 0,
        max_tokens: int = 2,
    ) -> Any:
        params = parent.sampling_params.clone()
        params.max_tokens = max_tokens
        params.min_tokens = 0
        params.temperature = 0.0
        params.allowed_token_ids = list(all_labels)
        params.logprobs = 27
        params.logprob_token_ids = None
        params.extra_args = {
            "mjev_protocol": _MASKED_ROW_PROTOCOL,
            "role": role,
            "position_suffix_start": position_suffix_start,
            "position_shift": position_shift,
            "copy_segments": [],
        }
        return params

    def make_row(
        parent: Any,
        request_id: str,
        tokens: list[int],
        params: Any,
    ) -> Any:
        return Request(
            request_id=request_id,
            prompt_token_ids=list(tokens),
            sampling_params=params,
            pooling_params=None,
            client_index=parent.client_index,
            arrival_time=parent.arrival_time,
            prompt_embeds=None,
            prompt_is_token_ids=None,
            mm_features=parent.mm_features,
            lora_request=parent.lora_request,
            cache_salt=parent.cache_salt,
            priority=parent.priority,
            trace_headers=parent.trace_headers,
            block_hasher=parent._block_hasher,
            resumable=False,
        )

    def register_row(
        self: Any,
        state: dict[str, Any],
        row: Any,
        role: str,
        question: int,
        candidate: int,
    ) -> None:
        parent_id = state["parent_id"]
        state["row_ids"].add(row.request_id)
        self._mjev_mask_row_to_parent[row.request_id] = (
            parent_id,
            role,
            question,
            candidate,
        )

    def install_exact_prefix(
        self: Any,
        source: Any,
        child: Any,
        prefix_tokens: int,
        block_size: int,
    ) -> None:
        coordinator = self.kv_cache_manager.coordinator
        managers = coordinator.single_type_managers
        source_tables = coordinator.get_blocks(source.request_id)
        if len(source_tables) != len(managers):
            raise RuntimeError("masked prefix KV cache group mismatch")
        full_pages, partial = divmod(prefix_tokens, block_size)
        copy_segments: list[list[Any]] = []
        for manager, source_table in zip(managers, source_tables):
            if manager.block_size != block_size:
                raise RuntimeError("masked Tree-KV requires uniform scheduler pages")
            if len(source_table) < full_pages + bool(partial):
                raise RuntimeError("masked source lacks the exact prefix pages")
            table = list(source_table[:full_pages])
            coordinator.block_pool.touch(
                [block for block in table if not block.is_null]
            )
            if partial:
                block = coordinator.block_pool.get_new_blocks(1)[0]
                table.append(block)
                manager.new_block_ids.append(block.block_id)
            manager.req_to_blocks[child.request_id] = table
            manager.num_cached_block[child.request_id] = full_pages
        if partial:
            start = full_pages * block_size
            copy_segments.append([source.request_id, start, start, partial])
        extra = child.sampling_params.extra_args
        assert isinstance(extra, dict)
        extra["copy_segments"] = copy_segments
        extra["block_size"] = block_size
        child.num_computed_tokens = prefix_tokens

    def masked_add_request(self: Any, request: Any) -> None:
        meta = getattr(request, "_mjev_masked", None)
        if meta is None:
            original_add(self, request)
            return
        layouts: list[MaskedQuestionLayout] = meta["layouts"]
        all_labels: list[int] = meta["all_labels"]
        encoded = "_".join(str(value) for value in all_labels)
        parent_id = request.request_id
        state: dict[str, Any] = {
            "parent_id": parent_id,
            "client_index": request.client_index,
            "layouts": layouts,
            "label_rows": meta["label_rows"],
            "all_labels": all_labels,
            "block_size": meta["block_size"],
            "root_prefix": meta["root_prefix"],
            "row_ids": set(),
            "question_ids": [],
            "candidate_ids": [[] for _ in layouts],
            "decision_ids": [],
            "candidate_ready": set(),
            "decision_scores": {},
            "decision_tokens": {},
            "decision_finished": set(),
            "template": None,
            "decisions_created": False,
            "copy_tokens": 0,
            "zero_copy_pages": 0,
            "started": time.perf_counter(),
        }
        self._mjev_mask_groups[parent_id] = state

        question_rows: list[Any] = []
        for question, layout in enumerate(layouts):
            tokens = layout.compact_candidate(0)
            params = row_params(
                request,
                all_labels,
                "question",
                position_suffix_start=layout.plan.common.end,
                position_shift=0,
            )
            if question == 0:
                row = request
                row.prompt_token_ids = list(tokens)
                row.sampling_params = params
                row.max_tokens = 2
                row._mjev_masked = None
            else:
                row = make_row(
                    request,
                    f"{parent_id}::mjev-label-{encoded}--mq{question}",
                    tokens,
                    params,
                )
            row._mjev_mask_phase = "held"
            row._mjev_mask_target = layout.plan.common.end
            question_rows.append(row)
            state["question_ids"].append(row.request_id)
            register_row(self, state, row, "question", question, 0)

        root = question_rows[0]
        if len(question_rows) == 1:
            root._mjev_mask_phase = "question_prefill"
            original_add(self, root)
        else:
            root._mjev_mask_phase = "root_prefill"
            root._mjev_mask_target = meta["root_prefix"]
            original_add(self, root)
            for child in question_rows[1:]:
                self.requests[child.request_id] = child

    def activate_root(self: Any, parent_id: str) -> None:
        state = self._mjev_mask_groups[parent_id]
        root = self.requests[state["question_ids"][0]]
        prefix = state["root_prefix"]
        if root.num_computed_tokens != prefix:
            raise RuntimeError("masked root missed its exact fork boundary")
        for question, row_id in enumerate(state["question_ids"][1:], start=1):
            child = self.requests[row_id]
            install_exact_prefix(self, root, child, prefix, state["block_size"])
            child.status = RequestStatus.WAITING
            child._mjev_mask_phase = "question_prefill"
            child._mjev_mask_target = state["layouts"][question].plan.common.end
            self._enqueue_waiting_request(child)
        root._mjev_mask_phase = "question_prefill"
        root._mjev_mask_target = state["layouts"][0].plan.common.end

    def activate_question(self: Any, parent_id: str, question: int) -> None:
        state = self._mjev_mask_groups[parent_id]
        layout: MaskedQuestionLayout = state["layouts"][question]
        question_row = self.requests[state["question_ids"][question]]
        common_end = layout.plan.common.end
        if question_row.num_computed_tokens != common_end:
            raise RuntimeError("masked question missed its exact candidate fork")
        encoded = "_".join(str(value) for value in state["all_labels"])
        candidate_ids = [question_row.request_id]
        question_row._mjev_mask_phase = "candidate_prefill"
        question_row._mjev_mask_target = len(layout.compact_candidate(0))
        self._mjev_mask_row_to_parent[question_row.request_id] = (
            parent_id,
            "candidate",
            question,
            0,
        )
        for candidate, span in enumerate(layout.plan.candidates[1:], start=1):
            params = row_params(
                question_row,
                state["all_labels"],
                "candidate",
                position_suffix_start=common_end,
                position_shift=span.start - common_end,
            )
            row = make_row(
                question_row,
                (
                    f"{parent_id}::mjev-label-{encoded}--"
                    f"mq{question}c{candidate}"
                ),
                layout.compact_candidate(candidate),
                params,
            )
            row._mjev_mask_phase = "candidate_prefill"
            row._mjev_mask_target = len(row.prompt_token_ids)
            register_row(self, state, row, "candidate", question, candidate)
            self.requests[row.request_id] = row
            install_exact_prefix(
                self, question_row, row, common_end, state["block_size"]
            )
            row.status = RequestStatus.WAITING
            self._enqueue_waiting_request(row)
            candidate_ids.append(row.request_id)
        state["candidate_ids"][question] = candidate_ids

    @staticmethod
    def coalesce_copy_segments(
        token_sources: list[tuple[str, int]],
        copied: list[bool],
    ) -> list[list[Any]]:
        segments: list[list[Any]] = []
        index = 0
        while index < len(token_sources):
            if not copied[index]:
                index += 1
                continue
            source_id, source_pos = token_sources[index]
            end = index + 1
            while (
                end < len(token_sources)
                and copied[end]
                and token_sources[end][0] == source_id
                and token_sources[end][1] == source_pos + (end - index)
            ):
                end += 1
            segments.append([source_id, source_pos, index, end - index])
            index = end
        return segments

    def create_decisions(self: Any, parent_id: str) -> None:
        state = self._mjev_mask_groups[parent_id]
        if state["decisions_created"]:
            return
        state["decisions_created"] = True
        block_size = state["block_size"]
        coordinator = self.kv_cache_manager.coordinator
        managers = coordinator.single_type_managers
        encoded = "_".join(str(value) for value in state["all_labels"])
        template_parent = self.requests[state["question_ids"][0]]

        for question, layout in enumerate(state["layouts"]):
            plan = layout.plan
            candidate_ids = state["candidate_ids"][question]
            token_sources: list[tuple[str, int] | None] = [None] * plan.decision.start
            common_source = candidate_ids[0]
            for position in range(plan.common.end):
                token_sources[position] = (common_source, position)
            for candidate, span in enumerate(plan.candidates):
                source_id = candidate_ids[candidate]
                source_start = plan.common.end
                for position in range(span.start, span.end):
                    token_sources[position] = (
                        source_id,
                        source_start + position - span.start,
                    )
            if any(value is None for value in token_sources):
                raise RuntimeError("masked merge source map has an uncovered token")
            sources = [value for value in token_sources if value is not None]

            # A destination page can reference a source page without a copy
            # only when every token maps to one aligned, contiguous source page.
            shareable: dict[int, tuple[str, int]] = {}
            num_prefix_pages = (plan.decision.start + block_size - 1) // block_size
            for page in range(num_prefix_pages):
                start = page * block_size
                end = min(start + block_size, plan.decision.start)
                if end - start != block_size:
                    continue
                source_id, source_pos = sources[start]
                if source_pos % block_size:
                    continue
                if all(
                    sources[index] == (source_id, source_pos + index - start)
                    for index in range(start, end)
                ):
                    shareable[page] = (source_id, source_pos // block_size)

            copied = [True] * plan.decision.start
            for page in shareable:
                for index in range(page * block_size, (page + 1) * block_size):
                    copied[index] = False
            copy_segments = coalesce_copy_segments(sources, copied)

            params = row_params(
                template_parent,
                state["all_labels"],
                "decision",
                max_tokens=1,
            )
            extra = params.extra_args
            assert isinstance(extra, dict)
            extra["copy_segments"] = copy_segments
            extra["block_size"] = block_size
            request_id = (
                f"{parent_id}::mjev-label-{encoded}--decision{question}"
            )
            decision = make_row(
                template_parent,
                request_id,
                list(layout.prompt_token_ids),
                params,
            )
            decision.num_computed_tokens = plan.decision.start
            register_row(self, state, decision, "decision", question, -1)

            for manager in managers:
                if manager.block_size != block_size:
                    raise RuntimeError("masked decision pages are not uniform")
                table = []
                for page in range(num_prefix_pages):
                    shared = shareable.get(page)
                    if shared is not None:
                        source_id, source_page = shared
                        source_table = manager.req_to_blocks[source_id]
                        if source_page >= len(source_table):
                            raise RuntimeError("masked zero-copy source page is absent")
                        block = source_table[source_page]
                        coordinator.block_pool.touch([block])
                        table.append(block)
                    else:
                        block = coordinator.block_pool.get_new_blocks(1)[0]
                        table.append(block)
                        manager.new_block_ids.append(block.block_id)
                manager.req_to_blocks[request_id] = table
                # The merged prefix is deliberately private.  Mark full pages
                # as already handled so token hashes never enter the global APC.
                manager.num_cached_block[request_id] = plan.decision.start // block_size

            state["copy_tokens"] += sum(segment[3] for segment in copy_segments)
            state["zero_copy_pages"] += len(shareable)
            state["decision_ids"].append(request_id)
            original_add(self, decision)

    def try_schedule_encoder_inputs(
        self: Any,
        request: Any,
        num_computed_tokens: int,
        num_new_tokens: int,
        *args: Any,
        **kwargs: Any,
    ):
        result = original_try_encoder(
            self, request, num_computed_tokens, num_new_tokens, *args, **kwargs
        )
        phase = getattr(request, "_mjev_mask_phase", None)
        if phase not in {"root_prefill", "question_prefill", "candidate_prefill"}:
            return result
        target = request._mjev_mask_target
        remaining = target - num_computed_tokens
        if remaining <= 0:
            raise RuntimeError(f"masked {phase} passed its activation boundary")
        if result[1] <= remaining:
            return result
        return result[0], remaining, result[2], result[3]

    @staticmethod
    def raw_scores(output: Any, labels: list[int]) -> dict[int, float]:
        envelope = output.new_logprobs
        if envelope is None or len(envelope.logprob_token_ids) != 1:
            raise RuntimeError("masked decision returned no compact logits")
        values: dict[int, float] = {}
        for token_id, value in zip(
            envelope.logprob_token_ids[0].tolist(),
            envelope.logprobs[0].tolist(),
        ):
            if int(token_id) in labels:
                values[int(token_id)] = float(value)
        if set(values) != set(labels):
            raise RuntimeError("masked decision omitted requested label logits")
        return values

    def update_from_output(self: Any, scheduler_output: Any, model_runner_output: Any):
        result = original_update(self, scheduler_output, model_runner_output)

        # Stage transitions happen only after the producing forward has
        # completed, so every child reads immutable source KV.
        for request_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(request_id)
            if request is None:
                continue
            mapping = self._mjev_mask_row_to_parent.get(request_id)
            if mapping is None:
                continue
            parent_id, _, question, _ = mapping
            phase = getattr(request, "_mjev_mask_phase", None)
            target = getattr(request, "_mjev_mask_target", None)
            if target is None or request.num_computed_tokens != target:
                continue
            if phase == "root_prefill":
                activate_root(self, parent_id)
            elif phase == "question_prefill":
                activate_question(self, parent_id, question)

        completed: set[str] = set()
        internal_ids = set(self._mjev_mask_row_to_parent)
        for bundle in result.values():
            kept = []
            for output in bundle.outputs:
                mapping = self._mjev_mask_row_to_parent.get(output.request_id)
                if mapping is None:
                    kept.append(output)
                    continue
                parent_id, role, question, candidate = mapping
                state = self._mjev_mask_groups[parent_id]
                if role == "candidate":
                    if output.new_token_ids:
                        state["candidate_ready"].add((question, candidate))
                        request = self.requests.get(output.request_id)
                        if request is not None and request in self.running:
                            self.running.remove(request)
                    # Candidate sampled tokens are control signals only and
                    # never cross the public EngineCore boundary.
                    continue
                if role == "decision":
                    if output.new_token_ids:
                        state["decision_tokens"][question] = int(output.new_token_ids[0])
                        state["decision_scores"][question] = raw_scores(
                            output, state["label_rows"][question]
                        )
                    if output.finished:
                        state["decision_finished"].add(question)
                        state["template"] = output
                    if len(state["decision_finished"]) == len(state["layouts"]):
                        expected = set(range(len(state["layouts"])))
                        if (
                            set(state["decision_tokens"]) != expected
                            or set(state["decision_scores"]) != expected
                        ):
                            raise RuntimeError("masked group completed with missing decisions")
                        template = state["template"] or output
                        payload = {
                            "scores": [
                                state["decision_scores"][index]
                                for index in range(len(state["layouts"]))
                            ],
                            "tree_kv": {
                                "copy_tokens": state["copy_tokens"],
                                "zero_copy_pages": state["zero_copy_pages"],
                                "elapsed_ms": round(
                                    (time.perf_counter() - state["started"]) * 1000,
                                    3,
                                ),
                            },
                        }
                        kept.append(
                            EngineCoreOutput(
                                request_id=parent_id,
                                new_token_ids=[
                                    state["decision_tokens"][index]
                                    for index in range(len(state["layouts"]))
                                ],
                                finish_reason=FinishReason.LENGTH,
                                stop_reason=(
                                    "mjev-masked-scores:"
                                    + json.dumps(payload, separators=(",", ":"))
                                ),
                                events=template.events,
                                prefill_stats=template.prefill_stats,
                                trace_headers=template.trace_headers,
                                num_nans_in_logits=0,
                            )
                        )
                        completed.add(parent_id)
                    continue
                # Root/question rows do not emit public output.
            bundle.outputs = kept
            if bundle.finished_requests is not None:
                bundle.finished_requests = {
                    request_id
                    for request_id in bundle.finished_requests
                    if request_id not in internal_ids
                }

        for parent_id, state in list(self._mjev_mask_groups.items()):
            total_candidates = sum(len(row) for row in state["candidate_ids"])
            if (
                not state["decisions_created"]
                and total_candidates > 0
                and len(state["candidate_ready"]) == total_candidates
            ):
                create_decisions(self, parent_id)

        for parent_id in completed:
            state = self._mjev_mask_groups.pop(parent_id)
            sources = [
                row_id
                for row_id in state["row_ids"]
                if row_id in self.requests
            ]
            original_finish(self, sources, RequestStatus.FINISHED_STOPPED)
            for row_id in state["row_ids"]:
                self._mjev_mask_row_to_parent.pop(row_id, None)
            bundle = result.get(state["client_index"])
            if bundle is not None:
                finished = set(bundle.finished_requests or ())
                finished.add(parent_id)
                bundle.finished_requests = finished
        return result

    def finish_requests(self: Any, request_ids: Any, finished_status: Any):
        if isinstance(request_ids, str) and request_ids in self._mjev_mask_groups:
            parent_id = request_ids
            state = self._mjev_mask_groups.pop(parent_id)
            internal = [row_id for row_id in state["row_ids"] if row_id in self.requests]
            result = original_finish(self, internal, finished_status)
            for row_id in state["row_ids"]:
                self._mjev_mask_row_to_parent.pop(row_id, None)
            return [(parent_id, state["client_index"])] if result else []
        return original_finish(self, request_ids, finished_status)

    Scheduler.__init__ = scheduler_init
    Scheduler.add_request = masked_add_request
    Scheduler._try_schedule_encoder_inputs = try_schedule_encoder_inputs
    Scheduler.update_from_output = update_from_output
    Scheduler.finish_requests = finish_requests
    Scheduler._mjev_masked_installed = True


def _install_masked_runner() -> None:
    """Apply original MRoPE positions and local-shard KV copy plans."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_mjev_masked_installed", False):
        return
    original_init_mrope = GPUModelRunner._init_mrope_positions
    original_update_states = GPUModelRunner._update_states
    original_execute_mm_encoder = GPUModelRunner._execute_mm_encoder

    def init_mrope_positions(self: Any, req_state: Any) -> None:
        original_init_mrope(self, req_state)
        params = getattr(req_state, "sampling_params", None)
        extra = getattr(params, "extra_args", None)
        if not isinstance(extra, dict) or extra.get("mjev_protocol") != _MASKED_ROW_PROTOCOL:
            return
        start = extra.get("position_suffix_start", 0)
        shift = extra.get("position_shift", 0)
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(shift, int)
            or isinstance(shift, bool)
            or start < 0
            or shift < 0
            or start > req_state.mrope_positions.shape[1]
        ):
            raise RuntimeError("invalid masked MRoPE position metadata")
        if shift:
            # Qwen3-Omni text positions advance equally in all three MRoPE
            # dimensions after the media prefix.  Removed sibling candidates
            # therefore translate to one scalar offset, while media positions
            # remain byte-for-byte native.
            req_state.mrope_positions[:, start:] += shift

    @staticmethod
    def copy_tree_kv(self: Any, destination: Any, extra: dict[str, Any]) -> None:
        import torch

        segments = extra.get("copy_segments")
        if not segments:
            return
        block_size = extra.get("block_size")
        if block_size not in {16, 32}:
            raise RuntimeError("masked KV copy has an invalid page size")
        if len(destination.block_ids) != 1:
            raise RuntimeError("Qwen3-Omni masked KV copy expects one attention group")

        source_blocks: list[int] = []
        source_offsets: list[int] = []
        destination_blocks: list[int] = []
        destination_offsets: list[int] = []
        for value in segments:
            if not isinstance(value, list) or len(value) != 4:
                raise RuntimeError("invalid masked KV copy segment")
            source_id, source_start, destination_start, length = value
            source = self.requests.get(source_id)
            if source is None or len(source.block_ids) != 1:
                raise RuntimeError(f"masked KV source {source_id!r} is unavailable")
            if not all(
                isinstance(number, int) and not isinstance(number, bool) and number >= 0
                for number in (source_start, destination_start, length)
            ) or length <= 0:
                raise RuntimeError("invalid masked KV copy coordinates")
            for offset in range(length):
                source_pos = source_start + offset
                destination_pos = destination_start + offset
                source_page = source_pos // block_size
                destination_page = destination_pos // block_size
                if (
                    source_page >= len(source.block_ids[0])
                    or destination_page >= len(destination.block_ids[0])
                ):
                    raise RuntimeError("masked KV copy addresses an absent page")
                source_blocks.append(source.block_ids[0][source_page])
                source_offsets.append(source_pos % block_size)
                destination_blocks.append(destination.block_ids[0][destination_page])
                destination_offsets.append(destination_pos % block_size)

        device = self.device
        src_blocks = torch.tensor(source_blocks, dtype=torch.long, device=device)
        src_offsets = torch.tensor(source_offsets, dtype=torch.long, device=device)
        dst_blocks = torch.tensor(destination_blocks, dtype=torch.long, device=device)
        dst_offsets = torch.tensor(destination_offsets, dtype=torch.long, device=device)
        started = time.perf_counter()
        for kv_cache in self.kv_caches:
            if kv_cache.ndim < 3 or kv_cache.shape[1] != 2 or kv_cache.shape[2] != block_size:
                raise RuntimeError(
                    f"unsupported Qwen3-Omni KV layout {tuple(kv_cache.shape)}"
                )
            # Logical layout is [page, K/V, token, ...].  Permuting K/V and
            # token gives one advanced-indexed row per copied token.  Assignment
            # scatters into the original non-contiguous view; clone prevents
            # aliasing when a source page is also a zero-copy destination page.
            order = (0, 2, 1, *range(3, kv_cache.ndim))
            token_view = kv_cache.permute(order)
            values = token_view[src_blocks, src_offsets].clone()
            token_view[dst_blocks, dst_offsets] = values

        # Correctness-first fence: every TP rank has finished its local 48-layer
        # shard copy before the decision suffix can read it.  The event-stream
        # replacement is intentionally a later, parity-gated optimization.
        torch.cuda.synchronize(device)
        print(
            "MJEV_TREE_KV_COPY "
            f"request={destination.req_id} tokens={len(source_blocks)} "
            f"layers={len(self.kv_caches)} ms={(time.perf_counter() - started) * 1000:.3f}",
            flush=True,
        )

    def update_states(self: Any, scheduler_output: Any):
        deferred = original_update_states(self, scheduler_output)
        for new_req in scheduler_output.scheduled_new_reqs:
            state = self.requests.get(new_req.req_id)
            if state is None:
                continue
            params = getattr(state, "sampling_params", None)
            extra = getattr(params, "extra_args", None)
            if isinstance(extra, dict) and extra.get("mjev_protocol") == _MASKED_ROW_PROTOCOL:
                copy_tree_kv(self, state, extra)
        return deferred

    def execute_mm_encoder(self: Any, scheduler_output: Any):
        masked = []
        for request_id in scheduler_output.scheduled_encoder_inputs:
            state = self.requests.get(request_id)
            params = getattr(state, "sampling_params", None)
            extra = getattr(params, "extra_args", None)
            if isinstance(extra, dict) and extra.get("mjev_protocol") == _MASKED_ROW_PROTOCOL:
                masked.append(request_id)
        if masked:
            print(
                "MJEV_MEDIA_ENCODER "
                f"requests={len(masked)} features="
                f"{sum(len(scheduler_output.scheduled_encoder_inputs[value]) for value in masked)}",
                flush=True,
            )
        return original_execute_mm_encoder(self, scheduler_output)

    GPUModelRunner._init_mrope_positions = init_mrope_positions
    GPUModelRunner._update_states = update_states
    GPUModelRunner._execute_mm_encoder = execute_mm_encoder
    GPUModelRunner._mjev_masked_installed = True


_INSTALLED = False
_INSTALL_ERROR = None
SUPPORTED_VLLM = '0.25.1'


def _validate_version():
    import vllm
    if vllm.__version__ != SUPPORTED_VLLM:
        raise RuntimeError(f'mJev experimental runtime requires vLLM {SUPPORTED_VLLM}, got {vllm.__version__}')


def _hook_status():
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.request import Request
    return {
        'request_metadata': bool(getattr(Request, '_mjev_metadata_installed', False)),
        'question_scheduler': bool(getattr(Scheduler, '_mjev_topology_a_installed', False)),
        'candidate_scheduler': bool(getattr(Scheduler, '_mjev_masked_installed', False)),
        'candidate_runner': bool(getattr(GPUModelRunner, '_mjev_masked_installed', False)),
        'label_projection': bool(getattr(GPUModelRunner, '_mjev_labels_installed', False)),
    }


def require_installed():
    _validate_version()
    status = _hook_status()
    if _INSTALL_ERROR is not None or not _INSTALLED or not all(status.values()):
        raise RuntimeError('mJev runtime hooks are not ready; restart the process')
    return status


def install() -> None:
    global _INSTALLED, _INSTALL_ERROR
    if _INSTALL_ERROR is not None:
        raise RuntimeError('Partial mJev runtime installation; restart the process') from _INSTALL_ERROR
    if _INSTALLED:
        require_installed()
        return
    try:
        _validate_version()
        _install_topology_a_scheduler()
        _install_masked_scheduler()
        _install_masked_runner()
        _install_legacy_v1_runner()
        if not all(_hook_status().values()):
            raise RuntimeError('Incomplete mJev runtime hook installation')
    except Exception as exc:
        _INSTALL_ERROR = exc
        raise
    _INSTALLED = True
