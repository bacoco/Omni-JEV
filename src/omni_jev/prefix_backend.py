"""Optional PyTorch/Transformers backend. No network or remote-code loading.

Supported loader scope: unquantized, full-attention GPT-2/Llama/Qwen2/Qwen3
causal LMs with a language head. Model families remain runtime candidates until
native parity tests and a pretrained smoke report pass on the target machine.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any


def snapshot_identity(directory: str | Path) -> dict:
    """Hash local artifact bytes; this is NOT authentication of an upstream revision."""
    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError("snapshot must be a local model directory with config.json")
    files = []
    for path in sorted(root.rglob("*")):
        if any(part in (".git", ".cache") for part in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            raise ValueError("materialize snapshot symlinks before loading")
        if not path.is_file():
            continue
        if path.suffix not in (".json", ".safetensors", ".model", ".txt", ".jinja", ".tiktoken"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                      "sha256": digest.hexdigest()})
    if not any(f["path"].endswith(".safetensors") for f in files):
        raise ValueError("only existing safetensors checkpoints are accepted")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files,
            "upstream_revision_verified": False}


class TorchCausalBackend:
    """Single-device eager reference backend; not a production batching server."""
    def __init__(self, model: Any, tokenizer: Any, *, revision: str, max_context: int):
        import torch
        if not revision or type(max_context) is not int or max_context < 2:
            raise ValueError("revision and context limit are required")
        if not callable(getattr(model, "get_output_embeddings", None)) or model.get_output_embeddings() is None:
            raise ValueError("a causal LM output head is required; embedding-only models do not qualify")
        self.model, self.tokenizer = model, tokenizer
        self.revision, self.max_context = revision, max_context
        model.eval()
        model.requires_grad_(False)
        self._stamp = self._parameters_stamp()
        self._config_stamp = self._configuration_stamp()
        self._last_logits_key = next((key for key in ("logits_to_keep", "num_logits_to_keep")
            if key in inspect.signature(model.forward).parameters), None)
        devices = {p.device for p in model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type not in ("cpu", "cuda"):
            raise ValueError("reference backend requires one CPU or CUDA device")
        self.device = next(iter(devices))
        self.environment = {"torch": torch.__version__, "device": str(self.device),
                            "dtype": str(next(model.parameters()).dtype),
                            "last_position_only_argument": self._last_logits_key}

    def _parameters_stamp(self):
        return tuple((name, id(p), p.data_ptr(), p._version, str(p.device), str(p.dtype), tuple(p.shape))
                     for name, p in self.model.named_parameters())

    def _configuration_stamp(self):
        config = getattr(self.model, "config", None)
        payload = config.to_dict() if hasattr(config, "to_dict") else repr(config)
        return json.dumps((payload, getattr(config, "_attn_implementation", None)),
                          sort_keys=True, default=str)

    def check_unchanged(self) -> None:
        if (self.model.training or any(p.requires_grad for p in self.model.parameters()) or
                self._parameters_stamp() != self._stamp or self._configuration_stamp() != self._config_stamp):
            raise RuntimeError("model changed after cache initialization; construct a new backend")

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.device)

    def _forward(self, ids, *, past=None, prefix_length=0, retain=False):
        import torch
        if (not ids or type(prefix_length) is not int or prefix_length < 0 or
                prefix_length + len(ids) > self.max_context):
            raise ValueError("invalid causal input length")
        if (past is None) != (prefix_length == 0):
            raise ValueError("past cache and prefix length must agree")
        kw = {"input_ids": torch.tensor([ids], dtype=torch.long, device=self.device),
              "attention_mask": torch.ones((1, prefix_length + len(ids)), dtype=torch.long, device=self.device),
              "position_ids": torch.arange(prefix_length, prefix_length + len(ids), device=self.device)[None, :],
              "use_cache": retain, "return_dict": True}
        if self._last_logits_key is not None:
            kw[self._last_logits_key] = 1
        if past is not None:
            kw["past_key_values"] = past
        with torch.no_grad():
            return self.model(**kw)

    def prefill(self, ids):
        cache = self._forward(ids, retain=True).past_key_values
        if cache is None:
            raise ValueError("model did not return past_key_values")
        return cache

    def fork(self, cache):
        # DynamicCache can mutate on continuation. Never hand out the stored object.
        return copy.deepcopy(cache)

    def cache_nbytes(self, cache) -> int:
        import torch
        seen, storage = set(), set()
        def visit(obj):
            if id(obj) in seen:
                return 0
            seen.add(id(obj))
            if isinstance(obj, torch.Tensor):
                key = (str(obj.device), obj.untyped_storage().data_ptr())
                if key in storage:
                    return 0
                storage.add(key)
                return obj.untyped_storage().nbytes()
            if isinstance(obj, dict):
                return sum(visit(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return sum(visit(v) for v in obj)
            if hasattr(obj, "__dict__"):
                return visit(vars(obj))
            return 0
        return visit(cache)

    def score(self, ids, candidates, *, past=None, prefix_length=0):
        import torch
        out = self._forward(ids, past=past, prefix_length=prefix_length, retain=past is not None)
        logits = out.logits[0, -1].float()
        if (logits.ndim != 1 or not candidates or min(candidates) < 0 or max(candidates) >= logits.numel()
                or len(set(candidates)) != len(candidates) or not bool(torch.isfinite(logits).all())):
            raise ValueError("invalid language-head scores or candidate IDs")
        index = torch.tensor(candidates, dtype=torch.long, device=logits.device)
        selected = logits[index]
        # Useful diagnostic: probability mass discarded by conditioning on codes.
        mass = torch.exp(torch.logsumexp(selected, 0) - torch.logsumexp(logits, 0)).item()
        return tuple(float(x) for x in selected.cpu().tolist()), min(1.0, max(0.0, mass))


def load_local_backend(directory: str | Path, *, device: str = "cpu", dtype: str = "float32",
                       max_context: int = 4096) -> TorchCausalBackend:
    """Load only local safetensors and a local native tokenizer. Never downloads."""
    if device not in ("cpu", "cuda") or dtype not in ("float32", "bfloat16", "float16"):
        raise ValueError("supported devices: cpu/cuda; dtypes: float32/bfloat16/float16")
    if device == "cpu" and dtype != "float32":
        raise ValueError("the CPU reference uses float32")
    identity = snapshot_identity(directory)
    root = str(Path(directory).expanduser().resolve(strict=True))
    # Optional imports occur only after validating the caller's local path.
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    cfg = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if (cfg.model_type not in ("gpt2", "llama", "qwen2", "qwen3") or cfg.is_encoder_decoder or
            getattr(cfg, "quantization_config", None) or getattr(cfg, "rope_scaling", None) or
            (getattr(cfg, "sliding_window", None) and getattr(cfg, "use_sliding_window", True))):
        raise ValueError("use an unquantized full-attention causal LM from the audited reference family set")
    if type(max_context) is not int or max_context < 2:
        raise ValueError("max_context must be a positive integer >= 2")
    native_limit = getattr(cfg, "max_position_embeddings", getattr(cfg, "n_positions", None))
    if native_limit is None or max_context > native_limit:
        raise ValueError("requested context exceeds the model's declared limit")
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    model, info = AutoModelForCausalLM.from_pretrained(
        root, config=cfg, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=getattr(torch, dtype), attn_implementation="eager",
        output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("checkpoint did not load exactly; inspect compatibility before using it")
    model.to(device)
    backend = TorchCausalBackend(model, tok, revision="local-sha256:" + identity["sha256"], max_context=max_context)
    backend.environment.update({"transformers": importlib.metadata.version("transformers"),
        "snapshot": identity, "attention_implementation": "eager", "pretrained_origin_verified": False})
    return backend
