"""Real attention/KV arithmetic on a tiny UNTRAINED PyTorch causal fixture.

Weights are seeded at random. These tests prove cache plumbing, not task quality,
compatibility with a downloaded checkpoint, or GPU speed.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from omni_jev.contracts import Candidate, ChoiceQuestion, RubricLevel, ScoreQuestion
from omni_jev.prefix_backend import TorchCausalBackend
from omni_jev.prefix_benchmark import compare, parity
from omni_jev.prefix_cache import PrefixDecisionEngine
from prefix_fixtures import ByteTokenizer, request


class TinyCausalFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 24)
        self.position = nn.Embedding(4096, 24)
        self.qkv = nn.ModuleList([nn.Linear(24, 72) for _ in range(2)])
        self.out = nn.ModuleList([nn.Linear(24, 24) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(24) for _ in range(2)])
        self.head = nn.Linear(24, 256)
        self.calls = []

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask, position_ids, use_cache,
                return_dict, past_key_values=None):
        b, count = input_ids.shape
        past_len = 0 if past_key_values is None else past_key_values["length"]
        assert return_dict and attention_mask.shape == (b, past_len + count)
        assert bool((attention_mask == 1).all())
        assert position_ids.tolist() == [list(range(past_len, past_len + count))]
        self.calls.append((count, past_len))
        x = self.embedding(input_ids) + self.position(position_ids)
        states = []
        for layer in range(2):
            q, k, v = self.qkv[layer](self.norm[layer](x)).chunk(3, dim=-1)
            q, k, v = [a.view(b, count, 2, 12).transpose(1, 2) for a in (q, k, v)]
            if past_key_values is not None:
                pk, pv = past_key_values["layers"][layer]
                k, v = torch.cat((pk, k), dim=2), torch.cat((pv, v), dim=2)
            allowed = torch.arange(past_len + count)[None, :] <= position_ids[0, :, None]
            scores = (q @ k.transpose(-1, -2)) / (12**0.5)
            weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
            attended = (weights @ v).transpose(1, 2).reshape(b, count, 24)
            x = x + self.out[layer](attended)
            states.append((k, v))
        cache = past_key_values if past_key_values is not None else {}
        if use_cache:
            cache.update(length=past_len + count, layers=states)
        return SimpleNamespace(logits=self.head(x), past_key_values=cache if use_cache else None)


@pytest.fixture
def backend():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(1729)
        model = TinyCausalFixture()
    return TorchCausalBackend(model, ByteTokenizer(), revision="random-fixture-seed-1729", max_context=4096)


@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
@pytest.mark.parametrize("mode", ["chat", "plain"])
def test_true_attention_cached_uncached_parity(backend, kind, mode):
    q = request().questions[0]
    if kind == "choice":
        q = ChoiceQuestion(id="q", instructions="Choose a topic", options=tuple(
            Candidate(id=f"c{i}", description=d) for i, d in enumerate(("invoice", "refund", "other"))))
    elif kind == "score":
        q = ScoreQuestion(id="q", instructions="Rate clarity", levels=tuple(
            RubricLevel(id=f"r{i}", index=i, description=str(i)) for i in range(4)))
    e = PrefixDecisionEngine(backend, mode=mode)
    report, _ = compare(e, [("a", request(question=q)), ("b", request("Different text", question=q))], repeats=2)
    assert report["cache_parity_passed"]
    assert report["cached"]["computed_tokens"] < report["uncached"]["computed_tokens"]
    assert e.cache_info()["tensor_bytes"] > 0


def test_true_attention_aba_isolation_and_model_change(backend):
    e = PrefixDecisionEngine(backend)
    before = e.run(request())
    e.run(request("Longer content " * 10))
    after = e.run(request())
    assert parity(before, after, atol=1e-6)["passed"]
    with torch.no_grad(): backend.model.head.weight.add_(0.1)
    with pytest.raises(RuntimeError, match="changed"): e.run(request())
    assert e.cache_info()["entries"] == 0


def test_cache_tensor_accounting_counts_shared_storage_once(backend):
    x = torch.zeros(10, dtype=torch.float32)
    assert backend.cache_nbytes({"a": x, "b": x[:3]}) == 40


def test_training_mode_invalidates_cache(backend):
    e = PrefixDecisionEngine(backend); e.run(request()); backend.model.train()
    with pytest.raises(RuntimeError): e.run(request())


def test_embedding_only_model_rejected():
    with pytest.raises(ValueError, match="output head"):
        TorchCausalBackend(nn.Linear(2, 2), ByteTokenizer(), revision="synthetic", max_context=8)


def test_configuration_change_invalidates_model(backend):
    e = PrefixDecisionEngine(backend); e.run(request())
    backend.model.config = SimpleNamespace(changed=True)
    with pytest.raises(RuntimeError, match="changed"): e.run(request())


def test_last_position_argument_is_used_when_available():
    class OptimizedFixture(TinyCausalFixture):
        def forward(self, *args, logits_to_keep=0, **kwargs):
            assert logits_to_keep == 1
            out = super().forward(*args, **kwargs)
            out.logits = out.logits[:, -logits_to_keep:]
            return out
    b = TorchCausalBackend(OptimizedFixture(), ByteTokenizer(), revision="random-optimized-fixture", max_context=4096)
    e = PrefixDecisionEngine(b)
    assert parity(e.run(request()), e.run(request(), use_cache=False), atol=1e-5)["passed"]
    assert b.environment["last_position_only_argument"] == "logits_to_keep"
