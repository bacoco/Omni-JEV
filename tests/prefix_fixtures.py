"""Untrained byte tokenizer and synthetic backend for offline mechanism tests."""
from copy import deepcopy
from omni_jev.contracts import DecisionRequest, NoulQuestion, Source, State


class ByteTokenizer:
    chat_template = "synthetic-template-v1"
    all_special_ids = []
    all_special_tokens = ["<system>", "<user>", "<assistant>"]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("utf-8")

    def apply_chat_template(self, messages, **kw):
        assert kw == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<assistant>"


class FakeBackend:
    """Arithmetic test double, NOT a neural model or benchmark-quality baseline."""
    revision = "synthetic-cache-double-v1"
    max_context = 4096

    def __init__(self):
        self.tokenizer = ByteTokenizer()
        self.calls = []
        self.changed = False
        self.environment = {"synthetic": True}

    def check_unchanged(self):
        if self.changed:
            raise RuntimeError("backend changed")

    def synchronize(self):
        pass

    def prefill(self, ids):
        self.calls.append(("prefill", len(ids)))
        return {"tokens": list(ids)}

    def fork(self, cache):
        return deepcopy(cache)

    def cache_nbytes(self, cache):
        return len(cache["tokens"]) * 8

    def score(self, ids, candidates, *, past=None, prefix_length=0):
        self.calls.append(("score", len(ids), prefix_length))
        if past is not None:
            assert len(past["tokens"]) == prefix_length
            past["tokens"].extend(ids)  # Deliberately mutate to test request isolation.
            ids = past["tokens"]
        return tuple(float(sum(ids) * (i + 1) % 101) / 10 for i in candidates), 0.01


def request(text="An invoice for two items.", *, question=None, state_id="s1"):
    return DecisionRequest(state=State(id=state_id, sources=(Source(id="text1", modality="text", text=text),)),
                           questions=(question or NoulQuestion(id="q", instructions="Is this an invoice?"),))
