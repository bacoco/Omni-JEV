"""Optional native Transformers cache tests with random weights; never download."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from omni_jev.prefix_backend import TorchCausalBackend
from omni_jev.prefix_benchmark import parity
from omni_jev.prefix_cache import PrefixDecisionEngine
from prefix_fixtures import ByteTokenizer, request


@pytest.mark.parametrize("family", ["gpt2", "llama"])
def test_native_random_model_cache_parity(family):
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(21)
        if family == "gpt2":
            cfg = transformers.GPT2Config(vocab_size=256, n_positions=4096, n_embd=24, n_layer=2, n_head=2,
                resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
            cfg._attn_implementation = "eager"
            model = transformers.GPT2LMHeadModel(cfg)
        else:
            cfg = transformers.LlamaConfig(vocab_size=256, hidden_size=24, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096)
            cfg._attn_implementation = "eager"
            model = transformers.LlamaForCausalLM(cfg)
    b = TorchCausalBackend(model, ByteTokenizer(), revision=f"native-random-{family}", max_context=4096)
    e = PrefixDecisionEngine(b)
    first = e.run(request()); e.run(request("Other input")); third = e.run(request())
    assert parity(first, third, atol=1e-5)["passed"]
    assert parity(first, e.run(request(), use_cache=False), atol=1e-5)["passed"]
