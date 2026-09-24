"""Synthetic logits only: this example does not load or call a trained model."""

from omni_jev.contracts import (
    Candidate, ChoiceQuestion, DecisionRequest, NoulQuestion, ProcessedSource,
    Provenance, RubricLevel, ScoreQuestion, Source, State,
)
from omni_jev.decoding import LogitRow, decode_batch


def synthetic_example():
    request = DecisionRequest(
        state=State(id="example", sources=(Source(id="message", modality="text", text="Synthetic fixture."),)),
        questions=(
            NoulQuestion(id="mentioned", instructions="Does the text mention the subject?", required_modalities=("text",)),
            ChoiceQuestion(id="category", instructions="Choose a category.", options=(
                Candidate(id="a", description="Category A"), Candidate(id="b", description="Category B"),
            )),
            ScoreQuestion(id="degree", instructions="Evaluate the described degree.", levels=(
                RubricLevel(id="none", index=0, description="No evidence."),
                RubricLevel(id="some", index=1, description="Some evidence."),
                RubricLevel(id="clear", index=2, description="Clear evidence."),
            )),
        ),
    )
    metadata = Provenance(
        model_revision="synthetic-fixture-not-a-model",
        processed_sources=(ProcessedSource(source_id="message", modalities=("text",)),),
    )
    rows = (
        LogitRow("mentioned", ("no", "yes"), (0.0, 2.0), metadata),
        LogitRow("category", ("b", None, "a"), (1.0, float("nan"), 0.0), metadata, (True, False, True)),
        LogitRow("degree", ("none", "some", "clear"), (0.0, 1.0, 2.0), metadata),
    )
    return request, decode_batch(request, rows, include_confidence=True)


if __name__ == "__main__":
    _, response = synthetic_example()
    print(response.model_dump_json(indent=2))
