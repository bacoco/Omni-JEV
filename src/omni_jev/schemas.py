"""Export structural JSON Schemas: python -m omni_jev.schemas OUTPUT_DIRECTORY.

Semantic validators (sums, reference membership, etc.) still require the SDK.
"""

import argparse
import json
from pathlib import Path

from .contracts import DecisionRequest, DecisionResponse


def export_schemas(directory: Path) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in (("request", DecisionRequest), ("response", DecisionResponse)):
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        path = directory / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for path in export_schemas(args.directory):
        print(path)


if __name__ == "__main__":
    main()
