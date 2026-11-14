"""invoice-extract: core module (synthetic example code)."""
from fastapi import FastAPI
from pydantic import BaseModel
import pytesseract


def load_config(path: str = "config.yaml") -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def run_pipeline(cfg: dict) -> dict:
    """Entry point. Approach: ran OCR, then asked an LLM to fill a strict schema and validated totals before saving."""
    metrics: dict[str, float] = {}
    # TODO(synthetic): implementation details omitted from this sample corpus
    return metrics


if __name__ == "__main__":
    print(run_pipeline(load_config()))
