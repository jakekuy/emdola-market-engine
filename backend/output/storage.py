"""
Storage — save and load batch results to/from disk.

Directory layout:
    data/runs/{batch_id}/
        results.json      — full BatchResult (serialised Pydantic model)
        personas.md       — LLM-generated persona narrative (populated Phase 6+)

Spec references: §9.4 (saved artefacts), §11 (output architecture).
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.models.results import BatchResult


# Default base path — relative to the project root.
DEFAULT_DATA_DIR = Path("data/runs")


def save_batch(
    result: BatchResult,
    base_dir: Path | str = DEFAULT_DATA_DIR,
) -> Path:
    """
    Serialise a BatchResult to disk and return the directory path.

    Creates `data/runs/{batch_id}/results.json` and (if present)
    `data/runs/{batch_id}/personas.md`.
    """
    base_dir = Path(base_dir)
    batch_dir = base_dir / result.batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    results_path = batch_dir / "results.json"
    results_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    if result.persona_narrative_md:
        personas_path = batch_dir / "personas.md"
        personas_path.write_text(result.persona_narrative_md, encoding="utf-8")

    return batch_dir


def load_batch(
    batch_id: str,
    base_dir: Path | str = DEFAULT_DATA_DIR,
) -> BatchResult:
    """Load a BatchResult from disk by batch ID."""
    base_dir = Path(base_dir)
    results_path = base_dir / batch_id / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No results found for batch '{batch_id}' at {results_path}."
        )
    raw = results_path.read_text(encoding="utf-8")
    return BatchResult.model_validate_json(raw)


def list_batches(base_dir: Path | str = DEFAULT_DATA_DIR) -> list[str]:
    """
    Return a list of batch IDs found under base_dir, sorted newest first
    (by directory mtime).
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []
    batch_dirs = [
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / "results.json").exists()
    ]
    batch_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in batch_dirs]
