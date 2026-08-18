from importlib.machinery import SourceFileLoader
from pathlib import Path


module = SourceFileLoader(
    "validate_multiref_dataset",
    str(Path(__file__).resolve().parents[1] / "scripts" / "validate_multiref_dataset.py"),
).load_module()


def test_get_references_prefers_multiref_keys():
    row = {"solution": "r0", "references": ["r0", "r1", "r2", "r3"]}
    assert module.get_references(row) == ["r0", "r1", "r2", "r3"]


def test_get_references_falls_back_to_solution():
    assert module.get_references({"solution": "r0"}) == ["r0"]


def test_normalized_collapses_whitespace_and_case():
    assert module.normalized(" A\n  B ") == "a b"
