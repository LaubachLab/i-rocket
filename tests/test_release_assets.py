"""Release-asset checks for benchmarks, conceptual tools, and metadata."""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

import interpretability
from irocket_model_selection import nested_stability_cv


ROOT = Path(__file__).resolve().parents[1]




def test_benchmark_and_tool_scripts_parse_without_optional_dependencies():
    paths = sorted((ROOT / "benchmarks").rglob("*.py"))
    paths += sorted((ROOT / "tools").glob("*.py"))
    assert paths
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_conceptual_tools_run_from_source_checkout():
    environment = dict(os.environ)
    environment["MPLBACKEND"] = "Agg"
    environment["KMP_WARNINGS"] = "0"
    subprocess.run(
        [sys.executable, "tools/kernel_explorer.py", "--help"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [sys.executable, "tools/pooling_explorer.py", "--self-test"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "self-test passed" in completed.stdout


def test_release_metadata_and_optional_modules():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert 'version = "0.7.0"' in pyproject
    assert "version: 0.7.0" in citation
    assert 'date-released: "2026-07-24"' in citation
    assert '"spectral"' in pyproject
    assert '"tshap_integration"' in pyproject
    assert '"tqdm>=4.64"' in pyproject
    assert 'tshap = [' in pyproject
    assert '"tshap==0.0.1"' in pyproject
    assert 'datasets = [' in pyproject
    assert "260724" in readme
    assert "package version:** `0.7.0`" in readme
    assert "Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google)" in readme


def test_release_documents_exist():
    for name in (
        "PIPELINE_DESIGN.md",
        "RELEASE_NOTES_260724.md",
        "VALIDATION_260724.md",
        "THIRD_PARTY.md",
        "tools/README.md",
    ):
        assert (ROOT / name).is_file(), name


def test_standard_nested_cv_and_trace_progress_defaults():
    signature = inspect.signature(nested_stability_cv)
    assert signature.parameters["outer_cv"].default == 10
    assert signature.parameters["inner_cv"].default == 3
    assert signature.parameters["show_progress"].default is None

    trace_signature = inspect.signature(
        interpretability.composite_activation_trace
    )
    assert trace_signature.parameters["show_progress"].default is None
    assert trace_signature.parameters["progress_threshold"].default == 500


def test_no_release_candidate_metadata_remains():
    paths = (
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
        ROOT / "CITATION.cff",
        ROOT / "pyproject.toml",
        ROOT / "RELEASE_NOTES_260724.md",
        ROOT / "THIRD_PARTY.md",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "0.8.0rc1" not in text
    assert "release candidate" not in text.lower()
