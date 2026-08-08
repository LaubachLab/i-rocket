"""Release-asset checks for benchmarks, conceptual tools, and metadata."""

from __future__ import annotations

import ast
from datetime import date
import inspect
import os
import re
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


def _metadata_value(pattern, text, label):
    match = re.search(pattern, text, flags=re.MULTILINE)
    assert match is not None, f"Could not find {label}."
    return match.group(1).strip()


def test_release_metadata_and_optional_modules():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    project_version = _metadata_value(
        r'^version\s*=\s*"([^"]+)"\s*$',
        pyproject,
        "project version",
    )
    citation_version = _metadata_value(
        r'^version:\s*"?([^"\n]+)"?\s*$',
        citation,
        "citation version",
    )
    citation_date = _metadata_value(
        r'^date-released:\s*"?([^"\n]+)"?\s*$',
        citation,
        "citation release date",
    )
    readme_tag = _metadata_value(
        r'^\*\*GitHub release/tag:\*\*\s*`([^`]+)`\s*$',
        readme,
        "README release tag",
    )
    readme_version = _metadata_value(
        r'^\*\*Python package version:\*\*\s*`([^`]+)`\s*$',
        readme,
        "README package version",
    )
    readme_date = _metadata_value(
        r'^\*\*Release date:\*\*\s*`([^`]+)`\s*$',
        readme,
        "README release date",
    )

    assert project_version == citation_version == readme_version
    assert readme_tag
    assert citation_date == readme_date
    assert date.fromisoformat(readme_date).isoformat() == readme_date
    assert '"spectral"' in pyproject
    assert '"tshap_integration"' in pyproject
    assert '"tqdm>=4.64"' in pyproject
    assert '"pandas>=1.5"' in pyproject
    assert 'tshap = [' in pyproject
    assert '"tshap==0.0.1"' in pyproject
    assert 'datasets = [' in pyproject
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
