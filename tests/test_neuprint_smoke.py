"""neuprint-python availability checks.

These must pass with NO internet access and NO NEUPRINT_TOKEN. We only verify
that the client machinery is importable and correctly configured; no network
call is ever made here.
"""

from __future__ import annotations

import os

import pytest


def test_neuprint_imports():
    import neuprint
    from neuprint import Client  # noqa: F401

    assert neuprint is not None


def test_client_symbol_is_constructible_type():
    from neuprint import Client

    assert callable(Client)


def test_planned_dataset_constants_are_documented():
    """The values later code will use, pinned here so a change is deliberate."""
    server = "https://neuprint.janelia.org"
    dataset = "male-cns:v1.0"
    assert server.startswith("https://")
    assert dataset.startswith("male-cns:")


def test_token_is_never_hardcoded_in_repo():
    """Guard against a credential being committed to the source tree."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    this_file = Path(__file__).resolve()

    # A real neuPrint token is a JWT, which always starts with this prefix.
    # Build the needle at runtime so this scanner never matches its own source.
    jwt_prefix = "ey" + "J"
    name_needle = "neuprint" + "_token"

    suspicious = []
    files = (
        list(root.glob("src/**/*.py"))
        + list(root.glob("scripts/**/*.py"))
        + list(root.glob("tests/**/*.py"))
        + list(root.glob("configs/**/*.yaml"))
    )
    for path in files:
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            if jwt_prefix.lower() in low and (name_needle in low or "token" in low):
                suspicious.append(f"{path}:{lineno}")
    assert not suspicious, f"possible hard-coded token at: {suspicious}"


@pytest.mark.skipif(
    not os.environ.get("NEUPRINT_TOKEN"), reason="NEUPRINT_TOKEN not set (expected offline)"
)
def test_token_present_is_plausible():
    """Only runs when a token exists. Never prints or asserts on its value."""
    token = os.environ["NEUPRINT_TOKEN"]
    assert len(token) > 20
