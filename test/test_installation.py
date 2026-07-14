import importlib
from importlib import metadata
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _installed_version():
    try:
        return metadata.version("idpfold2")
    except metadata.PackageNotFoundError:
        return None


def _run_setup_command(command):
    result = subprocess.run(
        [sys.executable, "setup.py", command],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_setup_metadata_declares_idpfold2_package():
    assert _run_setup_command("--name") == "idpfold2"
    assert _run_setup_command("--version") == "0.0.1"


def test_setup_declares_console_entry_points():
    setup_py = (PROJECT_ROOT / "setup.py").read_text(encoding="utf-8")

    assert "idpfold2-infer = src.inference:main" in setup_py
    assert "idpfold2-train = src.train:main" in setup_py
    assert "src.eval:main" not in setup_py


def test_installed_distribution_metadata_matches_source_when_available():
    version = _installed_version()
    if version is None:
        pytest.skip("idpfold2 is not installed in this Python environment.")

    assert version == _run_setup_command("--version")


def test_console_entry_points_are_registered():
    if _installed_version() is None:
        pytest.skip("idpfold2 is not installed in this Python environment.")

    scripts = {
        entry_point.name: entry_point.value
        for entry_point in metadata.entry_points(group="console_scripts")
    }

    assert scripts["idpfold2-infer"] == "src.inference:main"
    assert scripts["idpfold2-train"] == "src.train:main"


def test_console_scripts_are_on_path():
    if _installed_version() is None:
        pytest.skip("idpfold2 is not installed in this Python environment.")

    assert shutil.which("idpfold2-infer") is not None
    assert shutil.which("idpfold2-train") is not None


def test_core_package_imports():
    import src
    from src.common import atom37_constants
    from src.common import residue_constants

    assert src is not None
    assert "CA" in atom37_constants.atom_types
    assert "A" in residue_constants.restypes


def test_cli_modules_import_without_starting_jobs():
    if _installed_version() is None:
        pytest.skip("idpfold2 is not installed in this Python environment.")

    try:
        inference = importlib.import_module("src.inference")
        train = importlib.import_module("src.train")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Installed idpfold2 environment is missing dependency: {exc.name}")

    assert callable(inference.main)
    assert callable(train.main)
