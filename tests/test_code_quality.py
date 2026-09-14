"""Tests de qualité du code — vérifie le style et le formatage de src/ et app/"""

import subprocess
import sys


MODULES = ["src/", "app/"]


def test_flake8():
    """Vérifie que le code respecte les conventions PEP8 (flake8)."""
    result = subprocess.run(
        [sys.executable, "-m", "flake8", "--max-line-length=100", *MODULES],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, f"Violations flake8 :\n{result.stdout}"


def test_black_formatting():
    """Vérifie que le code est correctement formaté (black --check)."""
    result = subprocess.run(
        [sys.executable, "-m", "black", "--check", "--line-length=100", *MODULES],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, (
        f"Fichiers mal formatés (lancer 'black src/ app/' pour corriger) :\n{result.stdout}"
    )


def test_pylint():
    """Vérifie la qualité du code avec pylint (score minimum : 7/10)."""
    result = subprocess.run(
        [sys.executable, "-m", "pylint", "src/", "--fail-under=7.0",
         "--disable=C0114,C0115,C0116"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, f"Score pylint insuffisant :\n{result.stdout}"
