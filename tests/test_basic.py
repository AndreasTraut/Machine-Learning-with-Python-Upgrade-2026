"""
Basis-Tests für das Machine Learning Repository.

Diese Tests stellen sicher, dass die grundlegende Projektstruktur funktioniert.
"""
import sys
from pathlib import Path


def test_python_version():
    """Prüft, ob Python-Version >= 3.10 ist."""
    assert sys.version_info >= (3, 10), "Python 3.10 oder höher erforderlich"


def test_project_structure():
    """Prüft, ob wichtige Verzeichnisse existieren."""
    project_root = Path(__file__).parent.parent
    
    assert (project_root / "scripts").exists(), "scripts/ Verzeichnis fehlt"
    assert (project_root / "notebooks").exists(), "notebooks/ Verzeichnis fehlt"
    assert (project_root / "README.md").exists(), "README.md fehlt"
    assert (project_root / "requirements.txt").exists(), "requirements.txt fehlt"


def test_numpy_import():
    """Prüft numpy Import."""
    import numpy  # noqa: F401


def test_pandas_import():
    """Prüft pandas Import."""
    import pandas  # noqa: F401


def test_sklearn_import():
    """Prüft scikit-learn Import."""
    import sklearn  # noqa: F401


def test_matplotlib_import():
    """Prüft matplotlib Import."""
    import matplotlib  # noqa: F401


def test_seaborn_import():
    """Prüft seaborn Import."""
    import seaborn  # noqa: F401
