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


def test_imports():
    """Prüft, ob wichtige Bibliotheken importiert werden können."""
    try:
        import numpy
        import pandas
        import sklearn
        import matplotlib
        import seaborn
    except ImportError as e:
        assert False, f"Import fehlgeschlagen: {e}"
