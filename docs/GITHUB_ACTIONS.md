# GitHub Actions - CI/CD Setup

## 📋 Übersicht

Dieses Repository verwendet GitHub Actions für Continuous Integration (CI). Der Workflow prüft automatisch Code-Qualität und führt Tests aus bei jedem Push oder Pull Request zum `main` Branch.

## 🔧 Workflow-Konfiguration

**Datei:** `.github/workflows/ci.yml`

### Was wird geprüft?

1. **Python Setup** - Installation von Python 3.11
2. **Dependencies** - Installation aller Abhängigkeiten aus `requirements.txt`
3. **Code-Linting** - Prüfung des Code-Stils mit Ruff
4. **Tests** - Ausführung aller Tests mit pytest

### Workflow-Trigger

Der Workflow wird automatisch ausgeführt bei:
- `git push` zum `main` Branch
- Pull Requests zum `main` Branch

## 📊 Status Badge

Das Repository zeigt den aktuellen Build-Status in der README.md:

```markdown
![CI Status](https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026/actions/workflows/ci.yml/badge.svg)
```

Der Markdown-Ausschnitt zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `![CI Status](https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026/...`, mit der der zentrale Schritt direkt ausgeführt wird. Die Konfiguration definiert klar den Ablauf der Automatisierung, sodass Prüfungen und Ausführungen zuverlässig in gleicher Reihenfolge laufen. Weitere Details stehen in der offiziellen Dokumentation: https://docs.github.com/actions/using-workflows.


- ✅ **Grün** = Alle Checks erfolgreich
- ❌ **Rot** = Fehler gefunden
- 🟡 **Gelb** = Workflow läuft gerade

## 🧪 Tests

### Test-Verzeichnis

Alle Tests befinden sich im Verzeichnis `tests/`:

```
tests/
├── test_basic.py       # Basis-Tests für Projektstruktur
└── (weitere Tests)     # Zukünftige Tests hier hinzufügen
```

Dieser Block zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `tests/`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://scikit-learn.org/stable/user_guide.html.


### Tests lokal ausführen

```bash
# Alle Tests ausführen
pytest --verbose

# Nur spezifische Tests
pytest tests/test_basic.py -v
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `# Alle Tests ausführen`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


### Neue Tests hinzufügen

1. Erstelle eine neue Datei `tests/test_*.py`
2. Schreibe Test-Funktionen mit Präfix `test_`
3. GitHub Actions führt die Tests automatisch aus

**Beispiel:**
```python
def test_example():
    """Beschreibung was getestet wird."""
    assert 1 + 1 == 2
```

Der Python-Code zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `def test_example():`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://docs.python.org/3/tutorial/.


## 🎨 Code-Linting (Ruff)

### Ruff Konfiguration

Konfiguration in `ruff.toml`:
- **Ziel:** Python 3.10+
- **Ausgeschlossen:** Legacy-Code, generierte Dateien
- **Regeln:** pycodestyle, pyflakes, isort, pep8-naming, pyupgrade

### Lokal linting ausführen

```bash
# Alle Dateien prüfen
ruff check . --extend-exclude legacy

# Mit automatischer Korrektur
ruff check . --fix --extend-exclude legacy

# Nur spezifische Dateien
ruff check scripts/
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `# Alle Dateien prüfen`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


## 🔍 Workflow-Logs ansehen

1. Gehe zu GitHub Repository
2. Klicke auf Tab "Actions"
3. Wähle einen Workflow-Run aus
4. Klicke auf einzelne Jobs für Details

## ⚙️ Erweiterte Konfiguration

### Weitere Python-Versionen testen

Bearbeite `.github/workflows/ci.yml` und ergänze eine Matrix:

```yaml
jobs:
  build-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
```

Die YAML-Konfiguration zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `jobs:`, mit der der zentrale Schritt direkt ausgeführt wird. Die Konfiguration definiert klar den Ablauf der Automatisierung, sodass Prüfungen und Ausführungen zuverlässig in gleicher Reihenfolge laufen. Weitere Details stehen in der offiziellen Dokumentation: https://yaml.org/spec/1.2/spec.html.


**Hinweis:** Vor dem Hinzufügen neuer Python-Versionen sollte die Kompatibilität aller Dependencies geprüft werden.

### Tests auf mehreren Betriebssystemen

```yaml
jobs:
  build-and-test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.11']
```

Die YAML-Konfiguration zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `jobs:`, mit der der zentrale Schritt direkt ausgeführt wird. Die Konfiguration definiert klar den Ablauf der Automatisierung, sodass Prüfungen und Ausführungen zuverlässig in gleicher Reihenfolge laufen. Weitere Details stehen in der offiziellen Dokumentation: https://yaml.org/spec/1.2/spec.html.


## 📚 Weitere Ressourcen

- [GitHub Actions Dokumentation](https://docs.github.com/en/actions)
- [Ruff Dokumentation](https://docs.astral.sh/ruff/)
- [pytest Dokumentation](https://docs.pytest.org/)

## 🛠️ Troubleshooting

### Workflow schlägt fehl

1. Prüfe Workflow-Logs in GitHub Actions Tab
2. Teste lokal mit denselben Kommandos:
   ```bash
   ruff check . --extend-exclude legacy
   pytest --verbose
   ```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `ruff check . --extend-exclude legacy`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.

3. Behebe Fehler und pushe erneut

### Ruff findet zu viele Fehler

- Bearbeite `ruff.toml` und füge Regeln zu `ignore` hinzu
- Oder: Nutze `# noqa: <regel>` Kommentare in Code-Zeilen

### Tests schlagen lokal nicht fehl, aber in CI

- Prüfe Python-Version (lokal vs. CI)
- Prüfe installierte Package-Versionen
- Stelle sicher, dass alle Dependencies in `requirements.txt` sind
