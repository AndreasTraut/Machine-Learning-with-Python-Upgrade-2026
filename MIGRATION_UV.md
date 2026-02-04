# 📦 Migration zu uv — Moderne Dependency-Verwaltung

## 🎯 Ziel dieser Migration

Dieses Repository nutzt jetzt **[uv](https://docs.astral.sh/uv/)** anstelle der klassischen `requirements.txt` + `pip`-Kombination. uv bietet:

- ⚡ **10-100x schnellere** Installation & Dependency Resolution (geschrieben in Rust)
- 🔒 **Deterministisches Locking** via `uv.lock` für reproduzierbare Umgebungen
- 🧹 **Saubereres Paketmanagement** mit `pyproject.toml` (PEP 621 Standard)
- 🔄 **Volle Kompatibilität** mit bestehenden pip/venv Workflows

---

## 🔄 Was hat sich geändert?

### Vorher (Legacy)

```bash
python -m venv venv
source venv/bin/activate  # oder venv\Scripts\activate auf Windows
pip install -r requirements.txt
```

### Nachher (Modern)

```bash
uv sync  # Erstellt .venv und installiert Dependencies automatisch
source .venv/bin/activate  # oder .venv\Scripts\activate auf Windows
```

---

## 📂 Neue Dateien im Repository

| Datei | Zweck |
|-------|-------|
| `pyproject.toml` | Projekt-Metadaten und Dependencies (ersetzt `setup.py` + `requirements.txt`) |
| `uv.lock` | Lock-File mit exakten Versionen aller Dependencies (ähnlich wie `poetry.lock` oder `Pipfile.lock`) |
| `requirements.txt` | Wird **beibehalten** für Kompatibilität, aber nicht mehr primär verwendet |

---

## 🚀 Schnellstart für bestehende User

### Schritt 1: uv installieren

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Oder via pip (falls curl nicht verfügbar)
pip install uv
```

### Schritt 2: Alte venv löschen (optional)

```bash
# Deaktiviere alte venv falls aktiv
deactivate

# Lösche alte venv
rm -rf venv  # macOS/Linux
# oder
rmdir /s venv  # Windows
```

### Schritt 3: Neue Umgebung erstellen

```bash
# Ein Befehl für alles!
uv sync
```

Das erstellt:
- Virtuelle Umgebung in `.venv/`
- Installiert alle Dependencies aus `uv.lock`
- Garantiert identische Versionen wie im Lock-File

### Schritt 4: Umgebung aktivieren

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat
```

---

## 🛠️ Häufige Aufgaben mit uv

### Pakete hinzufügen

```bash
# Fügt Paket zu pyproject.toml hinzu und installiert es
uv add tqdm

# Fügt Entwicklungs-Paket hinzu
uv add --dev pytest
```

### Pakete entfernen

```bash
uv remove tqdm
```

### Lock-File aktualisieren

```bash
# Wenn pyproject.toml manuell geändert wurde
uv lock
```

### Dependencies neu installieren

```bash
# Löscht .venv und installiert neu
uv sync --reinstall
```

### Nur installieren (ohne Lock-File Update)

```bash
uv sync --frozen
```

---

## 🔄 Kompatibilität mit pip

Falls Sie uv nicht verwenden möchten, bleibt `requirements.txt` weiterhin verfügbar:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Hinweis:** Die `requirements.txt` wird manuell gepflegt und entspricht den Mindestversionen in `pyproject.toml`.

---

## ❓ FAQ

### Warum nicht Poetry?

Poetry ist ebenfalls ein exzellentes Tool, aber:
- uv ist **deutlich schneller** (geschrieben in Rust vs. Python)
- uv hat eine **kleinere Installation** und weniger Overhead
- uv ist **kompatibel mit pip** und erfordert keine Änderung bestehender Workflows
- uv ist **einfacher** für Einsteiger (ein Befehl statt mehrerer)

### Muss ich uv verwenden?

Nein! Das Repository funktioniert weiterhin mit `requirements.txt` + `pip`. uv ist eine Empfehlung für moderne, schnellere Workflows.

### Was ist mit conda/mamba?

Conda/mamba sind weiterhin gültige Alternativen, besonders für nicht-Python Dependencies. uv fokussiert sich auf pure Python-Packages und ist extrem schnell in diesem Bereich.

### Wird uv.lock in Git eingecheckt?

Ja! `uv.lock` sollte eingecheckt werden, um reproduzierbare Builds zu garantieren. Jeder Developer bekommt exakt die gleichen Package-Versionen.

---

## 📖 Weitere Ressourcen

- [uv Dokumentation](https://docs.astral.sh/uv/)
- [uv GitHub Repository](https://github.com/astral-sh/uv)
- [PEP 621 - pyproject.toml Standard](https://peps.python.org/pep-0621/)

---

**Viel Erfolg mit dem modernisierten Dependency Management! 🚀**
