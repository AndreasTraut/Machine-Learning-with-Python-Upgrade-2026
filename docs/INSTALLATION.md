# 🚀 Installation & Setup

> ➡️ **Zurück zur Übersicht:** [README.md](../README.md)

---

## 📋 Voraussetzungen

- **Python**: >= 3.10
- **Betriebssystem**: Windows, macOS oder Linux
- **Git**: Für das Klonen des Repositories

## 🛠️ Installationsmethoden

Dieses Projekt unterstützt zwei Installationsmethoden:

1. **Mit uv (Empfohlen)** ✨ — Modern, schnell, deterministisch
2. **Mit pip (Legacy)** — Traditioneller Ansatz für Kompatibilität

---

## Methode 1: Installation mit uv (Empfohlen) ✨

**[uv](https://docs.astral.sh/uv/)** ist ein modernes, schnelles Tool für Python-Dependency-Management mit deterministischem Locking.

### Schritt 1: uv installieren

Falls Sie uv noch nicht installiert haben:

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `curl -LsSf https://astral.sh/uv/install.sh | sh`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Der PowerShell-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


**Alternative: Installation via pip:**
```bash
pip install uv
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `pip install uv`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


### Schritt 2: Repository klonen

```bash
git clone https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026.git
cd Machine-Learning-with-Python-Upgrade-2026
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `git clone https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026.git`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


### Schritt 3: Umgebung erstellen und Dependencies installieren

Mit einem einzigen Befehl:

```bash
uv sync
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `uv sync`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


Dieser Befehl:
- Erstellt automatisch eine virtuelle Umgebung im `.venv`-Ordner
- Installiert alle Dependencies aus der `uv.lock`-Datei
- Garantiert identische Versionen wie in der Entwicklung

### Schritt 4: Virtuelle Umgebung aktivieren

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

Der PowerShell-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `.venv\Scripts\Activate.ps1`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


**Windows (cmd):**
```cmd
.venv\Scripts\activate.bat
```

Der CMD-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `.venv\Scripts\activate.bat`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


**macOS / Linux:**
```bash
source .venv/bin/activate
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `source .venv/bin/activate`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


### Schritt 5: Installation verifizieren

Führen Sie ein Beispielskript aus:

```bash
python scripts/Sklearn_MachineLearning_AirBnB.py
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `python scripts/Sklearn_MachineLearning_AirBnB.py`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://scikit-learn.org/stable/.


### ⚡ Vorteile von uv

- **🚀 Schneller**: 10-100x schneller als pip bei Installation & Dependency Resolution
- **🔒 Deterministisch**: `uv.lock` garantiert identische Versionen überall (Dev, CI, Produktion)
- **📦 Einfach**: Ein Befehl (`uv sync`) statt mehrerer Schritte
- **🔄 Kompatibel**: Funktioniert nahtlos mit `requirements.txt` und `pyproject.toml`
- **🔍 Transparent**: Zeigt klar, welche Dependencies installiert werden

> 📖 **Für bestehende User:** Siehe [MIGRATION_UV.md](../MIGRATION_UV.md) für einen detaillierten Migrationsleitfaden von pip zu uv

---

## Methode 2: Installation mit pip (Legacy)

Falls Sie kein uv verwenden möchten, können Sie weiterhin die klassische `requirements.txt` nutzen.

### Schritt 1: Repository klonen

```bash
git clone https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026.git
cd Machine-Learning-with-Python-Upgrade-2026
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `git clone https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026.git`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


### Schritt 2: Virtuelle Umgebung erstellen

```bash
python -m venv venv
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `python -m venv venv`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


### Schritt 3: Virtuelle Umgebung aktivieren

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

Der PowerShell-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `venv\Scripts\Activate.ps1`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


**Windows (cmd):**
```cmd
venv\Scripts\activate.bat
```

Der CMD-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `venv\Scripts\activate.bat`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


**macOS / Linux:**
```bash
source venv/bin/activate
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `source venv/bin/activate`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


### Schritt 4: Dependencies installieren

```bash
pip install -r requirements.txt
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `pip install -r requirements.txt`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://www.gnu.org/software/bash/manual/.


### Schritt 5: Installation verifizieren

Führen Sie ein Beispielskript aus:

```bash
python scripts/Sklearn_MachineLearning_AirBnB.py
```

Der Bash-Befehl zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `python scripts/Sklearn_MachineLearning_AirBnB.py`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://scikit-learn.org/stable/.


---

## 🐳 Docker Environment für Big Data (PySpark)

Für die **Big Data Beispiele (PySpark)** wird eine vorkonfigurierte Docker-Umgebung genutzt.

### Voraussetzungen

- Docker Desktop installiert und gestartet
- Mindestens 4 GB RAM für Docker

### Docker Image verwenden

Details zur Docker-Umgebung und PySpark-Setup:

- **🏗️ Infrastruktur:** [Docker Architektur & Tech-Stack](./DOCKER_INFO.md)
- **🔬 Anwendungsbeispiel:** [PySpark Clustering Workflow (TF-IDF & K-Means)](./PYSPARK_TFIDF.md)

---

## 🔧 Troubleshooting

### Problem: uv Befehl nicht gefunden

**Lösung:**
- Stellen Sie sicher, dass uv korrekt installiert ist
- Schließen und öffnen Sie Ihr Terminal neu
- Prüfen Sie die PATH-Variable

### Problem: Python-Version zu alt

**Lösung:**
- Dieses Projekt benötigt Python >= 3.10
- Aktualisieren Sie Python oder nutzen Sie pyenv/conda für mehrere Versionen

### Problem: Permission Error bei Installation

**Lösung (Windows):**
- Führen Sie PowerShell als Administrator aus
- Oder nutzen Sie `--user` Flag: `pip install --user -r requirements.txt`

**Lösung (macOS/Linux):**
- Nutzen Sie virtuelle Umgebungen (kein sudo!)
- Falls nötig: `chmod +x` für Skripte

### Problem: Import-Fehler trotz Installation

**Lösung:**
- Prüfen Sie, ob die virtuelle Umgebung aktiviert ist
- Verifizieren Sie mit `which python` (macOS/Linux) oder `where python` (Windows)
- Reinstallieren: `uv sync --reinstall` oder `pip install --force-reinstall -r requirements.txt`

---

## 📚 Weiterführende Dokumentation

- **[README.md](../README.md)** — Projekt-Übersicht und Quick Start
- **[MIGRATION_UV.md](../MIGRATION_UV.md)** — Migration von pip zu uv
- **[PROJECTS.md](./PROJECTS.md)** — Detaillierte Projektbeschreibungen
- **[ML_WORKFLOW.md](./ML_WORKFLOW.md)** — Machine Learning Workflow 2026

---

## 🆘 Hilfe & Support

Bei Problemen:
1. Prüfen Sie die [Issues auf GitHub](https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026/issues)
2. Erstellen Sie ein neues Issue mit Ihrer Fehlermeldung
3. Geben Sie Ihr Betriebssystem und Python-Version an

---

> **Zuletzt aktualisiert:** Februar 2026  
> **Autor:** Andreas Traut
