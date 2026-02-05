# 🚀 Machine Learning with Python — Upgrade 2026

![CI Status](https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026/actions/workflows/ci.yml/badge.svg)

**Vom Legacy-Code (2020) zu modernen Best Practices**

- **Autor:** Andreas Traut
- **Stand:** Januar 2026
- **Version:** 2026.1

## 📖 Über dieses Projekt

Dieses Repository fasst die Evolution eines Machine-Learning-Workflows von 2020 bis 2026 zusammen. Ziel ist es, alten Code nicht nur lauffähig zu halten, sondern ihn gemäß moderner Software-Engineering-Prinzipien (Modularität, Typisierung, Pipelines, Logging) zu refactoren und zu dokumentieren.

Die Originaldateien sind archiviert; der neuen Code demonstriert praktikable Patterns für Produktion und Forschung.

## 🔄 Evolution: 2020 vs. 2026 (Kurzvergleich)

| Feature | 2020 — Legacy | 2026 — Upgrade |
|---|---:|:---|
| Code-Struktur | Monolithische Skripte & Notebooks | Modulare Pakete, `if __name__ == "__main__"` |
| Daten-Pipelines | Manuelle Schritte (fillna, get_dummies) | `sklearn.pipeline.Pipeline`, `ColumnTransformer` |
| Typisierung | Dynamisch (keine Type Hints) | Typ-Hints (z. B. `pd.DataFrame` → `-> pd.DataFrame`) |
| Konfiguration | Hardcoded Pfade & Parameter | Zentralisierte Konfiguration & Konstanten |
| Logging | Viele `print()` | Professionelles `logging`-Modul |
| Reproduzierbarkeit | Sporadisches `random_state` | Konsequentes Seeding & genaue `requirements.txt` |
| Fehlerbehandlung | Kaum vorhanden | Validierung, `try/except`-Blöcke |

## 📋 Inhaltsverzeichnis

- [Ziele: Small Data vs. Big Data](#ziele-small-data-vs-big-data)
- [Projekt-Übersicht & Ordnerstruktur](#projekt-übersicht-ordnerstruktur)
- [Technischer Stack](#technischer-stack)
- [Installation & Quickstart](#installation-quickstart)
- [Fallstudien (Case Studies)](#fallstudien-case-studies)
- [ML-Workflow 2026](#ml-workflow-2026)
- [Lizenz & Credits](#lizenz-credits)

## 🔍 Ziele: Small Data vs. Big Data

Small Data (scikit-learn / pandas): In‑Memory-Verarbeitung, komplexes Feature-Engineering, schnelle Iteration — ideal für Datensätze, die in den Arbeitsspeicher passen (z. B. AirBnB-Listings).

Big Data (Apache Spark): Skalierbarkeit und verteiltes Rechnen. Spark-Beispiele werden in separaten Docker-Containern gehalten, um lokale Setups schlank zu halten.

## 📁 Projekt-Übersicht & Ordnerstruktur

Die Struktur trennt klar zwischen modernem Code und archiviertem Legacy-Material:

```
Machine-Learning-with-Python-Upgrade-2026/
├── README-UPGRADE2026.md
├── requirements.txt
├── CHANGELOG.md
├── docs/                    # Detaillierte Case-Study-Dokumentation
├── datasets/                # Rohdaten (lokal)
├── notebooks/               # Modernisierte Jupyter-Notebooks
├── scripts/                 # Produktionsreife Skripte
└── legacy/                  # Archiv (Originalcode 2020)
```

## 🛠️ Technischer Stack

- **Python**: >= 3.10
- **Dependency Management**: uv (modernes Locking & schnelle Installation)
- **pandas**: >= 2.0 (ggf. mit PyArrow-Backend)
- **scikit-learn**: >= 1.2 (z. B. `set_config(transform_output="pandas")`)
- **Visualisierung**: matplotlib, seaborn
- **Qualität**: flake8-Konformität, Typ-Hints

## 🚀 Installation & Quickstart

Dieses Projekt nutzt **[uv](https://docs.astral.sh/uv/)** für modernes, schnelles Dependency Management mit deterministischem Locking.

> 📖 **Für bestehende User:** Siehe [MIGRATION_UV.md](MIGRATION_UV.md) für einen detaillierten Migrationsleitfaden

### Methode 1: Mit uv (Empfohlen) ✨

1) **uv installieren** (falls noch nicht vorhanden)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Oder via pip
pip install uv
```

2) **Repository klonen**

```bash
git clone https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026.git
cd Machine-Learning-with-Python-Upgrade-2026
```

3) **Umgebung erstellen und Dependencies installieren** (ein Befehl!)

```bash
uv sync
```

Das erstellt automatisch eine virtuelle Umgebung (`.venv`) und installiert alle Dependencies aus `uv.lock`.

4) **Umgebung aktivieren**

- **Windows (PowerShell)**:
```powershell
.venv\Scripts\Activate.ps1
```

- **Windows (cmd)**:
```cmd
.venv\Scripts\activate.bat
```

- **macOS / Linux**:
```bash
source .venv/bin/activate
```

5) **Beispielskript ausführen (AirBnB)**

```bash
python scripts/Sklearn_MachineLearning_AirBnB.py
```

### Methode 2: Traditionell mit pip (Legacy)

Falls Sie kein uv verwenden möchten, können Sie weiterhin die klassische `requirements.txt` nutzen:

1) **Repository klonen**

```bash
git clone https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026.git
cd Machine-Learning-with-Python-Upgrade-2026
```

2) **Virtuelle Umgebung erstellen**

```bash
python -m venv venv
```

3) **Aktivieren**

- Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

- Windows (cmd):
```cmd
venv\Scripts\activate.bat
```

- macOS / Linux:
```bash
source venv/bin/activate
```

4) **Abhängigkeiten installieren**

```bash
pip install -r requirements.txt
```

5) **Beispielskript ausführen (AirBnB)**

```bash
python scripts/Sklearn_MachineLearning_AirBnB.py
```

### ⚡ Warum uv?

- **Schneller**: 10-100x schneller als pip bei Installation & Dependency Resolution
- **Deterministisch**: `uv.lock` garantiert identische Versionen überall
- **Einfach**: Ein Befehl (`uv sync`) statt mehrerer Schritte
- **Kompatibel**: Funktioniert nahtlos mit bestehenden `requirements.txt` und `pyproject.toml`

## 🐳 Docker Environment & Big Data

Für die **Big Data Beispiele (PySpark)** wird eine vorkonfigurierte Docker-Umgebung genutzt, um eine reibungslose Ausführung ohne komplexe lokale Installationen zu gewährleisten.

- **🏗️ Infrastruktur:** Detaillierter Einblick in den Tech-Stack (Java/Spark/Python Layer) innerhalb des Containers:  
  👉 **[Technische Architektur & Docker Details lesen](./docs/DOCKER_INFO.md)**

- **🔬 Deep Dive:** Anwendung der Umgebung am Beispiel Text Mining (TF-IDF & K-Means):  
  👉 **[PySpark Clustering Workflow ansehen](./docs/PYSPARK_TFIDF.md)**


## 📚 Fallstudien (Case Studies)

1) **AirBnB — Preisvorhersage (Full ML Pipeline)**

- Pfad: `scripts/Sklearn_MachineLearning_AirBnB.py`
- Fokus: ColumnTransformer, iterative Imputer, robustes Logging
- Doku: `docs/AIRBNB_PRICE_PREDICTION.md`

2) **Movies — Predicting NaNs**

- Pfad: `notebooks/movies/`
- Fokus: ML‑gestützte Imputation (DecisionTrees), Stratified vs. Random Sampling
- Doku: `docs/MOVIES_PREDICT_NANS.md`

## 🔄 ML-Workflow 2026

- Ingestion: typ‑sicheres Laden mit Validierung
- EDA: Verteilungen, Korrelationen, Visual Checks
- Preprocessing: Categorical → OneHot (handle_unknown='ignore'), Numerical → Scaler + Imputer
- Training: GridSearch / RandomizedSearch
- Evaluation: RMSE, R², Residual-Analyse
- Persistenz: Modell speichern mit `joblib`

## 📝 Lizenz & Credits

Dieses Projekt basiert auf dem ursprünglichen Werk von Andreas Traut.

Lizenz: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

Feedback, Issues und Pull Requests sind willkommen.
