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
- [Projekte (Case Studies)](#projekte-case-studies)
- [ML-Workflow 2026](#ml-workflow-2026)
- [Lizenz & Credits](#lizenz-credits)

## 📖 Detaillierte Dokumentation

- 📦 **[Installation & Setup](docs/INSTALLATION.md)** — Vollständige Installationsanleitung (uv & pip)
- 📚 **[Projekt-Übersicht](docs/PROJECTS.md)** — Detaillierte Beschreibungen aller Projekte
- 🔄 **[ML-Workflow](docs/ML_WORKFLOW.md)** — Machine Learning Best Practices 2026
- 🐳 **[Docker Setup](docs/DOCKER_INFO.md)** — Big Data Umgebung mit PySpark

## 🔍 Ziele: Small Data vs. Big Data

Small Data (scikit-learn / pandas): In‑Memory-Verarbeitung, komplexes Feature-Engineering, schnelle Iteration — ideal für Datensätze, die in den Arbeitsspeicher passen (z. B. AirBnB-Listings).

Big Data (Apache Spark): Skalierbarkeit und verteiltes Rechnen. Spark-Beispiele werden in separaten Docker-Containern gehalten, um lokale Setups schlank zu halten.

## 📁 Projekt-Übersicht & Ordnerstruktur

Die Struktur trennt klar zwischen modernem Code und archiviertem Legacy-Material:

```
Machine-Learning-with-Python-Upgrade-2026/
├── README.md
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── docs/                    # Detaillierte Dokumentation
│   ├── INSTALLATION.md      # Installationsanleitung
│   ├── PROJECTS.md          # Projekt-Übersicht
│   ├── ML_WORKFLOW.md       # ML Best Practices
│   └── ...
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

Dieses Projekt nutzt **[uv](https://docs.astral.sh/uv/)** für modernes, schnelles Dependency Management.

> 📦 **[Vollständige Installationsanleitung ansehen](docs/INSTALLATION.md)** — Detaillierte Schritte für uv und pip

### Quick Start (mit uv — empfohlen)

```bash
# 1. uv installieren (falls noch nicht vorhanden)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Repository klonen
git clone https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026.git
cd Machine-Learning-with-Python-Upgrade-2026

# 3. Umgebung erstellen und Dependencies installieren
uv sync

# 4. Umgebung aktivieren
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# 5. Beispielskript ausführen
python scripts/Sklearn_MachineLearning_AirBnB.py
```

### Alternative: pip-Installation

Für die klassische pip-Installation siehe **[docs/INSTALLATION.md](docs/INSTALLATION.md)**

> 📖 **Für bestehende User:** Siehe [MIGRATION_UV.md](MIGRATION_UV.md) für Migration von pip zu uv

## 🐳 Docker Environment & Big Data

Für **Big Data Beispiele (PySpark)** wird eine vorkonfigurierte Docker-Umgebung genutzt.

> 🏗️ **[Docker Architektur & Setup](docs/DOCKER_INFO.md)** — Technischer Stack & Infrastruktur  
> 🔬 **[PySpark Clustering Workflow](docs/PYSPARK_TFIDF.md)** — TF-IDF & K-Means Beispiel

## 📚 Projekte (Case Studies)

> 📖 **[Vollständige Projekt-Übersicht ansehen](docs/PROJECTS.md)** — Detaillierte Beschreibungen, Lernziele & Technologien

### Übersicht

**1. 🏠 AirBnB — Preisvorhersage**
- **Script:** `scripts/Sklearn_MachineLearning_AirBnB.py`
- **Fokus:** Vollständige ML-Pipeline mit ColumnTransformer, Iterative Imputation, Logging
- **Details:** [docs/AIRBNB_PRICE_PREDICTION.md](docs/AIRBNB_PRICE_PREDICTION.md)

**2. 🎬 Movies — Predict NaNs**
- **Notebook:** `notebooks/movies/Movies_Machine_Learning_Predict_NaNs.ipynb`
- **Fokus:** ML-gestützte Imputation statt simpler Mittelwerte
- **Details:** [docs/MOVIES_PREDICT_NANS.md](docs/MOVIES_PREDICT_NANS.md)

**3. 🎬 Movies — Stratified Sampling**
- **Notebook:** `notebooks/movies/Movies_Machine_Learning_StratifiedSample.ipynb`
- **Fokus:** Vergleich von Stratified vs. Random Sampling
- **Details:** [docs/MOVIES_STRATIFIED_SAMPLE.md](docs/MOVIES_STRATIFIED_SAMPLE.md)

## 🔄 ML-Workflow 2026

> 📖 **[Vollständiger Workflow & Best Practices](docs/ML_WORKFLOW.md)**

Alle Projekte folgen einem konsistenten, modernen Workflow:

- **Ingestion:** Typ-sicheres Laden mit Validierung
- **EDA:** Verteilungen, Korrelationen, Visual Checks
- **Preprocessing:** Pipelines mit ColumnTransformer (Categorical → OneHot, Numerical → Scaler + Imputer)
- **Training:** GridSearch / RandomizedSearch mit Cross-Validation
- **Evaluation:** RMSE, R², Residual-Analyse
- **Persistenz:** Modell speichern mit `joblib`, reproduzierbare Seeds

## 📝 Lizenz & Credits

Dieses Projekt basiert auf dem ursprünglichen Werk von Andreas Traut.

Lizenz: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

Feedback, Issues und Pull Requests sind willkommen.
