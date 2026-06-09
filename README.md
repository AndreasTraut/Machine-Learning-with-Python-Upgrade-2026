# 🚀 Machine Learning with Python — Upgrade 2026

![CI Status](https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026/actions/workflows/ci.yml/badge.svg)

**Vom Legacy-Code (2020) zu modernen Best Practices**

- **Autor:** Andreas Traut
- **Stand:** Januar 2026
- **Version:** 2026.1

## 📖 Über dieses Projekt

Dieses Repository fasst die Evolution eines Machine-Learning-Workflows von 2020 (siehe [alter Stand von 2020](https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026/blob/main/legacy%2FREADME.md)) bis 2026 zusammen. Ziel ist es, alten Code nicht nur lauffähig zu halten, sondern ihn gemäß moderner Software-Engineering-Prinzipien (Modularität, Typisierung, Pipelines, Logging) zu refactoren und zu dokumentieren.

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
Dieses Projekt zeigt eine vollständige, praxisnahe Machine-Learning-Pipeline zur Vorhersage von Unterkunftspreisen. Kategoriale und numerische Merkmale werden mit einem `ColumnTransformer` konsistent aufbereitet, damit alle Verarbeitungsschritte reproduzierbar bleiben. Fehlende Werte werden modellgestützt imputiert, statt nur mit einfachen Standardwerten ersetzt zu werden. Das Skript eignet sich als Vorlage für produktionsnahe Regressionsprojekte mit klarer Struktur und Logging.  
**Script:** `scripts/Sklearn_MachineLearning_AirBnB.py`  
**Details:** [docs/AIRBNB_PRICE_PREDICTION.md](docs/AIRBNB_PRICE_PREDICTION.md)

**2. 🎬 Movies — Predict NaNs**  
Dieses Notebook konzentriert sich auf den gezielten Umgang mit fehlenden Werten in Filmdaten. Statt pauschaler Mittelwert- oder Median-Ersetzung wird ein ML-Ansatz genutzt, um fehlende Werte datenabhängig zu schätzen. Dadurch bleiben Zusammenhänge zwischen Merkmalen besser erhalten und die Datenqualität steigt. Das Projekt macht nachvollziehbar, wann intelligente Imputation einen echten Mehrwert gegenüber einfachen Heuristiken bietet.  
**Notebook:** `notebooks/movies/Movies_Machine_Learning_Predict_NaNs.ipynb`  
**Details:** [docs/MOVIES_PREDICT_NANS.md](docs/MOVIES_PREDICT_NANS.md)

**3. 🎬 Movies — Stratified Sampling**  
Dieses Projekt vergleicht zufällige Stichproben mit stratifiziertem Sampling für Trainings- und Testdaten. Ziel ist es, wichtige Verteilungen im Datensatz auch nach dem Split stabil zu halten. Dadurch werden Modelle fairer bewertet, weil Testdaten die Gesamtpopulation besser repräsentieren. Das Notebook ist besonders hilfreich, um den Einfluss der Datenselektion auf Modellgüte und Generalisierung praktisch zu verstehen.  
**Notebook:** `notebooks/movies/Movies_Machine_Learning_StratifiedSample.ipynb`  
**Details:** [docs/MOVIES_STRATIFIED_SAMPLE.md](docs/MOVIES_STRATIFIED_SAMPLE.md)

## 🔄 ML-Workflow 2026

> 📖 **[Vollständiger Workflow & Best Practices](docs/ML_WORKFLOW.md)**

Alle Projekte folgen einem konsistenten, modernen Workflow in 9 Schritten (ausführlich erklärt in [docs/ML_WORKFLOW.md](docs/ML_WORKFLOW.md)):

1. **Problem definieren** — Ziel, Metriken und Randbedingungen klar festlegen.
2. **Daten sammeln & laden** — Datenquellen zusammenführen und robust einlesen.
3. **Explorative Datenanalyse (EDA)** — Verteilungen, Ausreißer und Muster verstehen.
4. **Datenvorverarbeitung** — fehlende Werte, Skalierung und Kodierung sauber vorbereiten.
5. **Feature Engineering** — aussagekräftige Merkmale aus Rohdaten ableiten.
6. **Train-Test-Split** — Daten korrekt in Trainings- und Testanteile trennen.
7. **Modelltraining & -evaluation** — Modelle trainieren und mit passenden Kennzahlen bewerten.
8. **Hyperparameter-Optimierung** — Modellparameter systematisch verbessern (z. B. via CV-Suche).
9. **Modell-Deployment & Monitoring** — Modell bereitstellen und laufend überwachen.

## 📝 Lizenz & Credits

Dieses Projekt basiert auf dem ursprünglichen Werk von Andreas Traut.

Lizenz: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

Feedback, Issues und Pull Requests sind willkommen.
