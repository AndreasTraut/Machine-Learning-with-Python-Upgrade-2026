# Machine Learning with Python - Upgrade 2025

👨‍💻 **Autor:** Andreas Traut  
📅 **Datum:** Dezember 2025 / Januar 2026  
🏷️ **Version:** 2025.1

---

## 📋 Inhaltsverzeichnis

- [🎯 Über dieses Repository](#-über-dieses-repository)
- [📁 Ordnerstruktur](#-ordnerstruktur)
- [🔍 Ziele: Small Data vs. Big Data](#-ziele-small-data-vs-big-data)
- [🚀 Quickstart](#-quickstart)
- [📦 Unterstützte Versionen](#-unterstützte-versionen)
- [✨ Was wurde aktualisiert?](#-was-wurde-aktualisiert)
- [🔄 Machine Learning Workflow](#-machine-learning-workflow)
- [📚 Dateien in diesem Repository](#-dateien-in-diesem-repository)
- [🔧 Installation & Setup](#-installation--setup)
- [🔁 Reproduzierbarkeit](#-reproduzierbarkeit)
- [📊 Datenquellen](#-datenquellen)
- [📚 Weitere Ressourcen](#-weitere-ressourcen)
- [📝 Lizenz & Beiträge](#-lizenz--beiträge)

## 🎯 Über dieses Repository

Dieses Repository enthält modernisierte Machine-Learning-Beispiele, aktualisiert nach den Best Practices von 2025/2026. Die **ursprünglichen Dateien** wurden in den Ordner `/legacy/` verschoben – alle aktualisierten Versionen befinden sich in den Hauptordnern.

## 📁 Ordnerstruktur

```
Machine-Learning-with-Python-Upgrade-2026/
├── README.md                 # Diese Datei - Hauptdokumentation
├── CHANGELOG.md             # Dokumentation aller Änderungen
├── requirements.txt         # Python-Abhängigkeiten
├── LICENSE                  # Lizenzinformationen
│
├── docs/                    # Detaillierte Projekt-Dokumentation
│   ├── MOVIES_PREDICT_NANS.md          # Movies: NaN-Vorhersage
│   ├── MOVIES_STRATIFIED_SAMPLE.md     # Movies: Stratified Sampling
│   ├── AIRBNB_PRICE_PREDICTION.md      # AirBnB: Preisvorhersage
│   └── ML_WORKFLOW.md                  # ML-Workflow Leitfaden
│
├── notebooks/               # Jupyter Notebooks (modernisiert)
│   ├── movies/             # Movies Machine Learning Beispiele
│   │   ├── Movies_Machine_Learning_Predict_NaNs.ipynb
│   │   └── Movies_Machine_Learning_StratifiedSample.ipynb
│   └── iot/                # IoT Sensor Data Beispiele
│
├── scripts/                 # Python Scripts (modernisiert)
│   └── Sklearn_MachineLearning_AirBnB.py
│
├── datasets/                # Datensätze
│   ├── AirBnB/             # AirBnB Listings Daten
│   ├── movies/             # Movies Database
│   ├── environmental-sensor-data-132k/
│   └── TF-idf/
│
├── images/                  # Bilder für Notebooks
│   └── movies/
│
├── media/                   # Screenshots und Diagramme
│
└── legacy/                  # Ursprüngliche Dateien (archiviert)
    ├── old_notebooks/      # Alte Jupyter Notebooks
    ├── old_scripts/        # Alte Python Scripts
    └── iot-example/        # Altes IoT Beispiel
```

## 🔍 Ziele: Small Data vs. Big Data

Dieses Repository demonstriert die **Unterschiede und Gemeinsamkeiten** zwischen **"Small Data"** (Scikit-Learn/Pandas) und **"Big Data"** (Spark) Ansätzen im Machine Learning.

### 🎯 Fokus des Repositories

Der Schwerpunkt liegt auf:

- ✅ Praktischen, wiederverwendbaren Code-Beispielen
- ✅ Vergleich von Scikit-Learn und Apache Spark ML
- ✅ Verständnis der Unterschiede zwischen kleinen und großen Datensätzen
- ✅ Verwendung von IDEs zusätzlich zu Jupyter-Notebooks

### 📊 Small Data vs. Big Data im Detail

**Small Data (Scikit-Learn):**
- Datensätze, die in den Arbeitsspeicher passen
- Einfache, schnelle Entwicklung
- Umfangreiche Bibliotheken (pandas, scikit-learn, matplotlib)
- Ideal für Prototyping und kleinere Projekte

**Big Data (Apache Spark):**
- Verteilte Verarbeitung großer Datensätze
- Skalierbare Algorithmen
- Komplexere Infrastruktur
- Für produktive, große Anwendungen

## 🚀 Quickstart

### Lokale Installation mit venv

```bash
# Python Virtual Environment erstellen
python3 -m venv venv

# Environment aktivieren
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Dependencies installieren
pip install -r requirements.txt

# Python-Skript ausführen (aus dem Hauptverzeichnis)
python scripts/Sklearn_MachineLearning_AirBnB.py

# Jupyter Notebook starten
jupyter lab
# Dann die Notebooks im Ordner notebooks/movies/ öffnen
```

### Mit Conda

```bash
# Conda Environment erstellen
conda create -n ml-python python=3.10
conda activate ml-python

# Dependencies installieren
pip install -r requirements.txt

# Oder mit conda:
conda install pandas numpy scikit-learn matplotlib seaborn jupyterlab
```

### Mit Docker

**Hinweis:** Die Beispiele sind für lokale Ausführung optimiert und benötigen kein Docker. 

Ein Docker-Setup für die Spark-Beispiele ("Big Data") könnte separat erstellt werden. Für die "Small Data" Beispiele in diesem Repository genügt eine lokale Installation mit Python und den in `requirements.txt` aufgeführten Paketen.

## 📦 Unterstützte Versionen

- **Python:** >= 3.10
- **pandas:** >= 2.0
- **numpy:** >= 1.24
- **scikit-learn:** >= 1.2
- **matplotlib:** >= 3.5
- **seaborn:** >= 0.12
- **jupyterlab:** >= 4.0
- **joblib:** >= 1.2

Siehe `requirements.txt` für genaue Versionsangaben.

## ✨ Was wurde aktualisiert?

### API-Änderungen und Modernisierung

1. **OneHotEncoder:** `handle_unknown='ignore'` Parameter hinzugefügt für robustere Verarbeitung unbekannter Kategorien
2. **SimpleImputer:** Moderne API statt veralteter `Imputer`
3. **ColumnTransformer:** Konsistente Verwendung für verschiedene Feature-Typen
4. **FunctionTransformer:** Für benutzerdefinierte Transformationen
5. **LinearRegression:** Veralteter `normalize` Parameter entfernt (jetzt `StandardScaler` in Pipeline)

### Code-Qualität

1. **F-Strings:** Statt `%s` oder `.format()` für bessere Lesbarkeit
2. **Type Hints:** Optional hinzugefügt für bessere Code-Dokumentation
3. **Logging:** `logging` Modul statt `print()` Statements
4. **Modulare Struktur:** Funktionen statt langer Skripte
5. **Docstrings:** Klare Dokumentation aller Funktionen

### Reproduzierbarkeit

1. **random_state:** Konsequent in allen stochastischen Operationen gesetzt
2. **Seeds:** Dokumentiert und konsistent verwendet
3. **Versionierung:** Klare Angaben zu Package-Versionen

### Error Handling

1. Bessere Fehlerbehandlung beim Laden von Dateien
2. Klare Fehlermeldungen mit Hinweisen zur Lösung
3. Validierung von Eingabedaten

## 🔄 Machine Learning Workflow

### 1. Daten einlesen

```python
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Lädt CSV-Datei und gibt DataFrame zurück."""
    return pd.read_csv(filepath)
```

### 2. Explorative Datenanalyse (EDA)

- Datenstruktur verstehen (`info()`, `describe()`)
- Visualisierungen erstellen (Histogramme, Scatter-Plots)
- Korrelationen analysieren
- Fehlende Werte identifizieren

### 3. Datenvorverarbeitung

**Fehlende Werte behandeln:**
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
# oder strategy="mean", "most_frequent", "constant"
```

**Kategorische Features encodieren:**
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```

**Numerische Features skalieren:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

### 4. Pipeline aufbauen

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Numerische Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

# Kategorische Pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Kombinierte Pipeline
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features)
])
```

### 5. Modellwahl und Training

**Regression:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

**Klassifikation:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 6. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X_train, y_train,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error'
)
rmse_scores = np.sqrt(-scores)
```

### 7. Hyperparameter-Optimierung

**Grid Search:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    model, param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

**Randomized Search:**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distributions = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(5, 30)
}

random_search = RandomizedSearchCV(
    model, param_distributions,
    n_iter=20, cv=5,
    random_state=42
)
```

### 8. Evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")
```

### 9. Modell Persistenz

```python
import joblib

# Modell speichern
joblib.dump(best_model, 'model.pkl')

# Modell laden
loaded_model = joblib.load('model.pkl')
```

## � Dateien in diesem Repository

### Notebooks

**[`notebooks/movies/Movies_Machine_Learning_Predict_NaNs.ipynb`](notebooks/movies/Movies_Machine_Learning_Predict_NaNs.ipynb)**
- Vorhersage fehlender Revenue-Werte in Movies-Dataset
- Zeigt Umgang mit Missing Data
- DecisionTree und RandomForest Regressoren

**[`notebooks/movies/Movies_Machine_Learning_StratifiedSample.ipynb`](notebooks/movies/Movies_Machine_Learning_StratifiedSample.ipynb)**
- Stratifiziertes Sampling für ausgewogene Train/Test-Splits
- Vergleich verschiedener Sampling-Strategien
- Pipeline-Erstellung und Cross-Validation

**[`scripts/Sklearn_MachineLearning_AirBnB.py`](scripts/Sklearn_MachineLearning_AirBnB.py)**
- Vollständiger Machine-Learning-Workflow für AirBnB-Preisvorhersage
- Zeigt Best Practices: modularer Aufbau, Type Hints, Logging
- Hyperparameter-Optimierung mit GridSearch und RandomizedSearch
- Verwendung in IDEs wie Spyder oder PyCharm empfohlen

---

> ➡️ **Detaillierte Projekt-Dokumentationen:** Siehe [`docs/`](docs/) Verzeichnis für ausführliche Beschreibungen

**🎓 Projekt-Guides:**
- **[Movies: Vorhersage fehlender Revenue-Werte](docs/MOVIES_PREDICT_NANS.md)** - ML statt Imputation für NaN-Werte
- **[Movies: Stratified Sampling](docs/MOVIES_STRATIFIED_SAMPLE.md)** - Repräsentative Train/Test-Splits erstellen
- **[AirBnB: Preisvorhersage](docs/AIRBNB_PRICE_PREDICTION.md)** - Vollständiger ML-Workflow mit Best Practices

**📖 Leitfäden:**
- **[Machine Learning Workflow](docs/ML_WORKFLOW.md)** - Schritt-für-Schritt Anleitung (9 Phasen)

---

### Konfiguration

**[`requirements.txt`](requirements.txt)**
- Minimale empfohlene Versionen aller Dependencies
- Für reproduzierbare Umgebungen

**[`CHANGELOG.md`](CHANGELOG.md)**
- Detaillierte Liste aller Änderungen
- Begründung für Updates

### Legacy

Die ursprünglichen Dateien befinden sich im Ordner [`legacy/`](legacy/):
- [`legacy/old_notebooks/`](legacy/old_notebooks/) - Alte Jupyter Notebooks
- [`legacy/old_scripts/`](legacy/old_scripts/) - Alte Python Scripts  
- [`legacy/iot-example/`](legacy/iot-example/) - IoT Sensor Data Beispiel

## � Installation & Setup

### Voraussetzungen

- Python 3.10 oder höher
- pip oder conda Package Manager
- Git (für Repository-Clone)

### Schritt-für-Schritt Anleitung

**1. Repository klonen:**
```bash
git clone https://github.com/AndreasTraut/Machine-Learning-with-Python-Upgrade-2026.git
cd Machine-Learning-with-Python-Upgrade-2026
```

**2. Virtuelle Umgebung erstellen (empfohlen):**

```bash
# Mit venv (Python Standard)
python -m venv venv

# Aktivieren:
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

**3. Dependencies installieren:**
```bash
pip install -r requirements.txt
```

**4. Datasets herunterladen:**
- **AirBnB:** [Inside Airbnb](http://insideairbnb.com/get-the-data.html) → Speichern unter `datasets/AirBnB/listings.csv`
- **Movies:** [Kaggle IMDB](https://www.kaggle.com/datasets) → Speichern unter `datasets/movies/movies.csv`

**5. Notebooks oder Scripts ausführen:**
```bash
# Jupyter Lab starten
jupyter lab

# Oder Python Script direkt ausführen
python scripts/Sklearn_MachineLearning_AirBnB.py
```

### IDE-Setup (optional)

Für die Arbeit mit den Python Scripts empfehlen wir:

- **[Spyder IDE](https://www.spyder-ide.org/)** - Teil von Anaconda
- **[PyCharm](https://www.jetbrains.com/pycharm/)** - Professional oder Community
- **[VS Code](https://code.visualstudio.com/)** - Mit Python Extension

**VS Code Extensions:**
- Python (Microsoft)
- Jupyter (Microsoft)
- Pylance (Microsoft)

## �🔁 Reproduzierbarkeit

Für reproduzierbare Ergebnisse:

1. **random_state setzen:**
   ```python
   # In train_test_split
   train_test_split(X, y, test_size=0.2, random_state=42)
   
   # In Modellen
   RandomForestRegressor(n_estimators=100, random_state=42)
   
   # In Cross-Validation
   cross_val_score(model, X, y, cv=5, random_state=42)
   
   # In Grid/Randomized Search
   GridSearchCV(model, param_grid, cv=5, random_state=42)
   ```

2. **Numpy seed setzen:**
   ```python
   import numpy as np
   np.random.seed(42)
   ```

3. **Exakte Versionen verwenden:**
   ```bash
   pip freeze > requirements-exact.txt
   ```

## 📊 Datenquellen

### AirBnB Dataset
- **Quelle:** [Inside Airbnb](http://insideairbnb.com/get-the-data.html)
- **Lizenz:** Creative Commons CC0 1.0 Universal "Public Domain Dedication"
- **Pfad:** `datasets/AirBnB/listings.csv`

### Movies Dataset
- **Quelle:** [Kaggle - IMDB Movies](https://www.kaggle.com/datasets)
- **Pfad:** `datasets/movies/`

**Hinweis:** Die Datasets müssen separat heruntergeladen werden. Die Skripte geben klare Anweisungen, falls Daten fehlen.

## 📚 Weitere Ressourcen

### Dokumentation
- [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

### Tutorials
- [Scikit-Learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

## 📝 Lizenz & Beiträge

### Lizenz

Dieses Werk ist lizenziert unter der **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

**Was bedeutet das?**
- ✅ Sie dürfen das Material teilen und bearbeiten
- ✅ Angemessene Nennung des Urhebers erforderlich
- ❌ Keine kommerzielle Nutzung
- ✅ Weitergabe unter gleichen Bedingungen

Um eine Kopie dieser Lizenz zu sehen, besuchen Sie:
http://creativecommons.org/licenses/by-nc-sa/4.0/

### Beiträge

Dieses Upgrade wurde erstellt, um die Code-Beispiele auf aktuelle Best Practices zu bringen.

**Feedback und Verbesserungen willkommen!**

- 🐛 **Issues:** Melden Sie Fehler oder schlagen Sie Verbesserungen vor
- 💡 **Diskussionen:** Teilen Sie Ihre Ideen und Fragen
- 🔧 **Pull Requests:** Beiträge sind herzlich willkommen

**Kontakt:**
- GitHub: [@AndreasTraut](https://github.com/AndreasTraut)
- LinkedIn: [Andreas Traut](https://www.linkedin.com/in/andreas-traut)

---

**Ursprüngliches Repository:** [AndreasTraut/Machine-Learning-with-Python](https://github.com/AndreasTraut/Machine-Learning-with-Python)
