# 📚 Projekt-Übersicht

> ➡️ **Zurück zur Übersicht:** [README.md](../README.md)

---

## 🎯 Einleitung

Dieses Repository enthält mehrere Machine-Learning-Projekte, die verschiedene Aspekte des ML-Workflows demonstrieren. Jedes Projekt zeigt **Best Practices 2026** und ist sowohl als **Jupyter Notebook** (für interaktives Lernen) als auch als **Python-Script** (für IDE-Nutzung) verfügbar.

---

## 📋 Projekte im Überblick

### 1. 🏠 AirBnB — Preisvorhersage (Full ML Pipeline)

**Typ:** Regression | **Level:** Intermediate | **Format:** Python Script

> 💾 **Script:** [`scripts/Sklearn_MachineLearning_AirBnB.py`](../scripts/Sklearn_MachineLearning_AirBnB.py)  
> 📊 **Dataset:** [Inside AirBnB](http://insideairbnb.com/get-the-data.html)  
> 📖 **Detaillierte Dokumentation:** [AIRBNB_PRICE_PREDICTION.md](./AIRBNB_PRICE_PREDICTION.md)

#### Projektziel

Entwicklung eines **vollständigen Machine-Learning-Workflows** zur Vorhersage von AirBnB-Preisen basierend auf Features wie Lage, Bewertungen, Zimmeranzahl und Verfügbarkeit.

#### Highlights

- ✅ **Moderne scikit-learn Pipelines** mit `ColumnTransformer`
- ✅ **Iterative Imputation** für fehlende Werte
- ✅ **Robustes Logging** statt print()-Statements
- ✅ **Type Hints** für bessere Code-Dokumentation
- ✅ **Hyperparameter-Optimierung** mit GridSearch
- ✅ **Modell-Persistenz** mit joblib

#### Lernziele

- Aufbau einer vollständigen ML-Pipeline
- Umgang mit gemischten Datentypen (numerisch + kategorial)
- Feature Engineering und Transformation
- Cross-Validation und Hyperparameter-Tuning
- Produktionsreife Code-Strukturierung

#### Technologien

- pandas, numpy
- scikit-learn (Pipelines, ColumnTransformer, GridSearchCV)
- matplotlib, seaborn (Visualisierung)
- logging (professionelles Logging)

---

### 2. 🎬 Movies — Vorhersage fehlender Revenue-Werte

**Typ:** Regression (NaN-Imputation) | **Level:** Beginner-Intermediate | **Format:** Jupyter Notebook

> 💾 **Notebook:** [`notebooks/movies/Movies_Machine_Learning_Predict_NaNs.ipynb`](../notebooks/movies/Movies_Machine_Learning_Predict_NaNs.ipynb)  
> 📊 **Dataset:** [Kaggle - IMDB Movies](https://www.kaggle.com/datasets)  
> 📖 **Detaillierte Dokumentation:** [MOVIES_PREDICT_NANS.md](./MOVIES_PREDICT_NANS.md)

#### Projektziel

Demonstration, wie **fehlende Werte (NaN)** nicht durch einfache Mittelwerte ersetzt werden, sondern durch **Machine Learning Modelle vorhergesagt** werden können. Ziel ist es, fehlende Revenue-Werte von Filmen basierend auf Jahr, Score, Genre und anderen Features zu schätzen.

#### Highlights

- ✅ **ML-basierte Imputation** statt simpler Durchschnittswerte
- ✅ **DecisionTree Regressor** für NaN-Vorhersage
- ✅ **Feature Engineering** mit One-Hot-Encoding für Genre
- ✅ **Visualisierung** von Residuen und Vorhersagequalität
- ✅ **Interaktives Notebook** zum Experimentieren

#### Lernziele

- Intelligenter Umgang mit fehlenden Werten
- Anwendung von Machine Learning zur Daten-Bereinigung
- Bewertung der Vorhersagequalität
- Verständnis von Residuen-Analyse

#### Technologien

- pandas, numpy
- scikit-learn (DecisionTreeRegressor, train_test_split)
- matplotlib, seaborn

---

### 3. 🎬 Movies — Stratified vs. Random Sampling

**Typ:** Data Sampling | **Level:** Beginner | **Format:** Jupyter Notebook

> 💾 **Notebook:** [`notebooks/movies/Movies_Machine_Learning_StratifiedSample.ipynb`](../notebooks/movies/Movies_Machine_Learning_StratifiedSample.ipynb)  
> 📊 **Dataset:** [Kaggle - IMDB Movies](https://www.kaggle.com/datasets)  
> 📖 **Detaillierte Dokumentation:** [MOVIES_STRATIFIED_SAMPLE.md](./MOVIES_STRATIFIED_SAMPLE.md)

#### Projektziel

Vergleich von **Stratified Sampling** und **Random Sampling** beim Erstellen von Trainings- und Test-Sets. Zeigt, warum stratifizierte Aufteilung bei unbalancierten Daten wichtig ist.

#### Highlights

- ✅ **Vergleich verschiedener Sampling-Strategien**
- ✅ **Visualisierung von Daten-Verteilungen**
- ✅ **Praktische Beispiele** mit IMDB-Daten
- ✅ **Best Practices** für Train-Test-Split

#### Lernziele

- Unterschied zwischen Random und Stratified Sampling
- Bedeutung von repräsentativen Trainings-Sets
- Umgang mit unbalancierten Daten
- Visualisierung von Datenverteilungen

#### Technologien

- pandas, numpy
- scikit-learn (train_test_split mit stratify)
- matplotlib

---

## 🔄 ML-Workflow 2026

Alle Projekte folgen einem konsistenten, modernen Machine-Learning-Workflow:

```
1. 📥 Data Ingestion
   ├── Typ-sicheres Laden mit Validierung
   └── Path-Handling mit pathlib

2. 🔍 Exploratory Data Analysis (EDA)
   ├── Verteilungen analysieren
   ├── Korrelationen identifizieren
   └── Visual Checks (Histogramme, Scatterplots)

3. 🔧 Preprocessing
   ├── Categorical → OneHot (handle_unknown='ignore')
   ├── Numerical → Scaler + Imputer
   └── Pipeline-basierte Transformation

4. 🎯 Model Training
   ├── Train-Test-Split (stratified wenn nötig)
   ├── Cross-Validation
   └── Hyperparameter-Optimierung (GridSearch/RandomizedSearch)

5. 📊 Evaluation
   ├── Metriken: RMSE, R², MAE
   ├── Residual-Analyse
   └── Feature Importance

6. 💾 Model Persistence
   ├── Modell speichern (joblib)
   └── Reproduzierbarkeit (random_state, Versioning)
```

> 📖 **Mehr Details:** [ML_WORKFLOW.md](./ML_WORKFLOW.md)

---

## 🐳 Big Data Projekte (PySpark)

Für größere Datasets nutzen wir **Apache Spark** in Docker-Containern:

### PySpark — Text Mining mit TF-IDF & K-Means

**Typ:** Clustering (Big Data) | **Level:** Advanced | **Format:** PySpark Script

> 📖 **Detaillierte Dokumentation:** [PYSPARK_TFIDF.md](./PYSPARK_TFIDF.md)  
> 🏗️ **Docker Setup:** [DOCKER_INFO.md](./DOCKER_INFO.md)

#### Projektziel

Demonstration von **Text Mining und Clustering** mit PySpark für große Textdatenmengen.

#### Highlights

- ✅ **Verteilte Verarbeitung** mit Apache Spark
- ✅ **TF-IDF Vektorisierung** für Text
- ✅ **K-Means Clustering** auf großen Datenmengen
- ✅ **Docker-basierte Umgebung** (keine lokale Spark-Installation)

---

## 🎓 Empfohlene Lernreihenfolge

Für Einsteiger empfehlen wir diese Reihenfolge:

1. **Movies — Stratified Sample** (Grundlagen: Sampling)
2. **Movies — Predict NaNs** (ML-basierte Imputation)
3. **AirBnB — Price Prediction** (Full Pipeline)
4. **PySpark — TF-IDF** (Big Data, optional)

---

## 📦 Technologie-Stack

Alle Projekte nutzen einen konsistenten Tech-Stack:

### Core Libraries

- **Python**: >= 3.10
- **pandas**: >= 2.0 (mit PyArrow-Backend optional)
- **numpy**: Numerische Operationen
- **scikit-learn**: >= 1.2 (ML-Algorithmen & Pipelines)

### Visualisierung

- **matplotlib**: Basis-Plotting
- **seaborn**: Statistische Visualisierungen

### Qualitätssicherung

- **Type Hints**: Für bessere Code-Dokumentation
- **logging**: Statt print()-Statements
- **ruff**: Linting & Code-Qualität

### Big Data (optional)

- **PySpark**: >= 3.0 (in Docker)
- **Docker**: Container-Umgebung

---

## 🔗 Verwandte Dokumentation

- **[README.md](../README.md)** — Projekt-Übersicht und Quick Start
- **[INSTALLATION.md](./INSTALLATION.md)** — Detaillierte Installationsanleitung
- **[ML_WORKFLOW.md](./ML_WORKFLOW.md)** — Machine Learning Best Practices
- **[CHANGELOG.md](../CHANGELOG.md)** — Versionshistorie (2020 → 2026)

---

## 💡 Weitere Projekte (geplant)

- **IoT Sensor Data** — Zeitreihen-Analyse
- **NLP Sentiment Analysis** — Text-Klassifikation
- **Computer Vision** — Bild-Klassifikation mit TensorFlow/PyTorch

---

> **Zuletzt aktualisiert:** Februar 2026  
> **Autor:** Andreas Traut
