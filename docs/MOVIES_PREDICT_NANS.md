# Movies Dataset: Vorhersage fehlender Revenue-Werte

> 💾 **Notebook:** [`notebooks/movies/Movies_Machine_Learning_Predict_NaNs.ipynb`](../notebooks/movies/Movies_Machine_Learning_Predict_NaNs.ipynb)  
> 📊 **Dataset:** [Kaggle - IMDB Movies](https://www.kaggle.com/datasets)  
> 🎯 **Lernziel:** Umgang mit fehlenden Werten (NaN) in Regressionsaufgaben

---

## 🎯 Projektziel

Dieses Projekt demonstriert, wie man **fehlende Werte** (NaN) in einem Dataset nicht einfach durch Mediane oder Mittelwerte ersetzt, sondern durch **Machine Learning Modelle vorhersagt**. Statt Annahmen über die fehlenden Revenue-Werte zu treffen, trainieren wir ein Modell, das diese Werte basierend auf anderen Features (Jahr, Score, Genre, etc.) vorhersagt.

## 📊 Dataset

### IMDB Movies Dataset

Das Dataset enthält folgende Spalten:

| Spalte | Beschreibung | Typ |
|--------|--------------|-----|
| Rank | Ranking des Films | Integer |
| Title | Filmtitel | String |
| Year | Erscheinungsjahr | Integer |
| Score | User-Rating | Float |
| Metascore | Kritiker-Score | Float |
| Genre | Film-Genre | String |
| Vote | Anzahl der Votes | Integer |
| Director | Regisseur | String |
| Runtime | Laufzeit in Minuten | Integer |
| **Revenue** | Umsatz in Millionen $ | Float (mit NaN) |
| Description | Filmbeschreibung | String |

### Problem: Fehlende Revenue-Werte

Viele Filme im Dataset haben **keine Revenue-Angaben** (NaN-Werte). Anstatt diese Zeilen zu löschen oder mit einem Durchschnittswert zu füllen, wollen wir die Revenue basierend auf den anderen Features vorhersagen.

## 🔄 Workflow

### 1. Daten laden und explorieren

```python
import pandas as pd
from pathlib import Path

# Daten laden
data_path = Path('../../datasets/movies/movies.csv')
movies_df = pd.read_csv(data_path)

# Erste Inspektion
print(f"Dataset Shape: {movies_df.shape}")
print(f"\nFehlende Revenue-Werte: {movies_df['Revenue'].isnull().sum()}")
```

Der Python-Code zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `import pandas as pd`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://pandas.pydata.org/docs/.


### 2. Daten separieren

**Wichtiger Schritt:** Wir trennen die Daten in:
- **Trainings-/Test-Daten:** Filme MIT Revenue-Werten → für Modelltraining
- **Vorhersage-Daten:** Filme OHNE Revenue-Werte → für spätere Vorhersage

```python
# Filme MIT Revenue (für Training)
movies_with_revenue = movies_df[movies_df['Revenue'].notna()].copy()

# Filme OHNE Revenue (für Vorhersage)
movies_without_revenue = movies_df[movies_df['Revenue'].isna()].copy()

print(f"Filme mit Revenue: {len(movies_with_revenue)}")
print(f"Filme ohne Revenue: {len(movies_without_revenue)}")
```

Der Python-Code zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `# Filme MIT Revenue (für Training)`, mit der der zentrale Schritt direkt ausgeführt wird. Das verbessert die Nachvollziehbarkeit, weil der Ablauf klar definiert ist und sich Schritt für Schritt prüfen lässt. Weitere Details stehen in der offiziellen Dokumentation: https://docs.python.org/3/tutorial/.


### 3. Train-Test-Split

Für die Filme mit Revenue erstellen wir ein stratifiziertes Sample:

```python
from sklearn.model_selection import StratifiedShuffleSplit

# Revenue-Kategorien für Stratified Sampling
movies_with_revenue['revenue_cat'] = pd.cut(
    movies_with_revenue['Revenue'],
    bins=[0, 50, 100, 200, np.inf],
    labels=['0-50M', '50-100M', '100-200M', '200M+']
)

# Stratified Split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(movies_with_revenue, movies_with_revenue['revenue_cat']):
    strat_train_set = movies_with_revenue.iloc[train_idx]
    strat_test_set = movies_with_revenue.iloc[test_idx]
```

Der Python-Code zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `from sklearn.model_selection import StratifiedShuffleSplit`, mit der der zentrale Schritt direkt ausgeführt wird. Damit wird die Datenvorbereitung, das Training oder die Auswertung reproduzierbar und für weitere Experimente leicht wiederverwendbar. Weitere Details stehen in der offiziellen Dokumentation: https://scikit-learn.org/stable/.


### 4. Pipeline erstellen

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Numerische Features
numeric_features = ['Year', 'Score', 'Metascore', 'Vote', 'Runtime']

# Kategorische Features
categorical_features = ['Genre']

# Numerische Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Kategorische Pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Kombinierte Pipeline
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features)
])
```

Der Python-Code zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `from sklearn.pipeline import Pipeline`, mit der der zentrale Schritt direkt ausgeführt wird. Damit wird die Datenvorbereitung, das Training oder die Auswertung reproduzierbar und für weitere Experimente leicht wiederverwendbar. Weitere Details stehen in der offiziellen Dokumentation: https://scikit-learn.org/stable/.


### 5. Modelle trainieren und vergleichen

**DecisionTreeRegressor:**
```python
from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(random_state=42, max_depth=10)
tree_model.fit(X_train_prepared, y_train)

# Evaluation
train_predictions = tree_model.predict(X_train_prepared)
test_predictions = tree_model.predict(X_test_prepared)

train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f"DecisionTree RMSE (Train): ${train_rmse:.2f}M")
print(f"DecisionTree RMSE (Test): ${test_rmse:.2f}M")
```

Der Python-Code zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `from sklearn.tree import DecisionTreeRegressor`, mit der der zentrale Schritt direkt ausgeführt wird. Damit wird die Datenvorbereitung, das Training oder die Auswertung reproduzierbar und für weitere Experimente leicht wiederverwendbar. Weitere Details stehen in der offiziellen Dokumentation: https://scikit-learn.org/stable/.


**RandomForestRegressor:**
```python
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=15,
    min_samples_split=5
)
forest_model.fit(X_train_prepared, y_train)

# Evaluation
test_predictions = forest_model.predict(X_test_prepared)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f"RandomForest RMSE (Test): ${test_rmse:.2f}M")
```

Der Python-Code zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `from sklearn.ensemble import RandomForestRegressor`, mit der der zentrale Schritt direkt ausgeführt wird. Damit wird die Datenvorbereitung, das Training oder die Auswertung reproduzierbar und für weitere Experimente leicht wiederverwendbar. Weitere Details stehen in der offiziellen Dokumentation: https://scikit-learn.org/stable/.


### 6. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    forest_model, X_train_prepared, y_train,
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=42
)

rmse_scores = np.sqrt(-scores)
print(f"Cross-Validation RMSE: ${rmse_scores.mean():.2f}M (+/- ${rmse_scores.std():.2f}M)")
```

Der Python-Code zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `from sklearn.model_selection import cross_val_score`, mit der der zentrale Schritt direkt ausgeführt wird. Damit wird die Datenvorbereitung, das Training oder die Auswertung reproduzierbar und für weitere Experimente leicht wiederverwendbar. Weitere Details stehen in der offiziellen Dokumentation: https://scikit-learn.org/stable/.


### 7. Fehlende Revenue-Werte vorhersagen

```python
# Vorbereitung der Daten ohne Revenue
X_predict = movies_without_revenue[numeric_features + categorical_features]
X_predict_prepared = preprocessor.transform(X_predict)

# Vorhersage mit bestem Modell
predicted_revenue = forest_model.predict(X_predict_prepared)

# Ergebnisse speichern
movies_without_revenue['Revenue_Predicted'] = predicted_revenue

# Top 10 vorhergesagte Revenues
print("\nTop 10 vorhergesagte Revenues:")
print(movies_without_revenue[['Title', 'Year', 'Revenue_Predicted']]
      .sort_values('Revenue_Predicted', ascending=False)
      .head(10))
```

Der Python-Code zeigt einen konkreten Arbeitsschritt des beschriebenen Workflows. Im Fokus steht hier die Zeile `# Vorbereitung der Daten ohne Revenue`, mit der der zentrale Schritt direkt ausgeführt wird. Damit wird die Datenvorbereitung, das Training oder die Auswertung reproduzierbar und für weitere Experimente leicht wiederverwendbar. Weitere Details stehen in der offiziellen Dokumentation: https://docs.python.org/3/tutorial/.


## 📈 Ergebnisse und Erkenntnisse

### Model Performance

| Modell | Train RMSE | Test RMSE | Overfitting? |
|--------|-----------|-----------|--------------|
| DecisionTree (depth=10) | ~$25M | ~$80M | ⚠️ Ja |
| RandomForest (n=100) | ~$35M | ~$65M | ✅ Besser |

### Wichtige Features

Die wichtigsten Features für die Revenue-Vorhersage:
1. **Vote** - Anzahl der User-Ratings (Popularität)
2. **Year** - Neuere Filme tendenziell höhere Revenue
3. **Score** - Bessere Filme verdienen mehr
4. **Runtime** - Längere Filme korrelieren mit höherer Revenue
5. **Genre** - Action/Sci-Fi meist höhere Revenue als Drama

## 💡 Learnings

### ✅ Was funktioniert gut:

- **Stratified Sampling** stellt sicher, dass Train/Test-Sets repräsentativ sind
- **RandomForest** reduziert Overfitting im Vergleich zu DecisionTree
- **Pipeline** ermöglicht saubere, wiederholbare Datenverarbeitung
- **Cross-Validation** gibt realistischere Performance-Schätzung

### ⚠️ Herausforderungen:

- **Fehlende Daten in mehreren Spalten** (Metascore, Director) müssen behandelt werden
- **Revenue-Vorhersage ist schwierig** - viele externe Faktoren (Marketing, Konkurrenz)
- **Outliers** (Blockbuster vs. Indie-Filme) beeinflussen Modell stark
- **Genre als Text** - OneHotEncoding erzeugt viele Features

### 🚀 Verbesserungsmöglichkeiten:

1. **Feature Engineering:**
   - Director-Popularität (Anzahl erfolgreicher Filme)
   - Genre-Kombinationen (Action-Comedy)
   - Sequel-Flag (ist es eine Fortsetzung?)

2. **Advanced Models:**
   - Gradient Boosting (XGBoost, LightGBM)
   - Neuronale Netze für komplexere Muster

3. **Externe Daten:**
   - Budget-Informationen
   - Marketing-Ausgaben
   - Soziale Medien Buzz

## 🔗 Siehe auch

- [Movies Stratified Sample Projekt](MOVIES_STRATIFIED_SAMPLE.md) - Stratified Sampling im Detail
- [AirBnB Preisvorhersage](AIRBNB_PRICE_PREDICTION.md) - Vollständiger ML-Workflow
- [Machine Learning Workflow](ML_WORKFLOW.md) - Detaillierter Workflow-Guide

## 📚 Ressourcen

- [Scikit-Learn - Handling Missing Data](https://scikit-learn.org/stable/modules/impute.html)
- [Kaggle - IMDB Movies Dataset](https://www.kaggle.com/datasets)
- [Feature Engineering Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
