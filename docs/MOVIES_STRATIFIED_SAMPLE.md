# Movies Dataset: Stratified Sampling

> 💾 **Notebook:** [`notebooks/movies/Movies_Machine_Learning_StratifiedSample.ipynb`](../notebooks/movies/Movies_Machine_Learning_StratifiedSample.ipynb)  
> 📊 **Dataset:** [Kaggle - IMDB Movies](https://www.kaggle.com/datasets)  
> 🎯 **Lernziel:** Stratifiziertes Sampling für repräsentative Train/Test-Splits

---

## 🎯 Projektziel

Dieses Projekt demonstriert die Bedeutung von **Stratified Sampling** beim Erstellen von Train- und Test-Sets. Ein guter Train/Test-Split stellt sicher, dass beide Sets die **gleiche Verteilung** wichtiger Merkmale aufweisen - besonders wichtig bei ungleich verteilten Daten (z.B. wenige High-Revenue-Filme, viele Low-Revenue-Filme).

## ❓ Warum Stratified Sampling?

### Problem: Random Split kann unrepräsentativ sein

**Beispiel - Revenue-Verteilung:**
- 70% der Filme: $0-50M Revenue
- 20% der Filme: $50-100M Revenue
- 8% der Filme: $100-200M Revenue
- 2% der Filme: $200M+ Revenue

**Bei zufälligem Split könnte passieren:**
- Train-Set: Zu viele Blockbuster → Modell lernt "unrealistische" Muster
- Test-Set: Zu wenige Blockbuster → Evaluation nicht repräsentativ

**Mit Stratified Sampling:**
- Train-Set: 70/20/8/2 Verteilung beibehalten
- Test-Set: 70/20/8/2 Verteilung beibehalten
→ **Beide Sets sind repräsentativ für die Gesamtdaten**

## 🔄 Workflow

### 1. Revenue-Kategorien erstellen

```python
import pandas as pd
import numpy as np

# Revenue in Kategorien einteilen
movies_df['revenue_cat'] = pd.cut(
    movies_df['Revenue'],
    bins=[0, 50, 100, 200, np.inf],
    labels=['0-50M', '50-100M', '100-200M', '200M+']
)

# Verteilung anzeigen
print(movies_df['revenue_cat'].value_counts(normalize=True))
```

**Ausgabe:**
```
0-50M       0.68
50-100M     0.21
100-200M    0.08
200M+       0.03
```

### 2. Visualisierung der Verteilung

```python
import matplotlib.pyplot as plt

# Histogram der Revenue-Kategorien
movies_df['revenue_cat'].value_counts().sort_index().plot(kind='bar')
plt.title('Revenue-Verteilung im gesamten Dataset')
plt.xlabel('Revenue-Kategorie')
plt.ylabel('Anzahl Filme')
plt.tight_layout()
plt.show()
```

### 3. Vergleich: Random vs. Stratified Split

**Random Split:**
```python
from sklearn.model_selection import train_test_split

# Zufälliger Split (NICHT empfohlen für ungleiche Verteilungen)
train_set, test_set = train_test_split(
    movies_df, 
    test_size=0.2, 
    random_state=42
)

print("Random Split - Test Set Verteilung:")
print(test_set['revenue_cat'].value_counts(normalize=True))
```

**Stratified Split:**
```python
from sklearn.model_selection import StratifiedShuffleSplit

# Stratifizierter Split (EMPFOHLEN)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(movies_df, movies_df['revenue_cat']):
    strat_train_set = movies_df.iloc[train_index]
    strat_test_set = movies_df.iloc[test_index]

print("\nStratified Split - Test Set Verteilung:")
print(strat_test_set['revenue_cat'].value_counts(normalize=True))
```

### 4. Vergleich der Verteilungen

```python
import pandas as pd

# Vergleichstabelle erstellen
comparison = pd.DataFrame({
    'Gesamt': movies_df['revenue_cat'].value_counts(normalize=True),
    'Random Split': test_set['revenue_cat'].value_counts(normalize=True),
    'Stratified Split': strat_test_set['revenue_cat'].value_counts(normalize=True)
})

print(comparison)
```

**Ergebnis:**
```
              Gesamt  Random Split  Stratified Split
0-50M          0.68         0.73             0.68
50-100M        0.21         0.17             0.21
100-200M       0.08         0.07             0.08
200M+          0.03         0.03             0.03
```

**Interpretation:**
- ✅ **Stratified Split:** Perfekte Übereinstimmung mit Gesamt-Verteilung
- ⚠️ **Random Split:** Abweichungen bei 0-50M (73% statt 68%) und 50-100M (17% statt 21%)

### 5. Sampling-Bias berechnen

```python
def sampling_bias(strategy_name, test_set):
    """Berechnet Abweichung von der Original-Verteilung"""
    bias = (test_set['revenue_cat'].value_counts(normalize=True) / 
            movies_df['revenue_cat'].value_counts(normalize=True) - 1) * 100
    
    return pd.DataFrame({
        f'{strategy_name} % Fehler': bias
    })

# Vergleich
random_bias = sampling_bias('Random', test_set)
stratified_bias = sampling_bias('Stratified', strat_test_set)

print(pd.concat([random_bias, stratified_bias], axis=1))
```

**Ausgabe:**
```
              Random % Fehler  Stratified % Fehler
0-50M                 +7.4%                  0.0%
50-100M               -19.0%                 0.0%
100-200M              -12.5%                +0.1%
200M+                  0.0%                  0.0%
```

## 📊 Praktisches Beispiel: Pipeline mit Stratified Data

### Kompletter Workflow

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Features definieren
numeric_features = ['Year', 'Score', 'Metascore', 'Vote', 'Runtime']
categorical_features = ['Genre']

# Preprocessing Pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ML Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Training mit stratified train set
X_train = strat_train_set[numeric_features + categorical_features]
y_train = strat_train_set['Revenue']

X_test = strat_test_set[numeric_features + categorical_features]
y_test = strat_test_set['Revenue']

# Model trainieren
model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"RMSE mit Stratified Sampling: ${rmse:.2f}M")
```

## 💡 Best Practices

### ✅ Wann Stratified Sampling verwenden?

1. **Ungleiche Klassenverteilungen:**
   - Wenige High-Revenue-Filme, viele Low-Revenue
   - Seltene Events (Fraud Detection, Krankheitsdiagnose)

2. **Kategorische Zielvariablen:**
   - Bei Klassifikation: Stratify nach Klassen
   - Bei Regression: Kategorien aus kontinuierlicher Variable erstellen

3. **Kleine Datasets:**
   - Jede Klasse muss im Train- UND Test-Set vertreten sein
   - Bei wenigen Samples pro Klasse besonders wichtig

### ⚠️ Häufige Fehler vermeiden

**Fehler 1: Zu viele/zu wenige Kategorien**
```python
# ❌ SCHLECHT: Zu viele Kategorien → zu wenige Samples pro Kategorie
movies_df['revenue_cat'] = pd.cut(movies_df['Revenue'], bins=20)

# ✅ GUT: 3-5 sinnvolle Kategorien
movies_df['revenue_cat'] = pd.cut(
    movies_df['Revenue'],
    bins=[0, 50, 100, 200, np.inf],
    labels=['0-50M', '50-100M', '100-200M', '200M+']
)
```

**Fehler 2: Vergessen, Kategorie-Spalte zu entfernen**
```python
# ❌ SCHLECHT: revenue_cat wird als Feature verwendet
X_train = strat_train_set.drop('Revenue', axis=1)

# ✅ GUT: Sowohl Revenue als auch revenue_cat entfernen
X_train = strat_train_set.drop(['Revenue', 'revenue_cat'], axis=1)
```

**Fehler 3: random_state nicht setzen**
```python
# ❌ SCHLECHT: Nicht reproduzierbar
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

# ✅ GUT: Reproduzierbare Ergebnisse
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
```

## 🔄 Cross-Validation mit Stratification

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Für Klassifikation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model, X, y,
    cv=skf,
    scoring='accuracy'
)

print(f"Cross-Validation Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

## 📈 Vergleich verschiedener Sampling-Strategien

| Strategie | Vorteile | Nachteile | Anwendung |
|-----------|----------|-----------|-----------|
| **Random** | Einfach, schnell | Unrepräsentativ bei ungleichen Verteilungen | Gleich verteilte Daten |
| **Stratified** | Repräsentativ, robust | Kategorien müssen definiert werden | Ungleiche Verteilungen |
| **Time-based** | Realistische Evaluation | Temporal Leakage möglich | Zeitreihen-Daten |
| **Group-based** | Verhindert Data Leakage | Komplex bei vielen Gruppen | Hierarchische Daten |

## 🔗 Siehe auch

- [Movies Predict NaNs Projekt](MOVIES_PREDICT_NANS.md) - Vorhersage fehlender Werte
- [AirBnB Preisvorhersage](AIRBNB_PRICE_PREDICTION.md) - Vollständiger ML-Workflow
- [Machine Learning Workflow](ML_WORKFLOW.md) - Best Practices

## 📚 Ressourcen

- [Scikit-Learn - Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Stratified Sampling erklärt](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)
- [Imbalanced Data Handling](https://imbalanced-learn.org/)
