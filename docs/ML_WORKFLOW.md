# Machine Learning Workflow - Detaillierter Leitfaden

> 🎯 **Ziel:** Schritt-für-Schritt Anleitung für strukturierte ML-Projekte  
> 📖 **Implementierung:** Siehe Projekt-Beispiele in [docs/](.)  
> 🧠 **Best Practices:** Basierend auf Upgrade 2026 Standards

---

## 🗺️ Übersicht

Ein typischer **Machine Learning Workflow** besteht aus 9 Hauptphasen:

```
1. Problem definieren
   ↓
2. Daten sammeln & laden
   ↓
3. Explorative Datenanalyse (EDA)
   ↓
4. Datenvorverarbeitung
   ↓
5. Feature Engineering
   ↓
6. Train-Test-Split
   ↓
7. Modelltraining & -evaluation
   ↓
8. Hyperparameter-Optimierung
   ↓
9. Modell-Deployment & Monitoring
```

---

## 1️⃣ Problem definieren

### Fragestellungen klären

**Welche Art von Problem?**
- 📊 **Regression:** Kontinuierliche Werte vorhersagen (Preis, Temperatur)
- 🏷️ **Klassifikation:** Kategorien zuordnen (Spam/Ham, Katze/Hund)
- 🔍 **Clustering:** Gruppen entdecken (Kundensegmente)
- 📈 **Zeitreihen:** Zeitabhängige Vorhersagen (Aktienkurse)

**Erfolgs-Metriken festlegen:**
- Regression: RMSE, MAE, R²
- Klassifikation: Accuracy, Precision, Recall, F1-Score
- Geschäftsziel: Umsatzsteigerung, Kostensenkung

**Beispiel - AirBnB Preisvorhersage:**
```python
# Problem: Regression (Preis vorhersagen)
# Metrik: RMSE (Root Mean Squared Error)
# Ziel: RMSE < $30 für produktives System
# Business-Impact: Bessere Preisempfehlungen für Hosts
```

---

## 2️⃣ Daten sammeln & laden

### Datenquellen identifizieren

**Mögliche Quellen:**
- 📂 CSV/Excel-Dateien
- 🗄️ Datenbanken (SQL, NoSQL)
- 🌐 APIs (REST, GraphQL)
- 📦 Public Datasets (Kaggle, UCI ML Repository)
- 🕷️ Web Scraping

### Daten laden mit Error Handling

```python
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Lädt CSV-Datei mit Fehlerbehandlung.
    
    Args:
        filepath: Pfad zur CSV-Datei
        
    Returns:
        DataFrame mit geladenen Daten
        
    Raises:
        FileNotFoundError: Wenn Datei nicht existiert
        pd.errors.EmptyDataError: Wenn Datei leer ist
    """
    if not filepath.exists():
        logger.error(f"Datei nicht gefunden: {filepath}")
        raise FileNotFoundError(
            f"Bitte Datei herunterladen und speichern unter: {filepath}"
        )
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✓ Daten geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
        return df
    except pd.errors.EmptyDataError:
        logger.error("Datei ist leer")
        raise
    except Exception as e:
        logger.error(f"Fehler beim Laden: {e}")
        raise

# Verwendung
data_path = Path('datasets/airbnb/listings.csv')
df = load_data(data_path)
```

---

## 3️⃣ Explorative Datenanalyse (EDA)

### Datenstruktur verstehen

```python
# Grundlegende Informationen
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nDatentypen:\n{df.dtypes}")
print(f"\nErste Zeilen:\n{df.head()}")

# Statistische Zusammenfassung
print(f"\nStatistik:\n{df.describe()}")

# Fehlende Werte
missing = df.isnull().sum()
print(f"\nFehlende Werte:\n{missing[missing > 0]}")
```

### Visualisierungen erstellen

**Verteilungen:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram für numerische Spalten
df['price'].hist(bins=50, edgecolor='black')
plt.xlabel('Preis')
plt.ylabel('Häufigkeit')
plt.title('Preis-Verteilung')
plt.show()

# Boxplot für Ausreißer-Erkennung
df.boxplot(column='price', by='neighbourhood')
plt.show()
```

**Korrelationen:**
```python
# Korrelationsmatrix
correlation = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Korrelationsmatrix')
plt.tight_layout()
plt.show()
```

**Scatter Matrix:**
```python
from pandas.plotting import scatter_matrix

features = ['price', 'reviews', 'availability', 'minimum_nights']
scatter_matrix(df[features], figsize=(12, 10), alpha=0.6)
plt.show()
```

---

## 4️⃣ Datenvorverarbeitung

### Fehlende Werte behandeln

**Strategien:**
```python
from sklearn.impute import SimpleImputer

# Numerisch: Median (robust gegen Ausreißer)
num_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# Kategorisch: Häufigster Wert
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Konstanter Wert
const_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')

# Alternative: Zeilen/Spalten löschen
df_clean = df.dropna()  # Alle Zeilen mit NaN entfernen
df_clean = df.dropna(axis=1, thresh=0.7 * len(df))  # Spalten mit >30% NaN
```

### Ausreißer behandeln

```python
# IQR-Methode
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Ausreißer entfernen oder clippen
df_no_outliers = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# Oder clippen (cap values)
df['price_clipped'] = df['price'].clip(lower=lower_bound, upper=upper_bound)
```

### Duplikate entfernen

```python
# Duplikate identifizieren
duplicates = df.duplicated()
print(f"Duplikate: {duplicates.sum()}")

# Duplikate entfernen
df_unique = df.drop_duplicates()

# Basierend auf bestimmten Spalten
df_unique = df.drop_duplicates(subset=['id'], keep='first')
```

---

## 5️⃣ Feature Engineering

### Skalierung numerischer Features

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: Mean=0, Std=1 (gut für normalverteilte Daten)
scaler = StandardScaler()
df[['price', 'reviews']] = scaler.fit_transform(df[['price', 'reviews']])

# MinMaxScaler: Werte zwischen 0 und 1
scaler = MinMaxScaler()

# RobustScaler: Robust gegen Ausreißer (verwendet Median)
scaler = RobustScaler()
```

### Encoding kategorischer Features

**OneHotEncoding:**
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown='ignore',  # Wichtig für Production!
    sparse_output=False
)

encoded = encoder.fit_transform(df[['neighbourhood', 'room_type']])
feature_names = encoder.get_feature_names_out()
```

**Label Encoding (nur für ordinale Daten):**
```python
from sklearn.preprocessing import LabelEncoder

# Nur verwenden wenn Reihenfolge wichtig: low < medium < high
df['size_encoded'] = LabelEncoder().fit_transform(df['size'])
```

### Feature Creation

```python
# Neue Features aus bestehenden erstellen
df['reviews_per_month'] = df['number_of_reviews'] / df['months_active']
df['price_per_room'] = df['price'] / df['bedrooms']

# Binning (Diskretisierung)
df['price_category'] = pd.cut(
    df['price'],
    bins=[0, 75, 150, 300, np.inf],
    labels=['budget', 'mid', 'high', 'luxury']
)

# Datum-Features
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
```

---

## 6️⃣ Train-Test-Split

### Einfacher Random Split

```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% für Test-Set
    random_state=42     # Reproduzierbarkeit!
)
```

### Stratified Split (empfohlen bei ungleichen Verteilungen)

```python
from sklearn.model_selection import StratifiedShuffleSplit

# Stratified nach Kategorien
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(X, y_categories):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

### Pipeline für Preprocessing

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Numerische Features
numeric_features = ['age', 'income', 'score']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Kategorische Features
categorical_features = ['city', 'category']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Kombiniert
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Daten transformieren
X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)  # Nur transform!
```

---

## 7️⃣ Modelltraining & -evaluation

### Modelle für Regression

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(kernel='rbf')
}
```

### Modelle für Klassifikation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

models = {
    'Logistic': LogisticRegression(random_state=42),
    'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'SVC': SVC(kernel='rbf', random_state=42),
    'NaiveBayes': GaussianNB()
}
```

### Training & Evaluation Loop

```python
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

results = {}

for name, model in models.items():
    # Training
    model.fit(X_train_prepared, y_train)
    
    # Vorhersagen
    y_train_pred = model.predict(X_train_prepared)
    y_test_pred = model.predict(X_test_prepared)
    
    # Metriken (Regression)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)
    
    results[name] = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'r2_score': r2
    }
    
    print(f"{name}:")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  R²: {r2:.3f}")
    print()
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X_train_prepared, y_train,
    cv=5,  # 5-Fold Cross-Validation
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

rmse_scores = np.sqrt(-scores)
print(f"CV RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")
```

---

## 8️⃣ Hyperparameter-Optimierung

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_prepared, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {np.sqrt(-grid_search.best_score_):.2f}")

best_model = grid_search.best_estimator_
```

### Randomized Search (schneller)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions,
    n_iter=50,  # Anzahl Kombinationen
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_prepared, y_train)
```

---

## 9️⃣ Modell-Deployment & Monitoring

### Modell speichern

```python
import joblib
from pathlib import Path

# Modell speichern
model_path = Path('models/best_model.pkl')
model_path.parent.mkdir(exist_ok=True)
joblib.dump(best_model, model_path)

# Modell laden
loaded_model = joblib.load(model_path)
```

### Preprocessing-Pipeline speichern

```python
# Preprocessing + Model als eine Pipeline
from sklearn.pipeline import Pipeline

full_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', best_model)
])

joblib.dump(full_pipeline, 'models/full_pipeline.pkl')

# Verwendung
loaded_pipeline = joblib.load('models/full_pipeline.pkl')
predictions = loaded_pipeline.predict(new_data)
```

### Predictions auf neuen Daten

```python
# Neue Daten
new_data = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],
    # ...
})

# Vorhersage
prediction = loaded_pipeline.predict(new_data)
print(f"Vorhersage: {prediction[0]:.2f}")
```

---

## ✅ Checkliste für ML-Projekte

### Vor dem Start:
- [ ] Problem klar definiert (Regression/Klassifikation)
- [ ] Success-Metrik festgelegt
- [ ] Datenquelle identifiziert

### Datenphase:
- [ ] Daten geladen mit Error Handling
- [ ] EDA durchgeführt (Visualisierungen)
- [ ] Fehlende Werte behandelt
- [ ] Ausreißer analysiert
- [ ] Duplikate entfernt

### Modeling:
- [ ] Train-Test-Split (stratified wenn nötig)
- [ ] Preprocessing-Pipeline erstellt
- [ ] Mehrere Modelle getestet
- [ ] Cross-Validation durchgeführt
- [ ] Hyperparameter optimiert

### Deployment:
- [ ] Bestes Modell gespeichert
- [ ] Full Pipeline gespeichert
- [ ] Dokumentation erstellt
- [ ] Unit Tests geschrieben

---

## 🔗 Siehe auch

- [AirBnB Preisvorhersage](AIRBNB_PRICE_PREDICTION.md) - Vollständiges Projekt-Beispiel
- [Movies Predict NaNs](MOVIES_PREDICT_NANS.md) - Umgang mit Missing Data
- [Installation Guide](INSTALLATION.md) - Setup für ML-Projekte

## 📚 Ressourcen

- [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Kaggle Learn](https://www.kaggle.com/learn) - Interaktive Kurse
