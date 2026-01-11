# Big Data Analyse: Text Mining & Clustering mit PySpark

Diese Dokumentation beschreibt den Workflow zur Analyse unstrukturierter Textdaten. Wir nutzen dazu die PySpark-Umgebung, um **TF-IDF** Berechnungen durchzuführen und Texte mittels **K-Means Clustering** zu gruppieren.

## 1. Die Datenbasis
Die Quelldaten befinden sich im Ordner `datasets/TF-idf`. Es handelt sich dabei um unstrukturierte Textdateien (z. B. Nachrichtenartikel, E-Mails).

**Ziel:** Der Algorithmus soll verstehen, worum es in den Texten geht, und thematisch ähnliche Texte automatisch gruppieren (Unsupervised Learning).

## 2. Der Prozess: Von Text zu Zahlen

Da Algorithmen keinen Text lesen können, müssen wir die Wörter in Vektoren (Zahlenreihen) umwandeln. Dies geschieht in zwei Schritten innerhalb der Spark-Umgebung.

### Schritt A: Map-Reduce & Tokenisierung
Im Hintergrund nutzt Spark das **Map-Reduce**-Prinzip:
1.  **Map:** Jeder Text wird in seine Bestandteile zerlegt (Wörter/Tokens).
2.  **Shuffle/Reduce:** Spark zählt das Vorkommen jedes Wortes über alle Dokumente hinweg.

### Schritt B: TF-IDF (Term Frequency - Inverse Document Frequency)
Das ist die Kern-Metrik. Sie bewertet, wie wichtig ein Wort für ein bestimmtes Dokument ist. Wir wollen charakteristische Wörter finden und Allerweltswörter ("und", "der") herunterstufen.

Die Berechnung erfolgt in drei mathematischen Schritten:

**1. Term Frequency (TF):**
Wie oft taucht ein Wort $t$ in einem Dokument $d$ auf?
$$TF(t, d) = \frac{\text{Anzahl des Wortes } t \text{ in } d}{\text{Gesamtzahl aller Wörter in } d}$$

**2. Inverse Document Frequency (IDF):**
Wie selten ist das Wort im gesamten Korpus?
$$IDF(t, D) = \log \left( \frac{N}{|\{d \in D : t \in d\}|} \right)$$
*Wobei:*
* $N$: Gesamtzahl der Dokumente.
* $|\{d \in D : t \in d\}|$: Anzahl der Dokumente, die das Wort $t$ enthalten.

**3. Das TF-IDF Gewicht:**
$$TF\text{-}IDF = TF(t,d) \times IDF(t, D)$$

**Der PySpark Code im Notebook setzt dies wie folgt um:**

```python
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
# 1. Text in Wörter zerlegen
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(df)
# 2. Häufigkeit zählen (TF)
HashingTF bildet Wörter auf Vektor-Indizes ab
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)
# 3. Gewichtung berechnen (IDF)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
```python

## 3. Unsupervised Learning: K-Means Clustering
Nachdem die Texte als Vektoren vorliegen, nutzen wir den K-Means Algorithmus, um Strukturen zu finden.
 * Funktionsweise: Der Algorithmus platziert k Mittelpunkte (Zentroiden) im Raum.
 * Minimierung: Ziel ist es, die quadratische Abweichung jedes Punktes zu seinem Cluster-Zentrum zu minimieren:

 J = \sum_{j=1}^{k} \sum_{i=1}^{n} ||x_i^{(j)} - \mu_j||^2
  
 * Ergebnis: Texte mit ähnlichem Wortschatz landen im selben Cluster.
Code-Beispiel:
```python
from pyspark.ml.clustering import KMeans
# Trainiere das Modell mit k=3 Clustern
kmeans = kMeans().setK(3).setSeed(1)
model = kmeans.fit(rescaledData)
# Zeige Vorhersagen
predictions = model.transform(rescaledData)
```

## 4. Visualisierung
Um die Ergebnisse zu überprüfen, wird oft eine Dimensionsreduktion (z.B. PCA) verwendet, um die hochdimensionalen Vektoren auf 2D-Koordinaten herunterzubrechen und als Scatterplot darzustellen.
