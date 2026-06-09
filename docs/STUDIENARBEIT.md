# MACHINE LEARNING MIT PYTHON – UPGRADE 2026

---

**Studienarbeit**

*Modernisierung eines Legacy-Machine-Learning-Projekts (2020) zu einem strukturierten, reproduzierbaren und wartbaren Workflow nach Best Practices 2026*

---

**Autor:** Andreas Traut

---

> 🚀 **Projektübersicht & Einstieg:**  
> [`README.md`](../README.md)
>
> 📚 **Projektlandschaft und Lernziele:**  
> [`docs/PROJECTS.md`](./PROJECTS.md)
>
> 🔄 **Methodischer ML-Prozess (2026):**  
> [`docs/ML_WORKFLOW.md`](./ML_WORKFLOW.md)
>
> 📦 **Setup und Ausführung:**  
> [`docs/INSTALLATION.md`](./INSTALLATION.md)

---

## Inhaltsverzeichnis

[Exposé](#exposé)

[Gliederung](#gliederung)

[1. Einleitung](#1-einleitung)
- [1.1 Problemstellung](#11-problemstellung)
- [1.2 Zielsetzung](#12-zielsetzung)
- [1.3 Relevanz und Kontext](#13-relevanz-und-kontext)
- [1.4 Abgrenzung der Rolle](#14-abgrenzung-der-rolle)
- [1.5 Theoretische Einordnung](#15-theoretische-einordnung)
- [1.6 CRISP-DM als Leitmodell](#16-crisp-dm-als-leitmodell)

[2. Analyse der Ausgangslage und Datenanforderungen](#2-analyse-der-ausgangslage-und-datenanforderungen)
- [2.1 Ausgangszustand des Legacy-Projekts](#21-ausgangszustand-des-legacy-projekts)
- [2.2 Datenquellen und Datenschnittstellen](#22-datenquellen-und-datenschnittstellen)
- [2.3 Anforderungen an Datenqualität und Reproduzierbarkeit](#23-anforderungen-an-datenqualität-und-reproduzierbarkeit)

[3. Datenaufbereitung und Feature Engineering](#3-datenaufbereitung-und-feature-engineering)
- [3.1 Pipeline-Design für Small-Data-Use-Cases](#31-pipeline-design-für-small-data-use-cases)
- [3.2 Feature Engineering in den Fallstudien](#32-feature-engineering-in-den-fallstudien)
- [3.3 Umgang mit fehlenden Werten und Sampling](#33-umgang-mit-fehlenden-werten-und-sampling)

[4. Modellentwicklung und Implementierung](#4-modellentwicklung-und-implementierung)
- [4.1 Auswahl geeigneter Modellansätze](#41-auswahl-geeigneter-modellansätze)
- [4.2 Umsetzung in Python](#42-umsetzung-in-python)
- [4.3 Training, Validierung und Modellvergleich](#43-training-validierung-und-modellvergleich)
- [4.4 Ableitung von Handlungsempfehlungen](#44-ableitung-von-handlungsempfehlungen)

[5. Evaluation und Qualitätssicherung](#5-evaluation-und-qualitätssicherung)
- [5.1 Technische Evaluation](#51-technische-evaluation)
- [5.2 Teststrategie im Repository](#52-teststrategie-im-repository)
- [5.3 Nutzenbewertung für Lern- und Projektziele](#53-nutzenbewertung-für-lern--und-projektziele)
- [5.4 Iterative Verbesserung](#54-iterative-verbesserung)

[6. Operationalisierung und Dokumentation](#6-operationalisierung-und-dokumentation)
- [6.1 Projektstruktur und Modularität](#61-projektstruktur-und-modularität)
- [6.2 Dokumentationsarchitektur](#62-dokumentationsarchitektur)
- [6.3 Übertragbarkeit auf Big-Data-Szenarien](#63-übertragbarkeit-auf-big-data-szenarien)
- [6.4 Perspektive für Automatisierung und Deployment](#64-perspektive-für-automatisierung-und-deployment)

[7. Zusammenfassung und Ergebnisse](#7-zusammenfassung-und-ergebnisse)
- [7.1 Ergebnisdarstellung](#71-ergebnisdarstellung)
- [7.2 Kernerkenntnisse](#72-kernerkenntnisse)
- [7.3 Beitrag zur datengestützten Entscheidungsfindung](#73-beitrag-zur-datengestützten-entscheidungsfindung)
- [7.4 Limitationen und Ausblick](#74-limitationen-und-ausblick)

[Abbildungsverzeichnis](#abbildungsverzeichnis)

[Literaturverzeichnis](#literaturverzeichnis)

---

## Exposé

Diese Studienarbeit beschreibt die strukturierte Modernisierung eines ursprünglich skriptzentrierten Machine-Learning-Projekts zu einer dokumentierten, nachvollziehbaren und wartbaren Projektbasis nach dem Stand 2026. Im Zentrum steht nicht nur die modelltechnische Umsetzung, sondern die durchgängige Professionalisierung des gesamten Workflows: von der Problemformulierung über die Datenaufbereitung und Modellierung bis hin zu Evaluation, Dokumentation und Reproduzierbarkeit.

Das Repository verbindet mehrere Fallstudien mit unterschiedlichem methodischen Schwerpunkt: eine vollständige Regressionspipeline zur Preisvorhersage, eine ML-basierte Imputation fehlender Werte sowie den Vergleich von Sampling-Strategien. Dadurch wird ein praxisnahes Lern- und Demonstrationssystem geschaffen, das sowohl für Ausbildung als auch für den Transfer in produktionsnahe Data-Science-Prozesse geeignet ist.

Die Arbeit orientiert sich methodisch am CRISP-DM-Modell und zeigt, wie moderne Python-Werkzeuge (u. a. pandas, scikit-learn, Pipeline-Ansätze, strukturierte Dokumentation und testbare Projektstruktur) dazu beitragen, Qualität und Wiederverwendbarkeit von ML-Projekten nachhaltig zu erhöhen.

---

## Gliederung

1. **Einleitung**
   - Problemstellung der Legacy-zu-Modern-Transformation
   - Zielsetzung, Kontext und methodische Einordnung

2. **Analyse der Ausgangslage und Datenanforderungen**
   - Bewertung des Ausgangsprojekts
   - Datenbasis und Qualitätsanforderungen

3. **Datenaufbereitung und Feature Engineering**
   - Workflow-Design für typische Small-Data-Szenarien
   - NaN-Behandlung, Kodierung und Sampling

4. **Modellentwicklung und Implementierung**
   - Modellwahl, Training und Validierung
   - Umsetzung in Python-Skripten und Notebooks

5. **Evaluation und Qualitätssicherung**
   - Metriken, Tests und iterative Verbesserung

6. **Operationalisierung und Dokumentation**
   - Struktur, Nachvollziehbarkeit, Skalierungsperspektive

7. **Zusammenfassung und Ergebnisse**
   - Kernerkenntnisse, Limitationen, Ausblick

---

## 1. Einleitung

### 1.1 Problemstellung

Viele gewachsene ML-Projekte starten als explorative Notebook- oder Skript-Sammlung. Ohne klare Struktur entstehen mittelfristig Probleme bei Wartbarkeit, Wiederverwendung und Qualitätssicherung. Das betrifft insbesondere Datenvorverarbeitung, Feature Engineering, konsistente Modellvalidierung und nachvollziehbare Dokumentation.

### 1.2 Zielsetzung

Ziel dieser Arbeit ist die methodische und technische Weiterentwicklung eines Legacy-Bestands zu einem robusten ML-Setup. Der Fokus liegt auf:

- konsistenter Projektstruktur,
- reproduzierbaren Workflows,
- dokumentierten Fallstudien,
- sauberer Trennung von Lernmaterial, Legacy-Artefakten und modernisiertem Code.

### 1.3 Relevanz und Kontext

Die Arbeit adressiert einen häufigen realen Transformationsfall: vorhandener Code soll nicht verworfen, sondern in moderne Engineering-Standards überführt werden. Das Projekt dient als Blaupause für Teams, die bestehende Data-Science-Artefakte stabilisieren und professionalisieren möchten.

### 1.4 Abgrenzung der Rolle

Die Arbeit geht über reine deskriptive Datenanalyse hinaus. Neben EDA werden prädiktive Verfahren, reproduzierbare Trainingsabläufe und evaluierbare Modellvergleiche behandelt. Damit liegt der Schwerpunkt im Bereich Data Science / ML Engineering.

### 1.5 Theoretische Einordnung

Die Umsetzung kombiniert statistische Analyse, überwachte Lernverfahren und strukturierte Modellbewertung. Neben Modellgüte steht der Engineering-Anteil im Vordergrund: klare Zuständigkeiten von Daten, Code, Doku und Tests.

### 1.6 CRISP-DM als Leitmodell

Die Umsetzung folgt dem CRISP-DM-Gedanken in angepasster Form:

- Business/Problem Understanding,
- Data Understanding,
- Data Preparation,
- Modeling,
- Evaluation,
- Deployment-/Operationalisierungsperspektive.

---

## 2. Analyse der Ausgangslage und Datenanforderungen

### 2.1 Ausgangszustand des Legacy-Projekts

Das Repository enthält weiterhin historische Materialien zur Nachvollziehbarkeit, trennt diese jedoch klar von der modernisierten Struktur. Die neue Organisation reduziert technische Schulden und verbessert Orientierung sowie Erweiterbarkeit.

### 2.2 Datenquellen und Datenschnittstellen

Die Fallstudien basieren auf tabellarischen Datensätzen aus dem ML-Lernkontext (u. a. AirBnB- und Film-Datensätze). Relevante Datenarbeit umfasst Laden, Bereinigung, Typbehandlung und transformationsfähige Aufbereitung.

> 📖 Siehe dazu: [`docs/PROJECTS.md`](./PROJECTS.md)

### 2.3 Anforderungen an Datenqualität und Reproduzierbarkeit

Wesentliche Anforderungen sind:

- konsistente Behandlung fehlender Werte,
- robuste Train/Test-Aufteilung,
- nachvollziehbare Transformationen,
- reproduzierbare Abläufe über definierte Abhängigkeiten und klare Projektstruktur.

> 📦 Setup-Referenz: [`docs/INSTALLATION.md`](./INSTALLATION.md)

---

## 3. Datenaufbereitung und Feature Engineering

### 3.1 Pipeline-Design für Small-Data-Use-Cases

Im Zentrum steht die strukturierte Verarbeitung heterogener Features (numerisch/kategorial) mit wiederverwendbaren Pipeline-Elementen. Dadurch werden manuelle Einzelschritte reduziert und Fehlerquellen minimiert.

### 3.2 Feature Engineering in den Fallstudien

Die Fallstudien zeigen projektabhängig unterschiedliche Feature-Strategien, darunter Kodierung kategorialer Merkmale, Skalierung numerischer Variablen und domänenspezifische Auswahl relevanter Prädiktoren.

### 3.3 Umgang mit fehlenden Werten und Sampling

Ein Schwerpunkt liegt auf zwei praxisrelevanten Fragestellungen:

- **ML-basierte NaN-Vorhersage** statt pauschaler Mittelwert-Imputation,
- **Stratified vs. Random Sampling** zur Stabilisierung von Modellbewertungen.

> 📓 Notebooks:  
> [`Movies_Machine_Learning_Predict_NaNs.ipynb`](../notebooks/movies/Movies_Machine_Learning_Predict_NaNs.ipynb)  
> [`Movies_Machine_Learning_StratifiedSample.ipynb`](../notebooks/movies/Movies_Machine_Learning_StratifiedSample.ipynb)

---

## 4. Modellentwicklung und Implementierung

### 4.1 Auswahl geeigneter Modellansätze

Die Modellwahl orientiert sich am Problemtyp (Regression) und an Anforderungen wie Interpretierbarkeit, Robustheit und Lernwert.

### 4.2 Umsetzung in Python

Die produktionsnähere Implementierung erfolgt skriptbasiert und fokussiert auf saubere Struktur, Logging und wiederholbare Ausführung.

> 💾 Referenz-Skript: [`Sklearn_MachineLearning_AirBnB.py`](../scripts/Sklearn_MachineLearning_AirBnB.py)

### 4.3 Training, Validierung und Modellvergleich

Die Validierung umfasst definierte Datensplits, Vergleich relevanter Gütemaße und dokumentierte Bewertung der Ergebnisse. Dadurch wird verhindert, dass Modellentscheidungen auf isolierten Einzelbeobachtungen beruhen.

### 4.4 Ableitung von Handlungsempfehlungen

Die Ergebnisse werden so aufbereitet, dass sie sowohl didaktisch als auch praktisch anschlussfähig sind: Welche Verfahren funktionieren unter welchen Datenbedingungen, und welche Trade-offs ergeben sich zwischen Einfachheit, Performance und Wartbarkeit?

---

## 5. Evaluation und Qualitätssicherung

### 5.1 Technische Evaluation

Die Evaluation erfolgt metrikenbasiert und berücksichtigt Modellfehler, Generalisierungsverhalten und Robustheit gegenüber Dateneigenschaften.

### 5.2 Teststrategie im Repository

Zusätzlich zur Modellbewertung ist eine grundlegende Repository-Testbasis vorhanden, die Struktur und zentrale Python-Abhängigkeiten absichert.

> 🧪 Tests: [`tests/test_basic.py`](../tests/test_basic.py)

### 5.3 Nutzenbewertung für Lern- und Projektziele

Der Nutzen zeigt sich in höherer Nachvollziehbarkeit, besserer Wartbarkeit und klareren Lernpfaden: vom Einstieg über konkrete Fallstudien bis zur methodischen Vertiefung.

### 5.4 Iterative Verbesserung

Die Struktur des Repositories unterstützt schrittweise Erweiterungen (neue Datensätze, zusätzliche Modelle, verbesserte Doku), ohne die bestehende Basis zu destabilisieren.

---

## 6. Operationalisierung und Dokumentation

### 6.1 Projektstruktur und Modularität

Die Trennung in `docs/`, `scripts/`, `notebooks/`, `legacy/` und `tests/` schafft klare Verantwortlichkeiten und erleichtert Teamarbeit sowie Review-Prozesse.

### 6.2 Dokumentationsarchitektur

Die Dokumentationslandschaft bildet Einstieg, Workflow und Fallstudien modular ab. Dadurch kann zielgruppengerecht gelernt werden (Überblick, Deep Dive, technische Umsetzung).

### 6.3 Übertragbarkeit auf Big-Data-Szenarien

Obwohl der Fokus auf Small Data liegt, sind Designprinzipien wie Pipeline-Denken, reproduzierbare Prozesse und saubere Trennung von Schichten auf Big-Data-Kontexte übertragbar.

> 🐳 Big-Data-Kontext: [`docs/DOCKER_INFO.md`](./DOCKER_INFO.md)  
> 🔬 PySpark-Beispiel: [`docs/PYSPARK_TFIDF.md`](./PYSPARK_TFIDF.md)

### 6.4 Perspektive für Automatisierung und Deployment

Perspektivisch lassen sich die gezeigten Workflows um weitergehende CI/CD-, Monitoring- und Packaging-Konzepte ergänzen, um den Weg in produktive Umgebungen zu verkürzen.

---

## 7. Zusammenfassung und Ergebnisse

### 7.1 Ergebnisdarstellung

Die Arbeit zeigt, wie ein historisch gewachsenes ML-Repository in eine moderne, didaktisch nutzbare und technisch belastbare Struktur überführt werden kann.

### 7.2 Kernerkenntnisse

- Struktur und Reproduzierbarkeit sind gleichrangig zur Modellgüte.
- Dokumentation ist ein zentraler Qualitätsfaktor im ML-Kontext.
- Kleine, klar abgegrenzte Fallstudien fördern nachhaltiges Lernen und bessere Wartbarkeit.

### 7.3 Beitrag zur datengestützten Entscheidungsfindung

Das Projekt stärkt die Fähigkeit, ML-Ergebnisse nachvollziehbar abzuleiten, kritisch zu bewerten und in Entscheidungsprozesse einzubetten.

### 7.4 Limitationen und Ausblick

Die Arbeit bildet bewusst keine vollautomatisierte Produktionsplattform ab. Der Ausblick liegt auf weiterer Automatisierung, breiterer Testabdeckung und zusätzlicher Modellvielfalt.

---

## Abbildungsverzeichnis

In diesem Dokument sind keine eigenen Abbildungen enthalten. Visualisierungen und Diagramme befinden sich in den verlinkten Notebooks und Fachdokumentationen.

---

## Literaturverzeichnis

- Chapman, P. et al. (2000): *CRISP-DM 1.0: Step-by-step data mining guide*.
- Géron, A. (2022): *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3rd ed.).
- scikit-learn Developers: *scikit-learn User Guide*. https://scikit-learn.org/stable/user_guide.html
- pandas Developers: *pandas Documentation*. https://pandas.pydata.org/docs/
- Projektinterne Dokumentation: [`README.md`](../README.md), [`docs/PROJECTS.md`](./PROJECTS.md), [`docs/ML_WORKFLOW.md`](./ML_WORKFLOW.md)
