# Copilot-Anweisungen für Machine-Learning-with-Python-Upgrade-2026

Diese Datei definiert projektspezifische Regeln für die Erstellung und Pflege von Dokumentationen im Machine-Learning-with-Python Repository.

---

## 📁 Dateiname und Speicherort

### Regeln für Dokumentationsdateien

- **Hauptdokumentation:** `README.md` im Repository-Root
  - Enthält Projekt-Übersicht, alle Beispiele, Installation und Quick Start
  - Maximal eine README.md pro Repository

- **Changelog:** `CHANGELOG.md` im Repository-Root
  - Dokumentiert alle Änderungen der Upgrade-2026-Version
  - Format: Markdown mit klarer Versionierung

- **Code-Notebooks:** `notebooks/{thema}/`
  - Format: `{Thema}_*.ipynb`
  - Beispiele: `notebooks/movies/`, `notebooks/iot/`
  - Jupyter Notebooks mit sprechenden Namen

- **Code-Scripts:** `scripts/`
  - Format: `{Thema}_*.py`
  - Beispiele: `Sklearn_MachineLearning_AirBnB.py`
  - Python-Dateien für IDE-Nutzung

- **Konfigurationsdateien:**
  - Requirements: `requirements.txt` im Repository-Root
  - Legacy-Code: Ordner `legacy/` für archivierte alte Versionen

### Namenskonventionen

- Markdown-Dateien: GROSSBUCHSTABEN für README.md und CHANGELOG.md
- Python-Module: PascalCase oder snake_case (Konsistenz mit Upgrade-2026)
- Ordnernamen: kleinbuchstaben mit Unterstrichen oder Bindestrichen
- Jupyter Notebooks: Sprechende Namen mit Unterstrichen

---

## 📋 Grundstruktur einer Markdown-Datei

### README.md Struktur

**Pflicht-Komponenten:**

1. **H1-Titel** mit Projektnamen und Version
2. **Einleitungsabsatz** mit Projektbeschreibung
3. **Autor-Informationen** (👨‍💻 Autor, Datum, Version)
4. **Inhaltsverzeichnis** (📋)
5. **Repository-Übersicht** (🎯 Über dieses Repository)
6. **Ordnerstruktur** (📁) - ASCII-Tree Darstellung
7. **Projektziele** (🔍 Ziele: Small Data vs Big Data)
8. **Quick Start Guide** (🚀)
9. **Technologie-Stack** (📦 Unterstützte Versionen)
10. **Module-Beschreibungen** (📚 Dateien in diesem Repository)
11. **Installation & Setup** (🔧)
12. **Lizenz & Beiträge** (📝)

### CHANGELOG.md Struktur

**Pflicht-Komponenten:**

1. **H1-Titel** mit "Changelog - Upgrade 2026"
2. **Versions-Header** (Version + Datum)
3. **Übersicht** der Änderungen
4. **Hauptziele** des Upgrades
5. **Detaillierte Änderungen** gruppiert nach Kategorien:
   - API-Modernisierungen
   - Code-Qualität
   - Reproduzierbarkeit
   - Error Handling
6. **Code-Beispiele** für Alt vs. Neu

### Metadaten-Block

**Verwendung von Blockquotes für Metadaten und wichtige Links:**

**Standard-Layout (immer in dieser Reihenfolge verwenden):**
```markdown
> ➡️ **Details siehe:** [Abschnitt-Titel](#anchor-link)  
> 💼 **[LinkedIn Post: Titel](https://www.linkedin.com/posts/...)**  
> 💾 **Modul:** `scripts/module_name.py` oder `notebooks/thema/notebook.ipynb`
```

**Variationen je nach Kontext:**
- Für README-Sektionen: `➡️ **Details siehe:**` mit internem Link zu einem anderen Abschnitt
- Für Code-Dokumentation: `📖 **Implementierung:**` mit Link zu Code-Dateien
- Optional: `🧠 **Dokumentation:**` für weiterführende Docs (z.B. CHANGELOG.md)
- Datasets: `📊 **Datenquelle:**` mit Link zu Kaggle oder anderen Quellen

**Beispiele:**

Für Notebook-Beschreibungen:
```markdown
> 💾 **Notebook:** [`notebooks/movies/Movies_Machine_Learning_Predict_NaNs.ipynb`](notebooks/movies/Movies_Machine_Learning_Predict_NaNs.ipynb)  
> 📊 **Dataset:** [Kaggle - IMDB Movies](https://www.kaggle.com/datasets)  
> 🎯 **Lernziel:** Umgang mit fehlenden Werten (NaN) in Regressionsaufgaben
```

Für Script-Beschreibungen:
```markdown
> 💾 **Script:** [`scripts/Sklearn_MachineLearning_AirBnB.py`](scripts/Sklearn_MachineLearning_AirBnB.py)  
> 📊 **Dataset:** [Inside AirBnB](http://insideairbnb.com/get-the-data.html)  
> 🔧 **IDE-Empfehlung:** Spyder, PyCharm oder VS Code
```

### Formatierungs-Regeln

- **Emojis:** Nutze thematisch passende Emojis für Überschriften und Aufzählungen
  - 📁 Dateien/Ordner
  - 🚀 Features/Start
  - ✅ Erfolg/Fertig
  - 🔧 Installation/Setup
  - 💾 Code/Module
  - 🧠 KI/Machine Learning
  - 📊 Daten/Analysen
  - 📚 Notebooks/Dokumentation
  - ⚠️ Warnung
  - ❓ Fragen

- **Blockquotes:** Für wichtige Hinweise oder Warnungen
  
- **Listen:**
  - Nutze `-` für unsortierte Listen
  - Nutze `1.` für sortierte Listen (Schritte, Anleitungen)
  - Nutze Checkmarks für Status: ✅ ❌ 🔧

- **Links:**
  - Relative Links zu Repository-Dateien: `[Titel](path/to/file)`
  - Externe Links: `[Titel](https://...)`
  - Kaggle Datasets verlinken

- **Code-Referenzen:**
  - Inline: Backticks für Dateinamen, Funktionen, Variablen
  - Pfade: `notebooks/movies/example.ipynb`
  - Funktionen: `function_name()`
  - Variablen: `VARIABLE_NAME`
  - Klassen: `ClassName`

---

## 💻 Code und SQL-Blöcke

### Python-Code-Blöcke

**Format:**
```python
# Kommentare auf Deutsch, präzise und erklärend
def function_name(param: Type) -> ReturnType:
    """
    Docstring auf Deutsch.
    
    Args:
        param: Beschreibung
        
    Returns:
        Beschreibung
    """
    # Schritt 1: Erklärung
    result = some_operation()
    
    # Schritt 2: Weitere Erklärung
    return result
```

**Regeln:**
- Kommentare immer auf Deutsch
- Funktionen mit Typ-Hints versehen
- Docstrings im Google-Stil (einzeilig für kurze, mehrzeilig mit Args/Returns für komplexe)
- Schritt-für-Schritt Kommentare bei komplexer Logik
- Fehlerbehandlung explizit kommentieren

### Bash/PowerShell-Blöcke

**Format:**
```powershell
# Beschreibung was der Befehl macht
python path/to/script.py --flag value
```

**Regeln:**
- Nutze `powershell` als Sprache für Windows-Befehle
- Nutze `bash` für Linux/Mac
- Jeder Befehl mit einzeiligem Kommentar davor
- Zeige erwartete Ausgabe in separatem Block wenn relevant

### SQL-Blöcke

**Format (falls in Zukunft relevant):**
```sql
-- Beschreibung der Query
SELECT 
    column1,
    column2,
    COUNT(*) as anzahl
FROM 
    table_name
WHERE 
    condition = 'value'
GROUP BY 
    column1, column2
ORDER BY 
    anzahl DESC;
```

**Regeln:**
- Kommentare auf Deutsch mit `--`
- Keywords in GROSSBUCHSTABEN
- Einrückung für Lesbarkeit
- Ein Konzept pro Zeile bei langen Listen

### JSON/YAML-Konfiguration

**Format:**
```json
{
  "key": "value",
  // Kommentar falls unterstützt
  "nested": {
    "detail": "explanation"
  }
}
```

**Regeln:**
- Einrückung mit 2 Spaces
- Deutsche Beschreibungen in String-Werten
- Struktur über Kommentare erklären

---

## ✅ Review und Tests

### Nach Erstellen/Ändern einer Markdown-Datei

**Pflicht-Checks:**

1. **Markdown-Viewer öffnen:**
   - In VS Code: `Ctrl+Shift+V` (Preview)
   - Oder: Rechtsklick → "Open Preview"

2. **Inhaltsverzeichnis prüfen:**
   - Alle Links funktionieren
   - Hierarchie ist korrekt
   - Keine doppelten Anker

3. **Interne Links testen:**
   - Relative Pfade zu anderen Markdown-Dateien
   - Anker-Links innerhalb des Dokuments (#section)
   - Links zu Code-Dateien

4. **Externe Links validieren:**
   - LinkedIn-Posts
   - GitHub-Links
   - Dokumentations-Links

5. **Code-Blöcke prüfen:**
   - Syntax-Highlighting funktioniert
   - Code ist vollständig (keine abgeschnittenen Zeilen)
   - Kommentare sind lesbar

6. **Formatierung:**
   - Überschriften-Hierarchie ist konsistent (H1 → H2 → H3)
   - Listen sind richtig eingerückt
   - Blockquotes werden korrekt dargestellt
   - Emojis werden angezeigt

7. **Mobile/Responsive Check (optional):**
   - Tabellen sind lesbar
   - Lange Code-Zeilen brechen korrekt um

### Vor dem Commit

- Rechtschreibprüfung (Deutsch)
- Prüfe ob alle TODOs entfernt oder als Issues angelegt sind
- Vergleiche mit bestehenden Dokumenten (Konsistenz)

---

## 🔧 Technische Details

### Projekttyp und Kontext

- **Projekttyp:** Python-basiertes Machine-Learning-Lern-Repository
- **Haupt-Technologien:** Python 3.10+, scikit-learn, pandas, numpy, matplotlib, seaborn
- **Infrastruktur:** Windows-Umgebung, lokale Entwicklung oder virtuelle Umgebungen
- **Sprache:** Deutsche Dokumentation, deutsche Code-Kommentare

### Zielgruppe

- **Primär:** Deutsche Data Scientists und ML-Engineers (Anfänger bis Fortgeschrittene)
- **Sekundär:** Studierende und Quereinsteiger in Machine Learning
- **Skill-Level:** Grundkenntnisse Python, Interesse an ML-Workflows

### Schema und Struktur

**Machine Learning Workflow:**
```
Datenquellen (AirBnB, Movies, IoT)
    ↓
Explorative Datenanalyse (EDA)
    ↓
Datenvorverarbeitung (Pipelines)
    ↓
Modelltraining & Evaluation
    ↓
Hyperparameter-Optimierung
    ↓
Modellpersistenz
```

**Modul-Abhängigkeiten:**
- Notebooks sind eigenständig ausführbar
- Scripts benötigen entsprechende Datasets
- Legacy-Code ist archiviert, aber referenziert

### Dokumentations-Prinzipien

1. **Klarheit:** Jedes Beispiel hat klare Lernziele
2. **Verlinkung:** README verlinkt zu Notebooks und Scripts
3. **Praxisnähe:** Immer vollständige Code-Beispiele
4. **Versionierung:** Changelog dokumentiert Upgrade-Änderungen
5. **Reproduzierbarkeit:** Random States und Seeds dokumentiert

### Collation und Encoding

- **Markdown-Encoding:** UTF-8 (für deutsche Umlaute und Emojis)
- **Zeilenenden:** LF (Unix-Style, `.gitattributes` setzen)
- **Einrückung:** Spaces bevorzugt (2 Spaces für JSON/YAML, 4 für Python)

### Best Practices

- Vermeide absolute Pfade in Dokumentation (außer in Code-Beispielen)
- Nutze Umgebungsvariablen für sensitive Daten
- Dokumentiere Breaking Changes prominent
- Halte Code-Beispiele synchron mit tatsächlichem Code
- Versioniere `requirements.txt` klar

---

## 🎯 Zusammenfassung für GitHub Copilot

Wenn du Markdown-Dateien in diesem Projekt erstellst oder bearbeitest:

1. ✅ Nutze die etablierte Ordnerstruktur (`notebooks/`, `scripts/`, `legacy/`)
2. ✅ Folge den Namenskonventionen (README.md, CHANGELOG.md, snake_case für Scripts)
3. ✅ Beginne mit H1-Titel und wichtigen Metadaten
4. ✅ Nutze thematische Emojis konsistent
5. ✅ Schreibe alle Texte auf Deutsch
6. ✅ Kommentiere Code-Blöcke ausführlich
7. ✅ Teste alle Links und das Inhaltsverzeichnis
8. ✅ Halte die Struktur konsistent mit bestehenden Docs

**Wichtigste Frage vor dem Erstellen:** *"Ist diese Dokumentation hilfreich für Lernende im Machine-Learning-Bereich?"*
