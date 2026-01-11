# Technische Architektur & Docker Umgebung

Dieses Projekt nutzt eine containerisierte Umgebung, um sicherzustellen, dass alle Abhängigkeiten (Java, Spark, Python-Bibliotheken) korrekt konfiguriert sind. Wir verwenden das Docker-Image `andreastraut/machine-learning-pyspark`.

## 1. Warum Docker?
Apache Spark benötigt ein komplexes Zusammenspiel aus Java (JVM), Scala und Python. Um Installationsprobleme auf lokalen Rechnern (Versionskonflikte, Pfad-Probleme) zu vermeiden, ist die gesamte Laufzeitumgebung in einem Docker-Container verpackt. Das garantiert **Reproduzierbarkeit**: Der Code läuft überall gleich.

## 2. Der "Tech Stack" im Detail

Das Image ist schichtweise aufgebaut (wie ein Stapel Pfannkuchen). Jede Schicht baut auf der vorherigen auf. Hier ist der Blick "unter die Haube" des Dockerfiles:

| Schicht | Technologie | Funktion |
| :--- | :--- | :--- |
| **Top** | **Jupyter Notebook** | Die Benutzeroberfläche im Browser (Port 8888). |
| **Mitte** | **Python (PySpark)** | Der Code-Bereich mit ML-Bibliotheken. |
| **Motor** | **Apache Spark & Hadoop** | Die Engine zur verteilten Datenverarbeitung. |
| **Basis** | **Linux & Java** | Das Betriebssystem und die Laufzeitumgebung. |

### Die Layer im technischen Detail

**1. OS Layer (Das Fundament)**
* **Basis:** Ein minimales Linux (z.B. Ubuntu/Debian).
* **Dockerfile-Logik:** `FROM ubuntu:latest`
* **Tools:** Installation von System-Werkzeugen wie `curl`, `wget` und `git`.

**2. JVM Layer (Die Laufzeitumgebung)**
* **Funktion:** Da Spark in Scala geschrieben ist, läuft im Hintergrund zwingend eine Java Virtual Machine (JVM), auch wenn wir Python nutzen.
* **Installation:** `apt-get install openjdk-8-jdk` (oder ähnlich). Ohne diesen Layer würde Spark nicht starten.

**3. Big Data Engine (Spark & Hadoop)**
* **Download:** Der Ersteller lädt die kompilierten Apache-Pakete via `wget` herunter und entpackt sie (`tar -xvf`).
* **Konfiguration:** Essenzielle Umgebungsvariablen werden fest im Image verankert, damit das System die Pfade findet:
    ```dockerfile
    ENV SPARK_HOME=/usr/local/spark
    ENV PATH=$PATH:$SPARK_HOME/bin
    ```

**4. Python & PySpark Layer**
* **Python:** Installation von Python 3.
* **Libraries:** Über `pip` werden die wichtigsten Data-Science-Pakete vorinstalliert:
    * `pyspark` (Die Brücke zum Spark-Cluster)
    * `numpy` (Numerische Berechnungen)
    * `pandas` (Datenanalyse)
    * `scikit-learn` (Klassisches ML)
    * `matplotlib` / `seaborn` (Visualisierung)

**5. Application Layer (Das Interface)**
* **Jupyter Notebook:** Dient als IDE, um interaktiv Code im Browser auszuführen.
* **Start-Befehl:** Das Dockerfile endet mit dem Befehl, der beim Start ausgeführt wird:
    ```dockerfile
    CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
    ```
    *Hinweis:* `0.0.0.0` ist notwendig, damit der Container Verbindungen von "außen" (deinem Host-PC) akzeptiert.

## 3. Nutzung

Du musst nichts installieren oder konfigurieren. Um die Umgebung zu starten, nutze folgenden Befehl:

```bash
docker pull andreastraut/machine-learning-pyspark
docker run -dp 8888:8888 andreastraut/machine-learning-pyspark:latest
```

 * -d: Startet im Hintergrund (Detached).
 * -p 8888:8888: Leitet den Container-Port auf deinen PC um.
Nach dem Start ist die Umgebung unter http://localhost:8888 im Browser erreichbar.