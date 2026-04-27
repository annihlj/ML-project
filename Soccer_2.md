# Forschungskonzept: Post-Game-Analyse von Fußballspielen mit Machine Learning

## 1. Einleitung und Zieldefinition

Fußballspiele entstehen durch ein Zusammenspiel aus offensiven Aktionen, defensiver Stabilität, Standardsituationen, Spielverlauf und Zufall. In diesem Projekt wird untersucht, welche im Datensatz verfügbaren Spielmerkmale mit dem Endergebnis eines Fußballspiels zusammenhängen. Im Mittelpunkt steht nicht die Vorhersage eines Spiels vor dem Anpfiff, sondern eine **Post-Game-Analyse**: Das Modell nutzt nach dem Spiel beobachtbare Leistungs- und Ereignismerkmale, um das Ergebnis nachträglich zu erklären und Muster zwischen Spielstatistiken und Resultat sichtbar zu machen.

Die zentrale Zielvariable ist das `Endergebnis` mit den drei Klassen:

- `H` = Heimsieg
- `D` = Unentschieden
- `A` = Auswärtssieg

Der verwendete Datensatz umfasst **3024 Spiele** und **179 Variablen**. In den vorliegenden Daten ist die Liga als `D1` codiert. Der Zeitraum reicht nach Datumsbereinigung von **2015-08-14 bis 2026-04-19**. Die Klassenverteilung lautet aktuell: **{'H': 1350, 'A': 920, 'D': 754}**. Diese Verteilung zeigt, dass die Klassen nicht vollständig gleich verteilt sind; deshalb wird im Projekt geprüft, ob geeignete Strategien für unausgewogene Daten notwendig sind.

### Forschungsfrage

**Inwiefern lassen sich Fußball-Endergebnisse anhand nach dem Spiel verfügbarer Spielstatistiken wie Schüsse, Schüsse aufs Tor, Ecken, Fouls, Karten und Halbzeitstand erklären, und welche Merkmale tragen am stärksten zur Unterscheidung zwischen Heimsieg, Unentschieden und Auswärtssieg bei?**

### Zielsetzung

Das Projekt verfolgt vier Ziele:

1. **Datenverständnis und Datenqualität:** Der Datensatz wird geladen, strukturell untersucht und hinsichtlich fehlender Werte, Datentypen, Ausreißer und möglicher Inkonsistenzen geprüft.
2. **Feature Engineering:** Aus bestehenden Spielstatistiken werden interpretierbare Differenzmerkmale gebildet, z. B. `ShotDiff`, `ShotOnTargetDiff`, `CornerDiff`, `FoulDiff` und `CardDiff`.
3. **Modellvergleich:** Eine interpretierbare logistische Regression dient als Baseline. Anschließend wird ein Random-Forest-Modell als nicht-lineares Vergleichsmodell trainiert.
4. **Evaluation und Interpretation:** Die Modelle werden anhand geeigneter Klassifikationsmetriken bewertet. Im Vordergrund stehen `Macro F1`, `Accuracy`, `Classification Report` und `Confusion Matrix`. Zusätzlich werden wichtige Merkmale interpretiert.

Wichtig ist die methodische Abgrenzung: Da Spielstatistiken wie Schüsse und Ecken erst während bzw. nach dem Spiel bekannt sind, wird das Projekt **nicht als Pre-Game-Prognose**, sondern als erklärende Analyse bereits abgeschlossener Spiele formuliert. Dadurch wird vermieden, dass das Modell als echtes Vorhersagesystem für zukünftige Spiele missverstanden wird.

---

## 2. Beschreibung der Methodik

### 2.1 Forschungsdesign

Das Projekt wird als überwachtes Klassifikationsproblem umgesetzt. Die Zielvariable `Endergebnis` ist diskret und besitzt drei Ausprägungen. Die Eingabevariablen bestehen aus numerischen und kategorialen Spielmerkmalen. Der Fokus liegt auf der Erklärung von Ergebniszusammenhängen und dem Vergleich verschiedener Modellansätze.

Die Vorgehensweise orientiert sich an einem reproduzierbaren Machine-Learning-Prozess:

1. Problemdefinition und Zielvariable festlegen
2. Datensatz laden und erste Strukturprüfung durchführen
3. Explorative Datenanalyse
4. Datenbereinigung
5. Feature Engineering
6. Feature Selection bzw. Reduktionsentscheidungen
7. Train-Test-Split
8. Modelltraining
9. Evaluation mit Klassifikationsmetriken
10. Interpretation und kritische Diskussion

### 2.2 Datenbasis

Die Daten werden aus der Datei `Datensatz_FootballData.csv` geladen. Enthalten sind u. a.:

- Stammdaten: `Liga`, `Datum`, `Uhrzeit`, `Heimmannschaft`, `Gastmannschaft`
- Ergebnisdaten: `Endergebnis_Heim_Tore`, `Endergebnis_Gast_Tore`, `Endergebnis`
- Halbzeitdaten: `Halbzeit_Heim_Tore`, `Halbzeit_Gast_Tore`, `Halbzeit_Ergebnis`
- Spielstatistiken: Schüsse, Schüsse aufs Tor, Fouls, Ecken, gelbe und rote Karten
- Wettquoten verschiedener Anbieter und Marktspalten

Für die erste Modellstufe werden primär Spielstatistiken verwendet. Ergebnisableitende Variablen wie `Endergebnis_Heim_Tore` und `Endergebnis_Gast_Tore` werden nicht als Input verwendet, weil sie die Zielvariable direkt definieren und damit zu Data Leakage führen würden.

### 2.3 Explorative Datenanalyse

Die explorative Datenanalyse dient dazu, die Hauptmerkmale des Datensatzes zu verstehen, Muster und Zusammenhänge zu entdecken, Ausreißer zu erkennen und erste Hypothesen zu formulieren.

Geplante EDA-Schritte:

- Größe und Struktur des Datensatzes prüfen
- Datentypen und fehlende Werte analysieren
- Zielvariable `Endergebnis` visualisieren
- Verteilung zentraler Spielstatistiken untersuchen
- Klassenunterschiede bei Schüssen, Schüssen aufs Tor und Ecken analysieren
- Korrelationen zwischen numerischen Merkmalen prüfen
- Auffälligkeiten bei fehlenden Werten in Wettquoten-Spalten dokumentieren

### 2.4 Datenbereinigung

Die Datenbereinigung umfasst die Behandlung von Fehlern, Inkonsistenzen, fehlenden Werten und irrelevanten Einträgen.

Geplante Bereinigungsschritte:

- Datumswerte vereinheitlichen, da im Datensatz zweistellige und vierstellige Jahreszahlen vorkommen
- Doppelte Zeilen prüfen und ggf. entfernen
- Spalten mit sehr hohem Fehlwertanteil prüfen und ggf. aus dem ersten Modell ausschließen
- Numerische fehlende Werte mit Median-Imputation behandeln
- Kategoriale fehlende Werte mit einer eigenen Kategorie oder dem häufigsten Wert behandeln
- Ergebnisnahe Variablen aus den Modellfeatures ausschließen, um Data Leakage zu vermeiden

### 2.5 Umgang mit unbalancierten Klassen

Da die Zielvariable nicht gleichmäßig verteilt ist, wird die Klassenbalance geprüft. Für die erste Modellierung wird die reale Verteilung zunächst beibehalten. Zusätzlich werden folgende Maßnahmen berücksichtigt:

- Stratified Train-Test-Split
- Bewertung mit `Macro F1`, da alle Klassen gleich stark berücksichtigt werden
- Optional: `class_weight="balanced"` bei der logistischen Regression
- Optional: Vergleich mit und ohne Klassenumgewichtung

### 2.6 Feature Engineering

Aus den Rohdaten werden neue Merkmale erstellt, die Fußballlogik besser abbilden als isolierte Einzelwerte. Besonders relevant sind Differenz-Features zwischen Heim- und Auswärtsteam:

- `ShotDiff = Heim_Schuesse - Gast_Schuesse`
- `ShotOnTargetDiff = Heim_Schuesse_aufs_Tor - Gast_Schuesse_aufs_Tor`
- `CornerDiff = Heim_Ecken - Gast_Ecken`
- `FoulDiff = Heim_Fouls - Gast_Fouls`
- `YellowCardDiff = Heim_Gelbe_Karten - Gast_Gelbe_Karten`
- `RedCardDiff = Heim_Rote_Karten - Gast_Rote_Karten`
- `HalfTimeGoalDiff = Halbzeit_Heim_Tore - Halbzeit_Gast_Tore`

Diese Merkmale sind gut interpretierbar: Ein positiver Wert bedeutet jeweils einen Vorteil bzw. höheren Wert für die Heimmannschaft, ein negativer Wert einen höheren Wert für die Gastmannschaft.

### 2.7 Feature Selection

Im ersten Schritt wird eine fachlich begründete Auswahl weniger, gut interpretierbarer Merkmale verwendet. Anschließend können Korrelationen, Mutual Information oder Modellimportanzen genutzt werden, um relevante Merkmale zu identifizieren und redundante Merkmale zu reduzieren.

Nicht aufgenommen werden zunächst:

- finale Tore, weil sie die Zielvariable direkt bestimmen
- Teamnamen, wenn der Fokus auf allgemeinen Spielmustern statt auf Teamstärke liegt
- sehr lückenhafte Wettquoten-Spalten
- IDs oder rein administrative Felder

### 2.8 Modellierung

Es werden drei Modellvarianten vorgesehen:

#### Dummy Classifier

Ein einfaches Referenzmodell, das z. B. die häufigste Klasse vorhersagt. Es zeigt, welche Leistung ohne echtes Lernen erreichbar ist.

#### Logistische Regression

Die logistische Regression dient als interpretierbare Baseline. Sie eignet sich, um lineare Zusammenhänge zwischen Features und Ergebniswahrscheinlichkeiten zu analysieren. Für numerische Merkmale wird Standardisierung eingesetzt.

#### Random Forest

Der Random Forest dient als nicht-lineares Vergleichsmodell. Er kann komplexere Zusammenhänge und Interaktionen zwischen Spielmerkmalen abbilden. Gleichzeitig lassen sich Feature Importances auswerten.

### 2.9 Evaluation

Da es sich um ein dreiklassiges Klassifikationsproblem handelt, werden mehrere Metriken kombiniert:

- **Accuracy:** Anteil korrekt klassifizierter Spiele
- **Macro F1:** Durchschnittliches F1-Maß über alle Klassen, unabhängig von der Klassengröße
- **Classification Report:** Precision, Recall und F1 je Klasse
- **Confusion Matrix:** Darstellung typischer Fehlklassifikationen, z. B. ob Unentschieden häufig als Heimsieg oder Auswärtssieg klassifiziert werden

Die wichtigste Hauptmetrik ist **Macro F1**, da die Klassen unterschiedlich häufig auftreten und alle Ergebnisarten fair berücksichtigt werden sollen.

### 2.10 Interpretation und kritische Diskussion

Die Ergebnisse werden nicht nur anhand der Modellleistung bewertet, sondern auch fachlich interpretiert. Dabei stehen folgende Fragen im Mittelpunkt:

- Welche Features sind besonders relevant?
- Werden Unentschieden schlechter erkannt als Siege?
- Unterscheiden sich logistische Regression und Random Forest deutlich?
- Gibt es Hinweise auf Data Leakage?
- Welche Grenzen ergeben sich aus der Post-Game-Perspektive?
- Welche Erweiterungen wären für eine echte Pre-Game-Prognose notwendig?

### 2.11 Reproduzierbarkeit

Das Notebook dokumentiert alle Schritte nachvollziehbar. Dazu gehören:

- feste Zufallsvariable `RANDOM_STATE`
- klare Trennung zwischen Datenaufbereitung, Training und Evaluation
- Verwendung von Pipelines
- Speicherung der wichtigsten Metriken in Tabellen
- Kommentierung der getroffenen Entscheidungen

---

## 3. Geplante Notebook-Struktur

1. Titel, Forschungsfrage und Zieldefinition
2. Setup und Imports
3. Daten laden
4. Erste Dateninspektion
5. Datumsbereinigung
6. Zielvariable und Klassenverteilung
7. Fehlende Werte und Datenqualität
8. Explorative Datenanalyse
9. Feature Engineering
10. Definition von X und y
11. Train-Test-Split
12. Preprocessing-Pipeline
13. Dummy Classifier
14. Logistische Regression
15. Random Forest
16. Modellvergleich
17. Confusion Matrix und Classification Report
18. Feature Importance und Interpretation
19. Kritische Diskussion
20. Fazit und nächste Schritte

---

## 4. Literaturhinweise für das Portfolio

- Dixon, M. J., & Coles, S. G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*. Journal of the Royal Statistical Society: Series C.
- Goddard, J. (2005). *Regression models for forecasting goals and match results in association football*. International Journal of Forecasting.
- Ulmer, B., & Fernandez, M. (2014). *Predicting Soccer Match Results in the English Premier League*. Stanford University.
- Baio, G., & Blangiardo, M. (2010). *Bayesian hierarchical model for the prediction of football results*. Journal of Applied Statistics.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
