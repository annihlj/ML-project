# Vorhersage des Spielausgangs in der Fußball-Bundesliga
## ML – Projekt | I. Eckel (), E. Ekici (), A.Holaj (214922)  | SoSe 2026 | Hochschule Heilbronn

---

## 1. Einleitung und Zieldefinition

### 1.1 Motivation und Problemkontext
Professioneller Fußball ist ein komplexes, dynamisches System, in dem der Spielausgang durch eine Vielzahl von Faktoren beeinflusst wird. Gleichzeitig ist die quantitative Analyse von Fußballergebnissen seit Jahrzehnten Gegenstand von Datenanalysen. Sie wird aus sportanalytischen, journalistischen und kommerziellen Interessen heraus betrieben (vgl. Dixon & Coles, 1997). 
Die vorliegende Arbeit nutzt einen Datensatz der Fußball-Bundesliga aus einem zehnjährigen Zeitraum (2015–2025), um eine Brücke zwischen der retrospektiven Erklärung (Post-Game-Analyse) und der prospektiven Vorhersagegüte (Pre-Game-Ausblick) zu schlagen.

### 1.2 Wissenschaftliche Fragestellungen

Das Projekt gliedert sich in zwei Hauptfragestellungen:

**F1:** Lässt sich der Ausgang eines Bundesliga-Spiels (Heimsieg / Unentschieden / Auswärtssieg) anhand von Spielstatistiken wie Schüssen aufs Tor, Ecken und Karten klassifizieren?

**F2:** Welche Spielstatistiken besitzen die höchste Vorhersagekraft für den Sieg einer Mannschaft?

**F3:** Mit welcher Präzision lässt sich die absolute Anzahl erzielter Tore auf Basis der erhobenen Variablen modellieren?

### 1.3 Datenbasis

Die Datenbasis stammt von [football-data.co.uk](https://www.football-data.co.uk/germanym.php) und umfasst 3024 Spielpaarungen der 1. Fußball-Bundesliga aus einem zehnjährigen Zeitraum (2015–2025) mit 179 Variablen: Spielmetadaten, Endergebnis, Halbzeitdaten, Match-Statistiken sowie Wettquoten von über 15 Anbietern.

---

## 2. Theoretischer Hintergrund

### 2.1 Statistische Modellierung von Fußballergebnissen

Dixon & Coles (1997) zeigten, dass Tore näherungsweise Poisson-verteilt sind und entwickelten ein Modell zur Korrektur von Niedrigtor-Spielen. Baio & Blangiardo (2010) erweiterten dies durch hierarchische Bayes-Modelle mit teamspezifischen Angriffs- und Abwehrstärken. Für die vorliegende Arbeit besonders relevant ist Goddard & Asimakopoulos (2004), die multinomiale Logit-Regression zur Dreieklassen-Vorhersage einsetzten und eine Baseline-Accuracy von ~53 % erzielten.

### 2.2 Heimvorteil als strukturelle Variable

Der Heimvorteil ist eines der robustesten Phänomene im Fußball (Pollard, 2006). Dies begründet die Verwendung von Differenz-Features (z. B. `Schuss_Differenz = Heim_Schuesse − Gast_Schuesse`), um relative Dominanz direkt abzubilden.


### 2.3 Literatur

- Baio, G., & Blangiardo, M. (2010). Bayesian hierarchical model for the prediction of football results. *Journal of Applied Statistics*, 37(2), 253–264.
- Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C*, 46(2), 265–280.
- Goddard, J., & Asimakopoulos, I. (2004). Forecasting football results and the efficiency of fixed-odds betting. *Journal of Forecasting*, 23(1), 51–66.
- Pollard, R. (2006). Home advantage in soccer. *Journal of Sport Behavior*, 29(2), 169.

---
### 3. Zielsetzung

Das Projekt verfolgt vier Ziele:

1. **Datenverständnis und Datenqualität:** Der Datensatz wird geladen, strukturell untersucht und hinsichtlich fehlender Werte, Datentypen, Ausreißer und möglicher Inkonsistenzen geprüft.
2. **Feature Engineering:** Aus bestehenden Spielstatistiken werden interpretierbare Merkmale gebildet.
3. **Modellvergleich:** Eine interpretierbare logistische Regression dient als Baseline. Anschließend wird ein Random-Forest-Modell als nicht-lineares Vergleichsmodell trainiert. ??????????????
4. **Evaluation und Interpretation:** Die Modelle werden anhand geeigneter Klassifikationsmetriken bewertet. Im Vordergrund stehen `Macro F1`, `Accuracy`, `Classification Report` und `Confusion Matrix`. Zusätzlich werden wichtige Merkmale interpretiert.????????????????

---

## 4. Methodik

### 4.1 Überblick über den Machine Learning Prozess

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

### 4.2 Datenbasis

Die Daten werden aus der Datei `Datensatz_FootballData.csv` geladen. Enthalten sind u. a.: ?????????

- Stammdaten: `Liga`, `Datum`, `Uhrzeit`, `Heimmannschaft`, `Gastmannschaft`
- Ergebnisdaten: `Endergebnis_Heim_Tore`, `Endergebnis_Gast_Tore`, `Endergebnis`
- Halbzeitdaten: `Halbzeit_Heim_Tore`, `Halbzeit_Gast_Tore`, `Halbzeit_Ergebnis`
- Spielstatistiken: Schüsse, Schüsse aufs Tor, Fouls, Ecken, gelbe und rote Karten
- Wettquoten verschiedener Anbieter und Marktspalten

Für die erste Modellstufe werden primär Spielstatistiken verwendet. Ergebnisableitende Variablen wie `Endergebnis_Heim_Tore` und `Endergebnis_Gast_Tore` werden nicht als Input verwendet, weil sie die Zielvariable direkt definieren und damit zu Data Leakage führen würden.

### 4.3 Explorative Datenanalyse (EDA)

Die EDA wird eingesetzt, um Hauptmerkmale zu verstehen, Muster zu entdecken, Ausreißer zu identifizieren und erste Hypothesen zu formulieren. Konkret umfasst die EDA:

- Größe und Struktur des Datensatzes prüfen
- Datentypen und fehlende Werte analysieren
- Zielvariable `Endergebnis` visualisieren
- Verteilung zentraler Spielstatistiken untersuchen
- Klassenunterschiede bei Schüssen, Schüssen aufs Tor und Ecken analysieren
- Korrelationen zwischen numerischen Merkmalen prüfen
- Auffälligkeiten bei fehlenden Werten in Wettquoten-Spalten dokumentier


### 4.4 Datenbereinigung

Die Datenbereinigung umfasst die Behandlung von Fehlern, Inkonsistenzen, fehlenden Werten und irrelevanten Einträgen.

Geplante Bereinigungsschritte:

- Datumswerte vereinheitlichen, da im Datensatz zweistellige und vierstellige Jahreszahlen vorkommen
- Doppelte Zeilen prüfen und ggf. entfernen
- Spalten mit sehr hohem Fehlwertanteil prüfen und ggf. aus dem ersten Modell ausschließen
- Numerische fehlende Werte mit Median-Imputation behandeln
- Kategoriale fehlende Werte mit einer eigenen Kategorie oder dem häufigsten Wert behandeln
- Ergebnisnahe Variablen aus den Modellfeatures ausschließen, um Data Leakage zu vermeiden

### 4.5 Umgang mit unbalancierten Klassen

Da die Zielvariable nicht gleichmäßig verteilt ist, wird die Klassenbalance geprüft. Für die erste Modellierung wird die reale Verteilung zunächst beibehalten. Zusätzlich werden folgende Maßnahmen berücksichtigt:

- Stratified Train-Test-Split
- Bewertung mit `Macro F1`, da alle Klassen gleich stark berücksichtigt werden
- Optional: `class_weight="balanced"` bei der logistischen Regression
- Optional: Vergleich mit und ohne Klassenumgewichtung

### 4.6 Feature Engineering

Aus den Rohdaten werden neue Merkmale erstellt, die Fußballlogik besser abbilden als isolierte Einzelwerte. Besonders relevant sind Differenz-Features zwischen Heim- und Auswärtsteam:

- `ShotDiff = Heim_Schuesse - Gast_Schuesse`
- `ShotOnTargetDiff = Heim_Schuesse_aufs_Tor - Gast_Schuesse_aufs_Tor`
- `CornerDiff = Heim_Ecken - Gast_Ecken`
- `FoulDiff = Heim_Fouls - Gast_Fouls`
- `YellowCardDiff = Heim_Gelbe_Karten - Gast_Gelbe_Karten`
- `RedCardDiff = Heim_Rote_Karten - Gast_Rote_Karten`
- `HalfTimeGoalDiff = Halbzeit_Heim_Tore - Halbzeit_Gast_Tore`

Diese Merkmale sind gut interpretierbar: Ein positiver Wert bedeutet jeweils einen Vorteil bzw. höheren Wert für die Heimmannschaft, ein negativer Wert einen höheren Wert für die Gastmannschaft.

### 4.7 Feature Selection

Im ersten Schritt wird eine fachlich begründete Auswahl weniger, gut interpretierbarer Merkmale verwendet. Anschließend können Korrelationen, Mutual Information oder Modellimportanzen genutzt werden, um relevante Merkmale zu identifizieren und redundante Merkmale zu reduzieren.

Nicht aufgenommen werden zunächst:

- finale Tore, weil sie die Zielvariable direkt bestimmen
- Teamnamen, wenn der Fokus auf allgemeinen Spielmustern statt auf Teamstärke liegt
- sehr lückenhafte Wettquoten-Spalten
- IDs oder rein administrative Felder

### 4.8 Modellierung

Es werden drei Modellvarianten vorgesehen:

#### Dummy Classifier

Ein einfaches Referenzmodell, das z. B. die häufigste Klasse vorhersagt. Es zeigt, welche Leistung ohne echtes Lernen erreichbar ist.

#### Logistische Regression

Die logistische Regression dient als interpretierbare Baseline. Sie eignet sich, um lineare Zusammenhänge zwischen Features und Ergebniswahrscheinlichkeiten zu analysieren. Für numerische Merkmale wird Standardisierung eingesetzt.

#### Random Forest

Der Random Forest dient als nicht-lineares Vergleichsmodell. Er kann komplexere Zusammenhänge und Interaktionen zwischen Spielmerkmalen abbilden. Gleichzeitig lassen sich Feature Importances auswerten.

### 4.9 Evaluation

Da es sich um ein dreiklassiges Klassifikationsproblem handelt, werden mehrere Metriken kombiniert:

- **Accuracy:** Anteil korrekt klassifizierter Spiele
- **Macro F1:** Durchschnittliches F1-Maß über alle Klassen, unabhängig von der Klassengröße
- **Classification Report:** Precision, Recall und F1 je Klasse
- **Confusion Matrix:** Darstellung typischer Fehlklassifikationen, z. B. ob Unentschieden häufig als Heimsieg oder Auswärtssieg klassifiziert werden

Die wichtigste Hauptmetrik ist **Macro F1**, da die Klassen unterschiedlich häufig auftreten und alle Ergebnisarten fair berücksichtigt werden sollen.

### 4.10 Interpretation und kritische Diskussion

Die Ergebnisse werden nicht nur anhand der Modellleistung bewertet, sondern auch fachlich interpretiert. Dabei stehen folgende Fragen im Mittelpunkt:

- Welche Features sind besonders relevant?
- Werden Unentschieden schlechter erkannt als Siege?
- Unterscheiden sich logistische Regression und Random Forest deutlich?
- Gibt es Hinweise auf Data Leakage?
- Welche Grenzen ergeben sich aus der Post-Game-Perspektive?
- Welche Erweiterungen wären für eine echte Pre-Game-Prognose notwendig?

### 4.11 Reproduzierbarkeit

Das Notebook dokumentiert alle Schritte nachvollziehbar. Dazu gehören:

- feste Zufallsvariable `RANDOM_STATE`
- klare Trennung zwischen Datenaufbereitung, Training und Evaluation
- Verwendung von Pipelines
- Speicherung der wichtigsten Metriken in Tabellen
- Kommentierung der getroffenen Entscheidungen

---

## 5. Geplante Notebook-Struktur

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
