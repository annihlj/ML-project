# Vorhersage des Spielausgangs in der Fußball-Bundesliga
## ML – Projekt | I. Eckel (), E. Ekici (), A.Holaj (214922)  | SoSe 2026 | Hochschule Heilbronn

---

## 1. Einleitung und Zieldefinition

### 1.1 Motivation und Problemkontext
Professioneller Fußball ist ein komplexes, dynamisches System, in dem der Spielausgang durch eine Vielzahl von Faktoren beeinflusst wird. Gleichzeitig ist die quantitative Analyse von Fußballergebnissen seit Jahrzehnten Gegenstand wissenschaftlicher Untersuchungen und wird motiviert durch sportanalytische, journalistische und kommerzielle Interessen (vgl. Dixon & Coles, 1997). Mit der zunehmenden Verfügbarkeit strukturierter Spieldaten haben sich maschinelle Lernverfahren als vielversprechende Methode zur systematischen Analyse von Spielverläufen entwickelt.

Die vorliegende Arbeit untersucht, inwiefern verschiedene Kennzahlen von Bundesliga-Spielen der Saison 2015 bis 2025 genutzt werden können, um den Spielausgang (Heimsieg / Unentschieden / Auswärtssieg) zu analysieren und zu erklären. Dabei wird bewusst zwischen einer post-game-analytischen Perspektive (Stufe 1) und einem pre-game-orientierten Ausblick (Stufe 3) unterschieden.

### 1.2 Wissenschaftliche Fragestellungen

Das Projekt gliedert sich in zwei Hauptfragestellungen:

**F1 (Klassifikation):** Lässt sich der Ausgang eines Bundesliga-Spiels (Heimsieg / Unentschieden / Auswärtssieg) anhand von Spielstatistiken wie Schüssen aufs Tor, Ecken und Karten klassifizieren?

**F2 (Feature Importance):** Welche Spielstatistiken besitzen die höchste Vorhersagekraft für den Sieg einer Mannschaft?

**F3 (Regression, optional):** Lässt sich die Anzahl der erzielten Tore anhand der verfügbaren Spielstatistiken schätzen?

### 1.3 Zielhierarchie und Ausbaustufen

**Stufe 1 – Baseline-Klassifikation:** Dreieklassige Klassifikation (H / D / A) anhand von Post-Game-Statistiken. Baseline: Multinomiale Logistische Regression. Hauptmodell: Random Forest.

**Stufe 2 – Regression:** Vorhersage der Toranzahl je Mannschaft als kontinuierliche Zielvariable (MAE, RMSE, Adjusted R²).

**Stufe 3 – Pre-Game-Ausblick:** Nutzung von Wettquoten als Proxy für Teamstärke – wird im Ausblick diskutiert, aber nicht vollständig implementiert.

> **Methodische Klarstellung:** In Stufe 1 werden ausschließlich Post-Game-Statistiken als Features verwendet. Das Modell erklärt statistische Zusammenhänge zwischen Spielverlauf und Ergebnis – es ist kein Pre-Game-Prognosemodell. Diese Einschränkung wird in der Diskussion explizit adressiert.

### 1.4 Datenbasis

Die Datenbasis stammt von [football-data.co.uk](https://www.football-data.co.uk/germanym.php) und umfasst 270 Spielpaarungen der 1. Fußball-Bundesliga der Saison 2025/26 (22.08.2025–19.04.2026) mit 185 Variablen: Spielmetadaten, Endergebnis, Halbzeitdaten, Match-Statistiken sowie Wettquoten von über 15 Anbietern.

---

## 2. Theoretischer Hintergrund

### 2.1 Statistische Modellierung von Fußballergebnissen

Dixon & Coles (1997) zeigten, dass Tore näherungsweise Poisson-verteilt sind und entwickelten ein Modell zur Korrektur von Niedrigtor-Spielen. Baio & Blangiardo (2010) erweiterten dies durch hierarchische Bayes-Modelle mit teamspezifischen Angriffs- und Abwehrstärken. Für die vorliegende Arbeit besonders relevant ist Goddard & Asimakopoulos (2004), die multinomiale Logit-Regression zur Dreieklassen-Vorhersage einsetzten und eine Baseline-Accuracy von ~53 % erzielten.

### 2.2 Maschinelles Lernen für Spielprognosen

Ulmer et al. (2013) verglichen verschiedene ML-Klassifikatoren für Fußballergebnisse und zeigten, dass Ensemble-Methoden konsistent besser abschneiden. Gemäß den Vorlesungsunterlagen (Fanton, 2026) bieten Entscheidungsbäume und Ensembles den Vorteil, keine manuelle Feature-Selection vorauszusetzen und direkt Feature Importances zu liefern – was Forschungsfrage F2 direkt adressiert.

### 2.3 Heimvorteil als strukturelle Variable

Der Heimvorteil ist eines der robustesten Phänomene im Fußball (Pollard, 2006). Dies begründet die Verwendung von Differenz-Features (z. B. `Schuss_Differenz = Heim_Schuesse − Gast_Schuesse`), um relative Dominanz direkt abzubilden.

### 2.4 Das Unentschieden-Problem und Klassenimbalance

In der Bundesliga enden typischerweise ca. 25–28 % aller Spiele unentschieden. Diese Klasse ist schwieriger zu klassifizieren als Siege, da sie keine eindeutige statistische Signatur aufweist. Gemäß Vorlesung (Fanton, 2026, Handout 30.03.26) wird bei unbalancierten Datenbeständen geprüft, ob Klassengewichtung (`class_weight='balanced'`) sinnvoll ist. Der Macro F1-Score wird als Hauptmetrik gewählt.

### 2.5 Literatur

- Baio, G., & Blangiardo, M. (2010). Bayesian hierarchical model for the prediction of football results. *Journal of Applied Statistics*, 37(2), 253–264.
- Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C*, 46(2), 265–280.
- Fanton, N. (2026). *Machine Learning – Vorlesungsunterlagen SoSe 2026*. Hochschule Heilbronn.
- Goddard, J., & Asimakopoulos, I. (2004). Forecasting football results and the efficiency of fixed-odds betting. *Journal of Forecasting*, 23(1), 51–66.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Pollard, R. (2006). Home advantage in soccer. *Journal of Sport Behavior*, 29(2), 169.
- Ulmer, B., Fernandez, M., & Peterson, M. (2013). Predicting soccer match results in the English Premier League. *Stanford CS229 Technical Report*.

---

## 3. Beschreibung der Methodik

### 3.1 Überblick über den Arbeitsprozess

Der Arbeitsprozess orientiert sich an der empfohlenen Reihenfolge aus den Vorlesungshandouts (Fanton, 2026, Handout 30.03.26) und folgt dem CRISP-DM-Vorgehensmodell:

1. Explorative Datenanalyse (EDA)
2. Datenbereinigung und Umgang mit fehlenden Werten / Ausreißern
3. Feature Engineering und Feature Selection
4. Modelltraining und Validierung
5. Evaluation und Diskussion

### 3.2 Explorative Datenanalyse (EDA)

Im Sinne der Vorlesung (Fanton, 2026) wird die EDA eingesetzt, um Hauptmerkmale zu verstehen, Muster zu entdecken, Ausreißer zu identifizieren und erste Hypothesen zu formulieren. Konkret umfasst die EDA:

- Klassenverteilung der Zielvariable `Endergebnis` (H / D / A)
- Deskriptive Statistiken aller numerischen Features
- Fehlende-Werte-Analyse mit Heatmap
- Boxplots der Spielstatistiken stratifiziert nach Spielergebnis
- Korrelationsmatrix der Feature-Variablen
- Heimvorteil-Analyse: Mittlere Tore, Schüsse und Ecken Heim vs. Gast

### 3.3 Datenbereinigung

Entsprechend der Vorlesungsinhalte (Fanton, 2026, Handout 30.03.26, Abschnitt 1) werden folgende Schritte durchgeführt:

**Fehlende Werte:** Spalten mit mehr als 50 % fehlenden Werten werden entfernt (betrifft hauptsächlich BetBrain-Spalten). Für verbleibende Fehlwerte wird Median-Imputation angewendet (robust gegenüber Ausreißern).

**Ausreißer:** Die IQR-Methode dient zur Identifikation. Da es sich um reale Sportdaten handelt (extreme Ergebnisse sind inhärent möglich), werden Ausreißer dokumentiert, aber nur nach sachlicher Begründung entfernt.

**Datenleak-Prüfung:** `Endergebnis_Heim_Tore`, `Endergebnis_Gast_Tore` und `Endergebnis` dürfen nie als Input-Features verwendet werden. Wettquoten werden nur im Ausblick (Stufe 3) berücksichtigt.

### 3.4 Feature Engineering

Basierend auf den Vorlesungsinhalten (Fanton, 2026, Handout 30.03.26, Abschnitt 3 – Feature Construction) werden folgende abgeleitete Merkmale erstellt:

| Feature | Formel | Begründung |
|---|---|---|
| `Schuss_Differenz` | Heim_Schuesse − Gast_Schuesse | Relative Schussdominanz |
| `SchussAufsTor_Differenz` | Heim_Schuesse_aufs_Tor − Gast_Schuesse_aufs_Tor | Erwarteter stärkster Prädiktor |
| `Ecken_Differenz` | Heim_Ecken − Gast_Ecken | Relative Feldüberlegenheit |
| `Fouls_Differenz` | Heim_Fouls − Gast_Fouls | Defensivintensität im Vergleich |
| `GelbeKarten_Differenz` | Heim_Gelbe_Karten − Gast_Gelbe_Karten | Disziplin im Vergleich |
| `Halbzeit_Ergebnis_kodiert` | H=1, D=0, A=−1 | Ordinal kodierter Halbzeitstand |

Positive Differenzwerte zeigen Heimdominanz an, negative Werte Gastdominanz.

### 3.5 Feature Selection

Zur Auswahl relevanter Merkmale kommen gemäß Vorlesung (Fanton, 2026, Handout 30.03.26, Abschnitt 4) folgende Verfahren zum Einsatz:

**Korrelationsbasierte Selektion (CFS):** Berechnung der Korrelation zwischen Features und Zielvariable sowie zwischen Features untereinander zur Identifikation von Redundanzen. Features mit paarweiser Korrelation > 0,85 werden reduziert.

**Mutual Information:** Berechnung der Transinformation mittels `mutual_info_classif` – besonders wertvoll für nicht-lineare Zusammenhänge und kategorische Merkmale.

**Integrierte Feature Importance (Random Forest):** Direkte Auswertung der baumbasierten Feature Importances zur Beantwortung von Forschungsfrage F2.

### 3.6 Modellwahl und Begründung

**Modell 1 – Logistische Regression (Baseline):** Die multinomiale Logistische Regression ist der kanonische Ansatz für Mehrklassen-Klassifikation mit interpretierbaren Koeffizienten (vgl. Goddard & Asimakopoulos, 2004). Die Annahme linearer Entscheidungsgrenzen ist eine bekannte Einschränkung, die in der Diskussion thematisiert wird.

**Modell 2 – Random Forest (Hauptmodell):** Ensemble-Methode auf Basis von Entscheidungsbäumen (Bagging). Gemäß Vorlesung (Fanton, 2026, Handout 20.04.26) kombinieren Ensemble-Methoden mehrere schwächere Modelle für höhere Genauigkeit und Robustheit. Random Forest ist robust gegenüber Multikollinearität, macht keine Verteilungsannahmen und liefert direkt Feature Importances.

**Hyperparameter-Tuning:** Für den Random Forest wird `GridSearchCV` mit 5-Fold Cross-Validation über `n_estimators` (50, 100, 200) und `max_depth` (None, 5, 10) eingesetzt.

### 3.7 Validierungsstrategie

Da Fußballdaten eine inhärente Zeitstruktur aufweisen, wird eine **chronologische Train-Test-Aufteilung** vorgenommen: Spieltage 1–26 (~80 %) als Training, Spieltage 27–34 (~20 %) als Test. Dies verhindert, dass das Modell von zukünftigen Spielen lernt und spiegelt die reale Prognosesituation wider.

Für das Hyperparameter-Tuning wird zusätzlich **StratifiedKFold** (k=5) auf dem Trainingsdatensatz eingesetzt, um die Klassenverteilung in jedem Fold zu erhalten.

### 3.8 Skalierung

Gemäß Vorlesung (Fanton, 2026, Folien 29–33) wird die Frage der Skalierung modellabhängig behandelt:

- **Logistische Regression:** StandardScaler (z-Score-Standardisierung: `x' = (x − μ) / σ`), da der Algorithmus empfindlich gegenüber unterschiedlichen Wertebereichen ist.
- **Random Forest:** Keine Skalierung erforderlich – baumbasierte Methoden sind skalenrobust.

Der Scaler wird ausschließlich auf dem Trainingsdatensatz gefittet und dann auf Test- und Validierungsdaten transformiert, um Data Leakage zu vermeiden.

### 3.9 Evaluationsmetriken

| Metrik | Begründung |
|---|---|
| **Macro F1-Score** | Hauptmetrik: bestraft schwache Unentschieden-Erkennung, ungewichtet über alle Klassen |
| **Accuracy** | Vergleichbarkeit mit Literatur-Benchmarks (~53 % Goddard, ~60 % Random Forest) |
| **Confusion Matrix** | Visualisiert systematische Fehlklassifikationen, insb. bei Unentschieden |
| **Classification Report** | Precision, Recall und F1 pro Klasse (H, D, A) |

---

*Abgabefrist Portfolio: 07.06.2026 | Vorträge: 15.06. und 22.06.2026*