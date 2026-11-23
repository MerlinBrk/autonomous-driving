# Autonomer Rennwagen (CarRacing-v3)

Modulare Pipeline für die Gymnasium-Simulation [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/):
**Spurerkennung → Pfadplanung → Querregelung → Längsregelung**, umgesetzt nur mit numpy und scipy.

| Messung (`main.py`, 50 Episoden) | Ø Score | Min | Max |
|---|---|---|---|
| normal | **923,4** | 900,0 | 935,2 |
| `--domain_randomize` | **905,6** | 452,5 | 943,7 |

Mindestanforderung: 300. Laufzeit ca. 19 ms pro Schritt (ohne Fenster).

> **Testumgebung:** Alle Messungen und Tests wurden ausschließlich auf einem Mac mit **Apple Silicon (macOS)** durchgeführt. Unter Windows oder Linux wurde das Projekt nicht getestet; Laufzeiten und Scores können dort abweichen.

## Dokumentation

- **[Projektdokumentation.md](Projektdokumentation.md)**: Umsetzung der Module, Parameter, Ergebnisse, Reglervergleich und Grenzen der Lösung
- [Projektbeschreibung.pdf](Projektbeschreibung.pdf): Aufgabenstellung

## Installation

Voraussetzungen: Python ≥ 3.12 und [uv](https://docs.astral.sh/uv/).

```bash
pip install uv
uv sync                               # .venv mit den Versionen aus uv.lock
uv run python test_installation.py    # Simulation öffnet sich, das Auto fährt zufällig
```

## Ausführung

```bash
uv run python main.py                     # Bewertung: 50 Episoden à 1000 Schritte
uv run python main.py --no_display        # ohne Fenster (schneller)
uv run python main.py --domain_randomize  # zufällige Farben für Straße, Gras, Hintergrund
uv run python test_pipeline.py            # Gesamtpipeline mit Debug-Bild und Geschwindigkeitsplot
```

Weitere Test-Skripte je Modul: `test_lane_detection.py`, `test_path_planning.py`, `test_lateral_control.py --mode stanley|pure_pursuit`, `test_longitudinal_control.py` und `compare_lateral_control.py` (Stanley vs. Pure Pursuit). In den interaktiven Skripten wechselt **Leertaste** die Strecke, **Esc** beendet.

Der Querregler wird in [car.py](car.py) gewählt; Standard ist Pure Pursuit:

```python
self._lateral_control = LateralControl(mode="pure_pursuit")   # oder mode="stanley"
```

## Erweiterungen und Parameter

Über die Mindestanforderung hinaus sind die folgenden Erweiterungen umgesetzt. Alle Parameter sind Konstruktor-Argumente der jeweiligen Klasse (die Standardwerte stehen in der Signatur). Begründungen und Ergebnisse stehen in der [Projektdokumentation](Projektdokumentation.md).

| Erweiterung | Aufruf / Parameter | Doku |
|---|---|---|
| **Domain Randomization:** Spurerkennung ohne feste Farbschwellen, die Straßenfarbe wird in jedem Bild geschätzt | `uv run python main.py --domain_randomize`, `uv run python test_lane_detection.py --domain_randomize` | 2.1, 3.1 |
| **Zwei Querregler** (Stanley und Pure Pursuit), per Parameter umschaltbar | `LateralControl(mode="stanley")` oder `LateralControl(mode="pure_pursuit")` in `car.py`; interaktiv `uv run python test_lateral_control.py --mode stanley` bzw. `--mode pure_pursuit` | 2.3 |
| **Reglervergleich** ohne Fenster (Ground-Truth-Pfad, feste Zielgeschwindigkeit, gleiche Strecken für beide Regler) | `uv run python compare_lateral_control.py --speeds 40 60 80 --seeds 10` | 3.2, 3.3 |
| **Tiefpass auf den Lenkwert** gegen Zittern, Startphase ohne Lenkung (die Kamera zoomt anfangs noch heran) | `LateralControl(smoothing=0.3, warmup_steps=50)` | 2.3 |
| **Pfad auch mit nur einem sichtbaren Rand**, Spline-Glättung, Krümmung als Maximum über den sichtbaren Pfad | `PathPlanning(num_waypoints=30, lane_width=20.0)` | 2.2 |
| **Zielgeschwindigkeit aus der Krümmung** (v = √(a_lat / κ)), sofort langsamer, nur allmählich schneller | `LongitudinalControl(v_min=25, v_max=90, a_lat=110)` | 2.4 |
| **PID-Regler** mit D-Anteil auf der Messgröße, Anti-Windup, getrennten Gas- und Bremswerten und Kopplung an die Lenkung | `LongitudinalControl(kp=0.1, ki=0.002, kd=0.02)` | 2.4 |

## Struktur

| Datei | Inhalt |
|---|---|
| `car.py` | verbindet die vier Module zur Pipeline |
| `lane_detection.py` | Spurerkennung über Segmentierung der Straßenfarbe |
| `path_planning.py` | Mittellinie, Spline-Glättung, Krümmung |
| `lateral_control.py` | Querregelung (Stanley, Pure Pursuit) |
| `longitudinal_control.py` | Zielgeschwindigkeit aus Krümmung, PID-Regler |
| `main.py`, `env_wrapper.py`, `input_controller.py` | Vorlage, unverändert |
| `test_*.py`, `compare_lateral_control.py` | Test- und Vergleichsskripte |
| `Projektdokumentation.md` | ausführliche Projektdokumentation |
| `Projektbeschreibung.pdf` | Aufgabenstellung |
