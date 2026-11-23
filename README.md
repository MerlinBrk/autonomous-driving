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
