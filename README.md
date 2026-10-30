# Abschlussprojekt: Autonomer Rennwagen

Dieses "Framework" enthält eine grundlegende Struktur für die Umsetzung einer modularen Pipeline für die Gymnasium Car Racing Simulation ([link](https://gymnasium.farama.org/environments/box2d/car_racing/)).

## 0. Requirements

- Python >=3.12
- [uv](https://docs.astral.sh/uv/) (Paket- und Umgebungsmanager für Python)

## 1. Installation

Das Projekt nutzt **uv** zur Verwaltung der Python-Version, der virtuellen Umgebung und der Abhängigkeiten.

### 1.1 uv installieren

```bash
pip install uv
```

Danach mit `uv --version` prüfen, ob uv verfügbar ist.

### 1.2 Projekt einrichten

1. In den Projektordner wechseln: `cd <working-dir>`
2. Umgebung erstellen und alle Pakete installieren: `uv sync`
   - Legt automatisch eine virtuelle Umgebung in `.venv/` an (und lädt bei Bedarf die passende Python-Version herunter).
   - Installiert exakt die Versionen aus `uv.lock`, sodass alle Entwickler dieselbe Umgebung haben.
3. Testen der Installation: `uv run python test_installation.py`. Danach sollte sich die Simulation öffnen und das Fahrzeug zufällig bewegen.

### 1.3 Arbeiten mit uv

- **Skripte ausführen:** `uv run python <datei>.py` (nutzt automatisch die `.venv`, kein manuelles Aktivieren nötig)
- **Paket hinzufügen:** `uv add <paket>` (aktualisiert `pyproject.toml` und `uv.lock`)
- **Paket entfernen:** `uv remove <paket>`
- **Nach einem `git pull`:** `uv sync` ausführen, falls sich `pyproject.toml` oder `uv.lock` geändert haben

> **Wichtig:** `pyproject.toml` und `uv.lock` werden immer mit committet, `uv.lock` wird nie von Hand bearbeitet. Der Ordner `.venv/` gehört **nicht** ins Repository.

## 2. Dateistruktur

Das Projekt ist wie folgt zu strukturieren:

```python
working-dir/
| - .gitignore # Definition von Dateien, die nicht von git getrackt werden sollen
| - car.py # Beschreibt das Fahrzeug und enthält die gesamte Pipeline
| - env_wrapper.py # Wrapper für die Gymnasium Car Racing Simulation
| - input_controller.py # Manuelle Steuerung des Fahrzeugs
| - lane_detection.py # Modul zur Spurerkennung
| - lateral_control.py # Modul zur Querregelung
| - longitudinal_control.py # Modul zur Längsregelung
| - main.py # Hauptdatei, die die Pipeline über 5 Iterationen ausführt
| - path_planning.py # Modul zur Pfadplanung
| - README.md # Diese Datei
| - test_installation.py # Testet die Installation
| - test_lane_detection.py # Testet die Spurerkennung
| - test_lateral_control.py # Testet die Querregelung
| - test_longitudinal_control.py # Testet die Längsregelung
| - test_path_planning.py # Testet die Pfadplanung
| - test_pipeline.py # Testet die gesamte Pipeline
```

## 3. Ausführung

Die test Dateien können mit `uv run python <test-file>.py` ausgeführt werden. Die Simulation wird sich öffnen und das Fahrzeug wird sich entsprechend der Pipeline bewegen. Um das Endergebnis zu simulieren, kann die `main.py` Datei ausgeführt werden. Sie führt die Pipeline über 5 Iterationen aus und berechnet den durchschnittlichen Score.

## 4. Hinweise

Die Dateien `main.py`, `env_wrapper.py` und `input_controller.py` dürfen nicht verändert werden, damit die Ergebnisse vergleichbar bleiben. Alle anderen Dateien können beliebig verändert werden.
