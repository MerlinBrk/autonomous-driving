# Projektdokumentation: Autonomer Rennwagen in CarRacing-v3

Ausführliche Beschreibung der Umsetzung, der Messergebnisse und der Grenzen. Kurzüberblick, Installation und Befehle stehen in der [README](README.md), die Aufgabenstellung in [Projektbeschreibung.pdf](Projektbeschreibung.pdf).

## 1. Überblick

Die Pipeline in [car.py](car.py) verarbeitet in jedem Simulationsschritt das 96×96-Kamerabild in vier Modulen:

```
Kamerabild ─► Spurerkennung ─► Pfadplanung ─► Querregelung   ─► Lenkung
              (Randpixel)      (Pfad, κ)   └► Längsregelung ─► Gas / Bremse
```

Die Pipeline nutzt ausschließlich **numpy** und **scipy**. OpenCV und matplotlib werden nur in den Test-Skripten zur Visualisierung verwendet. `main.py`, `env_wrapper.py` und `input_controller.py` sind gegenüber der Vorlage unverändert.

**Koordinatensystem:** Alle Module arbeiten in **Bildkoordinaten** des 96×96-Bildes (x nach rechts, y nach unten, Einheit Pixel). Das Auto steht fest bei (x=48, y=64) und fährt im Bild nach oben (−y). Ein Wechsel in Weltkoordinaten ist dadurch nicht nötig.

---

## 2. Umsetzung der Module

### 2.1 Spurerkennung ([lane_detection.py](lane_detection.py))

**Schnittstelle:** `detect(image) -> (left_boundary, right_boundary)`, jeweils ein `(N, 2)`-Array mit den Pixelkoordinaten `[x, y]` der Randpixel, **ungeordnet** (geordnet wird erst in der Pfadplanung, siehe 2.2). Ein leeres `(0, 2)`-Array bedeutet: Rand nicht sichtbar. Zusätzlich wird `self.debug_image` für die Visualisierung gesetzt (rot = links, blau = rechts, grau = andere Kanten).

**Vorgehen:**

1. **Vorverarbeitung:** Das Bild wird auf die Zeilen 0–76 zugeschnitten (die Statusleiste unten fällt weg). Das Auto wird mit einer festen Maske ausgeblendet.
2. **Straßen-Segmentierung über eine geschätzte Straßenfarbe** (statt fester Farbschwellen):
   - Gras und Hintergrund bilden in CarRacing ein Schachbrett aus zwei Farben, die sich in genau *einem* Kanal unterscheiden (beobachtet ca. 20–26, akzeptiert werden 15–30, `BG_CHANNEL_DIFF`). Dieses Farbpaar wird unter den häufigsten Farben gesucht und als „keine Straße“ markiert.
   - Die Straßenfarbe ist der Median direkt vor dem Auto. Liegt dort Gras oder ein Randstein (Auto neben der Straße), wird die häufigste verbleibende Farbe genommen. Rot/weiße Randsteine werden ausgeschlossen.
   - Ein Pixel gilt als Straße, wenn sein Farbabstand (alle 3 Kanäle) zur Straßenfarbe unter einer Toleranz liegt. Die Toleranz (Standard 14) wird verkleinert, wenn die Hintergrundfarbe der Straßenfarbe ähnelt.
   - **Dadurch funktioniert die Erkennung auch mit `--domain_randomize`** (optionale Erweiterung), ohne Parameter zu ändern.
3. **Aufräumen:** Kleine Löcher (Kantenglättung an Kachelgrenzen) werden gefüllt, kleine Fehl-Blobs entfernt (`scipy.ndimage.label`).
4. **Kantenerkennung:** Kante = Dilatation minus Erosion der Straßenmaske, also ein 2 px breites Band genau am Übergang Straße/Gras. Der Bildrand zählt dabei nicht als Kante, sonst wären linker und rechter Rand über den oberen Bildrand verbunden. Die Kanten werden in zusammenhängende Komponenten zerlegt.
5. **Zuordnung links/rechts:** Von unten nach oben wird die erste Bildzeile mit Straße gesucht und darin das Straßensegment, das dem Auto am nächsten liegt. Dessen linkes und rechtes Ende bestimmen, welche Kantenkomponente der linke bzw. rechte Rand ist. Das funktioniert auch, wenn das Auto neben der Straße steht oder mehrere Streckenteile sichtbar sind.
6. **Ausgabe:** Die Pixel der beiden gewählten Komponenten werden als Koordinatenliste zurückgegeben.

Alle festen Werte (Bildzuschnitt, Fahrzeugmaske, Farbtoleranzen, Mindestgrößen) sind als benannte Klassenkonstanten mit Kommentar oben in `LaneDetection` definiert, z. B. `CROP_HEIGHT`, `CAR_ROWS`/`CAR_COLS`, `ROAD_COLOR_TOL`, `MIN_SEGMENT_WIDTH`.

**Warum dieser Ansatz:** Ein einfacher Gradient auf dem Graukanal (Ansatz aus der Vorlesung) erkennt auch die Gras-Schachbrettkanten und die Randsteine, und er versagt bei zufälligen Farben. Über die Segmentierung der Straßenfläche entstehen Kanten nur dort, wo wirklich Straße auf Nicht-Straße trifft. Die Zuordnung über das Straßensegment vor dem Auto ist robuster als eine reine Links/Rechts-Aufteilung an der Bildmitte, die in engen Kurven falsch zuordnet.

### 2.2 Pfadplanung ([path_planning.py](path_planning.py))

**Schnittstelle:** `plan(left_boundary, right_boundary) -> (way_points, curvature)`. `way_points` ist ein `(30, 2)`-Array, `curvature` die maximale Krümmung κ des Pfads in 1/px.

**Vorgehen:**

1. **Ordnen und Neuabtasten:** Die Randpunkte werden zu einer Polylinie geordnet und mit konstantem Abstand von 2 px neu abgetastet. Damit sind beide Ränder direkt vergleichbar. Das Ordnen beginnt am Punkt nächst dem Auto und springt immer zum nächsten Nachbarn. Dabei werden alle Punkte im Umkreis von 2,5 px zu einem gemittelt; so wird aus dem 2 px breiten Kantenband eine dünne Linie. Bei einer Lücke > 6 px oder einer Richtungsumkehr endet die Linie, so werden andere sichtbare Streckenteile abgeschnitten.

   Das ist die **einzige Stelle, an der Randpunkte geordnet werden**. Früher hat auch die Spurerkennung die Pixel schon geordnet (einfache Nächster-Nachbar-Suche ohne Mitteln und ohne Umkehr-Erkennung). Das war doppelt: Die Pfadplanung muss ohnehin selbst ordnen, weil sie in `test_path_planning.py` direkt die Ground-Truth-Ränder der Simulation bekommt. Deshalb wurde das Ordnen aus der Spurerkennung entfernt.
2. **Mittellinie:**
   - Der Rand mit mehr Punkten ist die Referenz. Für jeden Referenzpunkt wird der nächste Punkt auf dem anderen Rand gesucht.
   - Ist der Abstand plausibel (±50 % der geschätzten Spurbreite), ist der Mittelpunkt des Paares ein Pfadpunkt.
   - Sonst (anderer Rand nicht sichtbar oder unplausibel) wird die Referenz entlang ihrer Normalen um die halbe Spurbreite in Richtung Straße verschoben. So entsteht auch mit **nur einem sichtbaren Rand** ein gültiger Pfad, z. B. in engen Kurven.
   - Die Spurbreite (Start 20 px) wird laufend aus gültigen Paaren mit einem Tiefpass (0,9 / 0,1) nachgeführt.
3. **Glättung:** Durch die Mittelpunkte wird ein glättender kubischer Spline gelegt (`scipy.interpolate.splprep`, s = 4·N) und an 30 Stellen ausgewertet.
4. **Krümmung:** Aus den Spline-Ableitungen: κ = |x′y″ − y′x″| / (x′² + y′²)^(3/2). Zurückgegeben wird das **Maximum** über den sichtbaren Pfad (ohne die Spline-Enden, die Ausreißer erzeugen). So „sieht“ die Längsregelung eine Kurve, bevor das Auto sie erreicht.
5. **Fallback:** Ohne brauchbare Ränder wird der letzte Plan weiterverwendet, ganz am Anfang ein gerader Pfad nach vorn.

**Warum:** Die Mittellinie reicht für die geforderte Leistung (siehe Ergebnisse). Der Spline glättet Pixelrauschen und liefert die Krümmung direkt über die zweite Ableitung.

### 2.3 Querregelung ([lateral_control.py](lateral_control.py))

**Schnittstelle:** `LateralControl(mode="stanley" | "pure_pursuit", ...)`, `control(trajectory, speed) -> steering` in [−1, 1] (+ = rechts).

**Gemeinsame Vorverarbeitung des Pfads:** Der Pfad wird in Fahrtrichtung orientiert, an Sprüngen > 15 px abgeschnitten (andere sichtbare Streckenteile) und mit 1 px Abstand neu abgetastet.

**Stanley-Regler** (`mode="stanley"`):

δ = ψ + arctan( k · e / (k_soft + v) )

- ψ: Kurswinkelfehler zwischen Fahrtrichtung (0, −1) und Pfadtangente. Die Tangente wird über `heading_window` = 24 px gemittelt; kürzere Fenster verstärken das Pixelrauschen und führen auf Geraden zu Schwingungen.
- e: vorzeichenbehafteter Querabstand zum nächsten Pfadpunkt.
- Parameter: `k=3.0`, `k_soft=5.0`.

**Pure-Pursuit-Regler** (`mode="pure_pursuit"`):

δ = arctan( 2 · L · sin(α) / l_d )

- Zielpunkt: erster Pfadpunkt vor dem Auto mit Abstand ≥ l_d.
- Geschwindigkeitsabhängiger Lookahead: l_d = clip(12 + 0,45 · v, 12, 40) px. Längerer Lookahead bei hoher Geschwindigkeit macht das Fahren auf Geraden ruhiger.
- Radstand L = 5 px (ca. 3,2 Welteinheiten × 1,55 px).

**Gemeinsame Erweiterungen:**

| Parameter | Wert | Wirkung |
|---|---|---|
| `smoothing` | 0.3 | Tiefpass auf den Lenkwert gegen Zittern. Werte > 0,5 erzeugen so viel Verzögerung, dass das Auto selbst zu schwingen beginnt. |
| `warmup_steps` | 50 | In den ersten ~50 Schritten zoomt die Kamera noch heran, Pixelabstände stimmen nicht und die Erkennung liefert Unsinn. Das Auto startet gerade auf der Strecke, also wird nicht gelenkt. |
| `max_gap` | 15 px | Größere Punktabstände gelten als anderer Streckenteil und werden abgeschnitten. |
| (kein Pfad) | — | Der letzte Lenkwert klingt mit Faktor 0,9 pro Schritt ab. |

Alle Parameter lassen sich im Konstruktor von `LateralControl` übergeben.

### 2.4 Längsregelung ([longitudinal_control.py](longitudinal_control.py))

**Schnittstelle:** `predict_target_speed(curvature) -> v_target` und `control(speed, target_speed, steering_angle) -> (acceleration, braking)`.

**Zielgeschwindigkeit:**

- Aus der Pfadkrümmung über eine zulässige Querbeschleunigung: v = √(a_lat / κ), begrenzt auf [25, 90].
- `a_lat = 110` wurde empirisch bestimmt; ab ca. 130 verlässt das Auto in Kurven die Strecke.
- **Asymmetrische Glättung:** Wird eine niedrigere Geschwindigkeit verlangt, gilt sie sofort (rechtzeitig bremsen). Höhere Werte werden nur langsam übernommen (5 % pro Schritt), weil die Krümmungsschätzung schwankt.
- Start mit `v_min`, denn „es ist ein Rennauto“.

**PID-Regler** (`kp=0.1`, `ki=0.002`, `kd=0.02`):

- **D-Anteil auf die gemessene Geschwindigkeit statt auf den Fehler.** Dadurch gibt es keinen Ausschlag (*derivative kick*), wenn die Zielgeschwindigkeit springt. Zusätzlich wird der D-Anteil tiefpassgefiltert, weil das Geschwindigkeitssignal verrauscht ist.
- **Anti-Windup:** Integriert wird nur, solange die Stellgröße nicht gesättigt ist; das Integral ist auf ±100 begrenzt.
- **Aufteilung:** Stellgröße > 0 → Gas, < 0 → Bremse. Beides wird getrennt an die Simulation übergeben. Die Bremse ist auf 0,8 begrenzt (darüber blockieren die Räder), Bremswerte < 0,05 werden ignoriert (rollen lassen statt bremsen).
- **Kopplung mit der Lenkung:** Gas und Bremse werden bei starkem Lenkeinschlag reduziert (Gas bis auf 30 %, Bremse bis auf 50 %). Gleichzeitig stark lenken und beschleunigen oder bremsen lässt die Hinterräder ausbrechen.


---

## 3. Ergebnisse und Vergleiche

### 3.1 Gesamtpipeline (`main.py`, 50 zufällige Seeds, 1000 Schritte)

| Messung | Ø Score | Std | Min | Max | Ø schlechteste 5 | Ø beste 5 |
|---|---|---|---|---|---|---|
| normal | **923,4** | 7,8 | 900,0 | 935,2 | 907,1 | 934,1 |
| `--domain_randomize` | **905,6** | 78,2 | 452,5 | 943,7 | 750,6 | 936,9 |

Ohne Farbrandomisierung ist die Pipeline sehr gleichmäßig: Alle 50 Episoden liegen zwischen 900 und 935. Mit `--domain_randomize` liegen 48 von 50 Episoden über 870, aber zwei Ausreißer (Seed 338417: 452,5 und Seed 440746: 618,0) ziehen den Mittelwert nach unten. Die Farben sind bei diesen beiden Strecken nicht die Ursache: Dieselben Strecken mit Standardfarben scheitern genauso (446 bzw. 607 Punkte), das Auto kommt dort in einer sehr engen Kurve von der Strecke ab. Die Fehlerfälle sind in Abschnitt 3.4 beschrieben.

### 3.2 Stanley vs. Pure Pursuit in der Gesamtpipeline

Gleiche 50 Strecken (Seeds 0–49) für beide Regler, alle anderen Module identisch:

| Querregler | Ø Score | Std | Min | Max |
|---|---|---|---|---|
| Stanley | 920,9 | 7,6 | 904,3 | 936,5 |
| Pure Pursuit | 920,0 | 7,9 | 902,7 | 936,1 |

Der Unterschied von 0,9 Punkten liegt deutlich unter der Streuung zwischen den Strecken; **in der Gesamtpipeline sind beide Regler gleichwertig**. Die Grenze setzt hier nicht die Querregelung, sondern die Zielgeschwindigkeit aus der Längsregelung.

### 3.3 Stanley vs. Pure Pursuit, isolierte Querregelung

`compare_lateral_control.py` nutzt den Ground-Truth-Pfad der Simulation und einen einfachen Regler, der eine feste Zielgeschwindigkeit hält. So wird nur die Querregelung verglichen. 10 Strecken (Seeds 0–9), Ø Score:

| Geschwindigkeit | Stanley | Pure Pursuit |
|---|---|---|
| 40 | **703,9** | 694,5 |
| 60 | 613,3 | **738,6** |
| 80 | 393,5 | **401,5** |

Die absoluten Werte sind niedriger als in der Gesamtpipeline, weil hier mit fester Geschwindigkeit und ohne Abbremsen vor Kurven gefahren wird. Bei 40 schafft das Auto die Runde nicht in 1000 Schritten, bei 80 kommt es in engen Kurven von der Strecke ab.

Reproduzieren: `uv run python compare_lateral_control.py --speeds 40 60 80 --seeds 10` (Standard ist `--seeds 5`)

**Bewertung des Vergleichs:**

- Bei niedriger Geschwindigkeit (40) sind beide Regler etwa gleich gut.
- Bei mittlerer Geschwindigkeit (60) ist **Pure Pursuit deutlich robuster** (+125 Punkte). Stanley reagiert stark auf den Querabstand zum *nächsten* Pfadpunkt; das Pixelrauschen des Pfads wirkt sich dadurch direkt auf den Lenkwinkel aus. Pure Pursuit zielt auf einen Punkt weiter vorne (geschwindigkeitsabhängiger Lookahead) und schneidet Kurven dadurch weicher an.
- Bei 80 sind beide Regler überfordert; ohne Abbremsen vor Kurven reicht keine Querregelung.

Deshalb ist **Pure Pursuit der Standard in `car.py`**: In der Gesamtpipeline sind beide gleich gut, in der isolierten Messung ist Pure Pursuit bei mittlerer Geschwindigkeit klar robuster und hat damit mehr Reserve.

### 3.4 Grenzen der Lösung

- **Feste Bildgeometrie:** Fahrzeugposition, Maske und Zuschnitt sind für das 96×96-Bild fest codiert.
- **Start-Phase:** Während der ersten 50 Schritte wird bewusst nicht gelenkt. Das funktioniert, weil CarRacing immer auf einem geraden Stück startet.
- **Spitzkehren und sich berührende Streckenteile:** Liegen zwei Streckenabschnitte direkt nebeneinander, kann die Zuordnung zum falschen Abschnitt springen. Der Pfad wird dann über den Fallback (letzter Plan, Abklingen der Lenkung) überbrückt.
- **Pfad = Mittellinie:** Es wird keine Ideallinie (Kurven schneiden) gefahren. Die Geschwindigkeit in Kurven ist deshalb durch die Kurvenkrümmung der Mitte begrenzt.
- **Ausreißer:** In der Messung mit `--domain_randomize` bricht der Score in 2 von 50 Episoden ein (452 und 618), ohne Randomisierung trat das in den 50 Episoden nicht auf. Dahinter stecken zwei verschiedene Fehlerfälle. Zur Unterscheidung wurde dieselbe Strecke (gleicher Seed) einmal mit zufälligen Farben und einmal mit den Standardfarben von CarRacing gefahren; dazu wurden die Farben im Env nach dem Reset umgesetzt, die Strecke bleibt dabei identisch.
  1. **Sehr enge Kurve bei hoher Geschwindigkeit** (Seed 338417 und 440746, die beiden Ausreißer aus 3.1). Die Spurerkennung arbeitet bis kurz vor dem Fehler korrekt, das Auto fährt aber mit v ≈ 75 in die Kurve, kommt von der Strecke ab und findet nicht zurück. Mit Standardfarben scheitert dieselbe Strecke an derselben Stelle (446 bzw. 607 Punkte), die Farbrandomisierung ist daran also nicht beteiligt, und der Fall kann auch ohne sie auftreten. Es gibt keine Wiederherstellungsstrategie: Steht das Auto im Gras, sieht die Spurerkennung keine zusammenhängende Straße mehr (teils werden die Kanten eines hellen Gras-Quadrats als Fahrbahnrand ausgegeben).
  2. **Straßenfarbe nahe der Hintergrundfarbe.** Dieser Fall führte in der Messung aus 3.1 zu keinem Ausreißer, wurde aber in einem weiteren Lauf mit `--domain_randomize` gefunden (50 Episoden, Ø 912,2; Seed 57554: Farbabstand zwischen Straße und Hintergrund ca. 20, 488 Punkte, mit Standardfarben auf derselben Strecke 915). Die Straßenkacheln haben drei leicht unterschiedliche Farbtöne (Abstand bis ca. 9). Wegen der ähnlichen Hintergrundfarbe schrumpft die Toleranz der Straßensegmentierung (siehe 2.1) auf ca. 7, und jede dritte Kachel fällt aus der Straßenmaske. Die Kanten zerfallen in Rechtecke, der Pfad bekommt Schleifen mit sehr hoher Krümmung, die Zielgeschwindigkeit wird immer wieder auf `v_min` gedrückt, und das Auto kriecht mit v ≈ 25–35 durch die Runde.
- **Parameter sind empirisch abgestimmt** (Pixel-Einheiten, nicht physikalisch kalibriert).
