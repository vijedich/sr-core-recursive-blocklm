# Testfeld (später): Adaptive rekursive Tiefe — wann bringt Iteration r>2 echten Gewinn?

*Notiz angelegt 2026-06-18. Zweck: festhalten, dass Tiefengewinn NICHT eingebildet war —
er war in mehreren Konfigurationen klar messbar. Auf TinyStories fällt das Modell aber in
eine zweistufige Eintritt+Kern-Struktur, in der weitere Iterationen kaum beitragen. Für einen
echten Nachweis adaptiver Tiefe braucht es später einen Datensatz, bei dem zusätzliche
Verarbeitungsschritte tatsächlich erforderlich sind.*

## Kernaussage

Es gab zwei unterschiedliche Betriebsarten:

**(A) Echte iterative Verfeinerung** — Zwischenzustand → weitere Verarbeitung → bessere Ausgabe.
Beobachtet bei härteren synthetischen Regimes, end-/linear-lastiger Loss-Gewichtung, harter
komplementärer Blockwahl.

**(B) Zweistufiger Pfad** — kategoriesensitiver Einstieg → residenter Verarbeitungskern → danach
kaum weitere Veränderung. Beobachtet bei den späteren TinyStories-Modellen.

Für TinyStories scheint das Modell gelernt zu haben, dass mehr Tiefe schlicht unnötig ist —
viele nächste Tokens sind mit relativ lokaler Sprachinformation vorhersagbar.

## Konfigurationen MIT echtem Tiefengewinn (gemessen)

**1. Synthetik, Loss-Gewichtungsvergleich (L1 − L_final):**
- gleichgewichtet: ≈ −0,003 (flach — Artefakt der Gleichgewichtung: jede Iteration wird gleich
  stark auf die korrekte Antwort trainiert → Iter 1 lernt schon möglichst alles)
- linear ansteigend: ≈ +0,037
- endlastig: ≈ +0,071

**2. Synthetik, REPEAT-Regime (stärkster Fall):**
- gleichgewichtet: Tiefengewinn ≈ 0,03
- linear / endlastig: ≈ 0,30–0,31
- (FIB unter endlastiger Gewichtung leicht positiv, aber insgesamt zu schwer / nicht sauber gelöst)

**3. Hard-Diversity-Smoke-Test — echte U-Kurve über Iterationen:**
- 6,267 → 6,144 → 6,097 → 6,084 → 6,090 → 6,109
- Verbesserung bis ~Iter 4 um ≈ 0,183 Nats, danach wieder schlechter.
- Hinweis auf: nützliche Rekursion, ABER optimale begrenzte Tiefe + Bedarf nach dynamischem Halten.

**4. Früher Phase-3-Lauf (harte Diversity) — Anytime-Abstand wächst im Training:**
- Schritt 1000 ≈ 0,003; 1500 ≈ 0,011; 2000 ≈ 0,014; 2500 ≈ 0,024; final ≈ 0,030
- Gleichzeitig Jaccard ↓, Diversity ↑ → komplementärere Blockwahl erzeugt Tiefengewinn.
- Preis hier: schlechtere Gesamtqualität + mehr Cache-Misses.

## Konfigurationen OHNE echten Tiefengewinn

**Phase 2 / TinyStories:** Per-Iteration-Kurve praktisch flach. Zustand änderte sich teils noch,
Output kaum besser. Forced Diversity verbesserte spätere Iterationen kurzfristig (Potenzial da),
das normale Routing nutzte es aber nicht.

**Curriculum A:** klar gelernt — r1 Eintritt/Preprocessing, r2 Verarbeitungskern, r3–r6 nahezu
redundant. Zustandsänderung nach dem Übergang ≈ 0,001. Keine fortlaufende Verfeinerung.

**Curriculum C (über mehrere Seeds stabil):**
- Per-Iteration-Loss vollständig flach
- Jaccard r1→r2 niedrig, r2→r6 ≈ 0,99
- Zustandsänderung nach r2 ≈ 0; Output-KL nach r2 ≈ 0
- Depth-Truncation: kein Qualitätsvorteil weiterer Iterationen
- Verbesserte Lastverteilung / Gini / Bank-Nutzung / Kategorien-Trennung, ABER nicht die
  nützliche Rechentiefe.
- State-Reset-Schaden ≈ +0,068 Nats zeigt: fortgeschriebener Zustand IST relevant — aber die
  tatsächlich benötigte Tiefe auf TinyStories ist ~1–2 Stufen, nicht 6.

## Konsequenz für die aktuelle Arbeit

Die TinyStories-Skalierungsmodelle testen vor allem **konstantes Working Set, Routing-Lokalität,
Blockbank-Skalierung** — NICHT mehr wirklich rekursive Tiefenverfeinerung.

Für den nächsten CPU-Benchmark sogar günstig: **R=2 dürfte fast dieselbe Qualität wie R=6
erreichen und deutlich weniger rechnen** (vgl. L1≈Lfin im H1-Sweep).

## Späteres Testfeld (TODO, wenn adaptive Tiefe nachgewiesen werden soll)

- Datensatz/Aufgabe, bei der zusätzliche Verarbeitungsschritte WIRKLICH nötig sind (mehrstufige
  Inferenz/Komposition), nicht lokal vorhersagbar.
- End-/linear-lastige Loss-Gewichtung (nicht gleichgewichtet) — sonst flacht die Kurve künstlich ab.
- Dynamisches Halten (adaptive R pro Token / Ponder-artig) — die U-Kurve (#3) legt eine optimale
  begrenzte Tiefe nahe, die token-abhängig variieren dürfte.
- Komplementäre Blockwahl als Hebel für Tiefengewinn (#4), aber Qualitäts-/Cache-Preis beachten.
