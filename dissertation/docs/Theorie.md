# Vorgehensplan: Rekursiv vertieftes, block-sparsames Sprachmodell mit prädiktivem Parameter-Streaming

---

# Teil I: Zielsetzung und Prämisse

## 1. Das eigentliche Problem

Große Sprachmodelle (30B, 70B, 100B+ Parameter) passen nicht in den VRAM handelsüblicher Hardware.
Der aktuelle Stand der Praxis: Layer-by-Layer-Offloading — das Modell liegt im RAM oder auf SSD,
jede Schicht wird bei Bedarf in den VRAM geladen, berechnet und wieder entladen.

Konkret auf einer typischen Consumer-GPU (6–8 GB VRAM, PCIe 16 GB/s):

```
7B-Modell in fp16 = ~14 GB Gewichte

Pro erzeugtem Token werden alle 14 GB durch die PCIe-Leitung bewegt:
14 GB ÷ 16 GB/s ≈ 0,9 Sekunden pro Token
→ ~1 Token/Sekunde
```

Das ist kein Algorithmus-Problem und kein Optimierungs-Problem. Es ist ein strukturelles Problem:
Ein dicht ausgeführtes Modell braucht bei jedem Token alle seine Gewichte — also werden sie alle geladen.

## 2. Die Kernhypothese

Ein Modell kann so trainiert werden, dass es bei der Verarbeitung jedes Tokens nur einen
**kleinen, vorhersagbaren Bruchteil seiner Gewichte** benötigt.

Wenn dieser Bruchteil:
- **klein** ist (z.B. 1–5 % der Gesamtgewichte pro Token),
- **vorhersagbar** ist (der nächste benötigte Block ist bekannt, bevor er gebraucht wird),
- **reproduzierbar** ist (ähnliche Eingaben aktivieren ähnliche Gewichte),

dann kann ein Modell, das größer ist als der verfügbare VRAM, auf kleiner Hardware laufen —
nicht weil es schneller ist, sondern weil es nie das gesamte Modell gleichzeitig braucht.

**Das Ziel ist Ermöglichung, nicht Optimierung.**

## 3. Abgrenzung zu bestehenden Ansätzen

| Ansatz | Funktionsprinzip | Vergleich |
|---|---|---|
| Layer-Offloading (llama.cpp) | Alle Schichten sequenziell laden | Baseline — 100 % der Gewichte pro Token |
| GPTQ / AWQ | Gewichte komprimieren | Weniger Bytes, aber immer noch alle Schichten |
| Standard-MoE (Mixtral) | Top-k von N Experten aktivieren | Ähnliche Idee, aber keine räumliche Struktur, kein Prefetching, kein 3D-Index |
| **Dieser Ansatz** | Sparse-Aktivierung + 3D-Topologie + prädiktives Streaming | Wenige Blöcke pro Token, Nächster Block vorhersagbar |

Standard-MoE löst das Qualitätsproblem (wenige aktive Parameter). Es löst nicht das
Prefetching-Problem: Der Router weiß, welche Experten aktiviert werden, aber nicht *wo* sie
im Speicher liegen oder welche als nächstes kommen werden. Die 3D-Struktur ist genau dafür da.

---

# Teil II: Kernkonzepte

## 4. Hochdimensionaler Zustandsraum

Der aktuelle sprachliche und semantische Zustand wird als hochdimensionaler Vektor dargestellt:

```
h_r ∈ ℝ^d
```

- `h_r`: Zustand nach Iteration `r`
- `d`: z.B. 256, 512, 1024 Dimensionen

Dieser Vektor repräsentiert nicht nur die aktuelle Token-Bedeutung, sondern auch den
Verarbeitungsfortschritt über Iterationen. Er ist das einzige, was zwischen Iterationen
weitergegeben wird.

## 5. Räumlicher Parameterraum — die 3D-Wissenstopologie

Die Parameterblöcke sind in einer **dreidimensionalen Topologie** organisiert:

```
c_b = (x_b, y_b, z_b)
```

Jeder Block besitzt:
- eine Block-ID,
- eine Position im 3D-Raum,
- einen semantischen Routing-Schlüssel `k_b`,
- einen Rechenoperator `F_b(h)`,
- (später) Verweise auf häufig folgende Blöcke.

**Warum dreidimensional?** Nicht aus physikalischen Gründen, sondern weil:
1. Der Raum lernbar strukturiert werden soll (Blöcke, die gemeinsam aktiviert werden, sollen
   sich annähern),
2. geografische Nähe als Prefetch-Heuristik genutzt werden kann,
3. der Router „Richtung" im Parameterraum entwickeln kann — nicht nur „Block X oder Y",
   sondern „in Richtung Region (3,4,5)".

## 6. Parameterblöcke

Das Modell besteht aus einer Bank von `B` Blöcken:

```
𝒲 = {W₁, W₂, …, W_B}
```

Jeder Block implementiert eine Transformation:

```
F_b(h) = W₂,b · σ(W₁,b · LN(h))
```

Blöcke können auch Attention-Operatoren, SSM-Operatoren oder hybride Strukturen sein.
**Attention-Blöcke sind für Streaming besonders geeignet**, weil sie quadratisch viele FLOPs
bei linear vielen Parametern erzeugen — höhere arithmetische Intensität bedeutet mehr
Berechnung pro geladenem Byte.

Blöcke müssen groß genug sein, um die Latenz des Ladevorgangs durch ausreichend Berechnung
zu amortisieren.

## 7. Router

Ein Router entscheidet in jeder Iteration, welche Blöcke ausgeführt werden.

**Basis-Router:**
```
R(h_r) → Blockauswahl 𝒮_r ⊆ 𝒲,  |𝒮_r| = k
```

**Erweiterter Router (Phase 3+):**
```
R(h_r, μ_{r-1}, v_{r-1}, u_{r-1}) → Blockauswahl, Confidence, Bewegungsrichtung
```

Dabei:
- `μ_{r-1}`: bisheriges Aktivierungszentrum im 3D-Raum
- `v_{r-1}`: Bewegungsrichtung in der letzten Iteration
- `u_{r-1}`: Unsicherheit (bestimmt Neighborhood-Radius)

Der Router erzeugt nicht nur eine Blockauswahl, sondern liefert auch einen **Prefetch-Hinweis**:
die vorhergesagte Zielregion für die nächste Iteration.

## 8. Neighborhoods und aktive Blockmengen

In Phase 1 bewertet der Router alle Blöcke. Ab Phase 3 wird die Kandidatenmenge durch die
räumliche Topologie eingeschränkt:

```
C_r = Neighborhood(μ_r)
```

Score für jeden Kandidaten:
```
s_{r,b} = sim(q_r, k_b) − β · d(μ_r, c_b) + γ · cache(b)
```

- `sim`: semantische Ähnlichkeit zwischen Query und Block-Key
- `β · d(μ_r, c_b)`: Abzug für räumliche Entfernung
- `γ · cache(b)`: Bonus für bereits geladene Blöcke

Die aktive Blockmenge:
```
𝒮_r ⊆ C_r,  |𝒮_r| = k
```

---

## 9. Tunnel — der Weg durch den Parameterraum

### Definition

Ein **Tunnel** ist die vollständige Sequenz von Blockaktivierungen, die ein Eingabe-Token
über alle Iterationen nimmt:

```
Tunnel_t = (𝒮₀, 𝒮₁, 𝒮₂, …, 𝒮_R)
```

Im 3D-Parameterraum beschreibt dieser Tunnel einen Pfad:
```
μ₀ → μ₁ → μ₂ → … → μ_R
```

wobei `μ_r` das geometrische Zentrum der in Iteration `r` aktiven Blöcke ist.

### Eigenschaften

**Eintrittsphase (r=0):** Der Tunnel beginnt breit. Der Zustandsvektor `h₀` zeigt grob auf
eine semantische Region, aber das Modell muss erst erkunden. Jaccard-Overlap zwischen
Eintritts- und Folgeiterationen ist niedrig (~0,13). Das ist die teuerste Phase — hier werden
die meisten neuen Blöcke geladen.

**Hauptkanal (r≥1):** Nach der ersten Iteration konvergiert der Tunnel auf eine eng definierte
Route. Jaccard-Overlap steigt auf 0,93–0,95. Die Verarbeitung bleibt in einer lokalen
Nachbarschaft des Parameterraums. Fast keine neuen Blöcke werden geladen.

**Form und Breite:** Verschiedene Eingabetypen erzeugen verschiedene Tunnel-Formen. Einfache,
vorhersehbare Tokens erzeugen enge, direkte Tunnel. Komplexe Tokens erzeugen breitere Tunnel
mit mehr Iterationen.

### Warum Tunnel das Streaming-Problem lösen

Bei Layer-Offloading bewegt sich die Verarbeitung zwangsläufig durch alle Schichten
(vertikal, Schicht 1 → Schicht N). Bei einem Tunnel-Modell bewegt sie sich durch einen
kleinen Bereich des 3D-Raums (lokal, Nachbarschaft um μ).

```
Layer-Offloading:  |geladene Bytes pro Token| = |gesamtes Modell|
Tunnel-Modell:     |geladene Bytes pro Token| = |k × block_size × (1 + ε)|
```

Mit `k=4` aktiven Blöcken und realistischer Blockgröße wird pro Token ein Bruchteil der
Gesamtgewichte benötigt, unabhängig von der Gesamtmodellgröße.

---

## 10. Leiterbahn — konsolidierte, indizierte Tunnel

### Analogie

Im menschlichen Nervensystem gibt es zwei Arten der Signalverarbeitung:

- **Kortikale Verarbeitung:** Flexible, bewusste Auswahl von Informationspfaden durch den
  gesamten Cortex. Langsam, aufwändig, variabel.
- **Subkortikale Pfade / Rückenmark-Reflexe:** Häufig wiederholte Signalwege werden als
  eigenständige Strukturen konsolidiert (Basalganglien, spinale Reflexbögen). Schnell,
  automatisch, ohne kortikale Beteiligung.

Oft genutzte Bewegungsabläufe (Gehen, Fahrradfahren) werden im Rückenmark und den
Basalganglien als eigene Schaltkreise gespeichert und können ohne vollständige kortikale
Entscheidungsfindung ausgeführt werden.

### Definition

Eine **Leiterbahn** ist ein konsolidierter Tunnel-Prototyp: ein häufig aktivierter
Tunnel-Typ, der als eigenständiger Index-Eintrag gespeichert wird.

```
Leiterbahn_L = {
    Eingabesignatur:    Clustering-Merkmal im h₀-Raum
    Blocksequenz:       (𝒮₀^L, 𝒮₁^L, …, 𝒮_R^L)
    Prefetch-Plan:      welche Blöcke wann geladen werden sollen
    Qualitätskurve:     erwartetes L(r) pro Iteration
    Haltepunkt:         typische Iterations-Tiefe für diesen Eingabetyp
    Varianz:            wie stabil die Route ist (niedrig = verlässlich nutzbar)
}
```

### Leiterbahn vs. Hub-Blöcke

Ein Hub-Block ist ein einzelner Block, der sehr häufig aktiviert wird und daher permanent
im VRAM gehalten wird. Eine Leiterbahn ist eine **Sequenz** — sie enthält nicht nur die
heißen Blöcke, sondern auch die Reihenfolge, die Eintritts-Signatur und den Prefetch-Plan.

```
Hub-Block:   Block #7 wird oft gebraucht → immer im VRAM behalten
Leiterbahn:  "Wenn h₀ in Region A liegt, dann folgt 0→7→3→7→12→7→3 → früh stoppen"
```

### Wie Leiterbahnen das Eintritts-Problem lösen

Die schwierigste Phase ist die Eintrittsiteration (Jaccard ~0,13 — völlig neu). Ohne
Vorwissen muss der Router erst suchen. Mit einem Leiterbahn-Index:

```
h₀ → Eingabesignatur berechnen
   → Ähnlichkeit zu bekannten Leiterbahn-Signaturen prüfen
   → Wenn Match (confidence > θ): gesamten Tunnel vorab laden
   → Sonst: normale Routing-Suche
```

Wenn ein Eingabe-Token einer bekannten Leiterbahn entspricht, entfällt die
Eintrittsphase komplett — der vollständige Tunnel ist bereits im VRAM.

### Leiterbahnen als separater Index

Der Leiterbahn-Index ist kein Modellbestandteil, sondern eine **Laufzeit-Datenstruktur**:

```
RAM:
├── Modellgewichte (alle Blöcke)
│   └── Block_0 … Block_B
└── Leiterbahn-Index
    ├── L₁: Signatur + Blocksequenz + Prefetch-Plan
    ├── L₂: Signatur + Blocksequenz + Prefetch-Plan
    └── …

VRAM:
├── Aktuell aktive Blöcke (k Stück)
├── Hub-Blöcke (permanent resident)
└── Prefetch-Buffer (nächste Iteration)
```

Leiterbahnen entstehen nicht durch explizites Training, sondern durch **Routing-Analyse**
nach dem Training: Häufige Tunnel werden identifiziert, geclustert und indiziert. Der Index
kann kontinuierlich aktualisiert werden (neue Eingabetypen erzeugen neue Leiterbahnen).

### Klassifizierung von Leiterbahnen

| Typ | Eigenschaft | Beispiel |
|---|---|---|
| **Stabile Leiterbahn** | Hohe Routingkonstanz, geringe Varianz | Syntaktische Muster, häufige Phrasen |
| **Domänen-Leiterbahn** | Aktiv nur in bestimmtem Themengebiet | Medizinische Begriffe, Code-Syntax |
| **Kontext-Leiterbahn** | Abhängig von Satz-/Gesprächskontext | Koreférenz-Auflösung |
| **Transitions-Leiterbahn** | Verbindet zwei Regionen | Themenübergang, Schlussfolgerung |

---

## 11. Rekursive Verarbeitung

Die aktiven Blöcke aktualisieren den Zustand:

```
h_{r+1} = h_r + Σ_{b ∈ 𝒮_r} α_{r,b} · F_b(h_r)
```

Nach jeder Iteration kann eine Tokenvorhersage erzeugt werden:

```
p_r(x_{t+1}) = softmax(W_out · h_r)
```

Das Modell ist damit ein **Anytime-Modell**: Es kann nach jeder Iteration stoppen.

Wichtige Hypothese:
```
L_{r+1} < L_r  für ausreichend schwierige Eingaben
```

Für triviale Eingaben (die nach Iteration 1 bereits gelöst sind) gilt diese Relation nicht —
das ist erwünscht, nicht problematisch.

## 12. Dynamische Tiefe

Das Modell soll nicht für jedes Token dieselbe Iterationstiefe verwenden.
Ein Haltemodul schätzt den erwarteten Qualitätsgewinn:

```
ΔQ̂_r = L̂_r − L̂_{r+1}
```

Stoppen, wenn der Gewinn die Kosten nicht übersteigt:
```
ΔQ̂_r < λ_C · C_{r+1} + λ_T · T_{r+1}
```

Im Kontext des Streaming-Ziels: `T_{r+1}` sind die Transfer-Kosten der nächsten Iteration.
Wenn kein neuer Block geladen werden muss (alle bereits im VRAM), sind die Transferkosten
nahezu null — das Modell darf tiefer rechnen, ohne Penalty.

## 13. Streaming-Architektur

```
SSD / Massenspeicher:
└── gesamte Parameterbasis (alle Blöcke)

RAM:
├── häufig benötigte Blöcke (LRU-Cache, hot tier)
└── Leiterbahn-Index

VRAM:
├── aktuell aktive Blöcke (k Stück × block_size)
├── Hub-Blöcke (permanent resident, ~15 % der Blöcke)
├── Prefetch-Buffer (nächste Iteration, asynchron geladen)
└── Zustandsvektoren + Compute-Buffer
```

Transfer pro Iteration:
```
Δ_r = 𝒮_{r+1} \ 𝒮_r     (nur neue Blöcke, nicht alle aktiven)
```

**Kritische Eigenschaft:** Die Menge neu zu ladender Blöcke ist nach Iteration 0 sehr klein
(Jaccard 0,93+). Der Großteil der Transferkosten fällt einmalig in der Eintrittsphase an.
Das Leiterbahn-System reduziert auch diese Kosten, indem es bekannte Eintrittsrouten
vorab lädt.

---

# Teil III: Wissenschaftliche Hypothesen

## 14. Primäre Hypothese — Ermöglichung statt Optimierung

Ein Sprachmodell kann so trainiert werden, dass bei der Verarbeitung jedes Tokens
nur ein kleiner, vorhersagbarer Bruchteil seiner Gewichte aktiviert wird — unabhängig von
der Gesamtmodellgröße. Damit können Modelle, deren Gesamtgröße den verfügbaren VRAM
übersteigt, auf kleiner Hardware betrieben werden, ohne vollständiges Layer-by-Layer-Offloading
zu benötigen.

**Maßstab:** Der Vergleich gilt gegen Layer-Offloading (Baseline: 100 % der Gewichte pro
Token), nicht gegen dichte Ausführung im VRAM.

## 15. Tunnel-Hypothese

Ähnliche Eingaben nehmen ähnliche Tunnel durch den Parameterraum. Innerhalb eines Tunnels
ist die Blockauswahl ab Iteration 1 hoch vorhersagbar (Jaccard > 0,9). Die Tunnel-Struktur
ist stabil genug, um Prefetching zu ermöglichen: Der nächste benötigte Block kann geladen
werden, während der aktuelle Block berechnet wird.

## 16. Leiterbahn-Hypothese

Häufig aktivierte Tunnel konsolidieren sich zu stabilen, indizierbaren Mustern. Ein
Leiterbahn-Index kann beim Erkennen einer bekannten Eingabesignatur den vollständigen
Tunnel vorab laden und die Eintrittslatenz eliminieren. Dies entspricht der Konsolidierung
von Prozedurgedächtnis im biologischen Nervensystem.

## 17. Anytime-Hypothese

Das Modell kann nach jeder Iteration eine auswertbare Vorhersage erzeugen. Für schwierige
Eingaben verbessert sich die Qualität mit jeder Iteration. Für einfache Eingaben (nach
Iteration 1 gelöst) wird das Modell durch das Halt-Modul früh gestoppt. Das dynamische
Anhalten ist ein direkter Transfer-Hebel: Weniger Iterationen bedeuten weniger Blöcke geladen.

## 18. Spezialisierungshypothese

Verschiedene Parameterblöcke lernen reproduzierbar verschiedene Funktionen. Ähnliche
Eingaben aktivieren ähnliche Blöcke. Domänenspezifische Eingaben aktivieren domänenlokale
Regionen im Parameterraum. Diese Spezialisierung ist die Voraussetzung dafür, dass die
3D-Topologie semantisch kohärent wird.

## 19. Topologie-Hypothese

Durch Training werden gemeinsam aktive Blöcke im 3D-Raum angenähert (Co-Activation →
Proximity). Geografische Nähe im Parameterraum korreliert mit Routing-Ähnlichkeit. Das
ermöglicht Neighborhood-basiertes Prefetching: Blöcke nahe der aktuellen Position sind
mit hoher Wahrscheinlichkeit als nächstes gefragt.

## 20. Breiten-Tiefen-Hypothese

Ein Modell mit geringer gleichzeitig aktiver Breite kann einen Teil der Qualität einer
breiteren Ausführung durch zusätzliche rekursive Tiefe kompensieren — bei gleichzeitig
geringerem Transfer-Aufwand, weil die Folge-Iterationen hauptsächlich bereits geladene
Blöcke nutzen.

---

# Teil IV: Vergleichsmodelle

## 21. Modell A — Dichte Referenz

Feste Tiefe, alle Blöcke einmal angewendet. Zeigt: Qualitätsgrenze bei gegebener
Parameterzahl und gegebenem Compute.

## 22. Modell B — Rekurrente Referenz

Ein geteilter Block, mehrfach angewendet. Zeigt: Was reine Rekursion ohne Routing leistet
(und nicht leistet). Referenz für den Nutzen dynamischer Auswahl.

## 23. Modell C — Dynamisch blockselektives Modell mit Tunnel-Verhalten

Bank aus `n_blocks` Blöcken, Top-`k` via Router pro Iteration, `R` Iterationen.

```
Embedding → Context-Encoder
          → Iteration 1: Router → k Blöcke → Update
          → Iteration 2: Router → k Blöcke → Update
          → …
          → Readout nach jeder Iteration
```

Dieses Modell erzeugt messbare Tunnel, zeigt Lokalität nach Iteration 1, und hat eine
Hub-Struktur. Es ist die Architektur, auf der der Leiterbahn-Index aufbaut.

---

# Teil V: Trainingsphasen

## 24. Phase 1 — Grundfunktion und Tunnel-Entstehung

**Ziele:**
- Completion auf synthetischen Daten lernen
- Routing stabilisieren (kein Kollaps)
- Tunnel-Verhalten messen: Entsteht zeitliche Lokalität?
- Anytime-Kurve mit end-lastiger Gewichtung prüfen

**Konfiguration:**
- Koordinaten fest (nicht im Routing genutzt)
- Weiches Top-k-Routing mit Load-Balance-Loss
- Deep Supervision: Readout nach jeder Iteration
- **End-lastige Loss-Gewichtung** (nicht equal — equal erzwingt flache Anytime-Kurve)
- Kein Halt-Modul, kein echtes Streaming

**Erfolgskriterium:**
- `L_final < L₁` für schwierige Regimes bei end-lastiger Gewichtung
- Jaccard-Overlap Iter 2→3 > 0,8
- Routing-Entropie > 0,9 (kein Kollaps)
- Reproduzierbare Spezialisierung (MI_norm > 0,1 über mehrere Seeds)

**Status: Abgeschlossen.** Alle Kriterien erfüllt auf synthetischen Daten.

## 25. Phase 2 — Härteres Routing

**Ziele:**
- Von weichem Softmax zu diskreter Blockauswahl
- Spezialisierung verstärken
- Tunnel-Formen schärfer werden lassen

**Maßnahmen:**
- Temperatur-Curriculum (schrittweise absenken)
- Gumbel-Softmax / Straight-Through-Top-k
- Nutzungsstatistiken je Block
- Exploration-Budget für selten aktivierte Blöcke

## 26. Phase 3 — Räumliche Topologie trainieren

**Ziele:**
- Gemeinsam aktive Blöcke sollen sich im 3D-Raum annähern
- Tunnel werden räumlich kohärent
- Geografische Nachbarschaft als Prefetch-Heuristik aktivieren

**Implementierung:**
- Blockkoordinaten werden trainierbar
- Schattenkoordinaten für stabilen Such-Index: `c̄_b ← ρc̄_b + (1−ρ)c_b`
- Distanzterm in Router-Score: `s_{r,b} = sim(q_r, k_b) − β · d(μ_r, c_b) + γ · cache(b)`
- Co-Activation-Loss: Blöcke, die gemeinsam aktiv sind, werden räumlich angenähert
- Repulsion-Loss: verhindert Kollaps aller Blöcke an einem Punkt

**Erfolgskriterium:**
- Häufig gemeinsam aktive Blöcke liegen signifikant näher als zufällig
- Geografische Nachbarschaft sagt Routing-Ähnlichkeit vorher

## 27. Phase 4 — Leiterbahn-Index aufbauen

**Kein Retraining** — reine Analyse der gelernten Tunnel.

**Vorgehen:**
- Routing-Traces auf großem Validierungsset aufzeichnen
- Tunnel-Sequenzen clustern (ähnliche Eintrittsregion + ähnliche Pfade)
- Stabile Cluster als Leiterbahnen indizieren
- Für jede Leiterbahn: Eingabesignatur, Blocksequenz, Prefetch-Plan, Varianz
- Hub-Blöcke identifizieren (Top-15 % nach Aktivierungsfrequenz)

**Erfolgskriterium:**
- Leiterbahn-Index deckt X % der Token ab (d.h. bekannte Signatur → bekannter Tunnel)
- Prefetch-Precision: Wie oft ist der nächste Block durch Leiterbahn bekannt?

## 28. Phase 5 — Hardware-Kosten simulieren und optimieren

**Ziele:**
- Reale Transfer-Kosten in den Trainings-Loss einbeziehen
- Tunnels mit hohem Transfer-Aufwand bestraft
- Leiterbahn-gestütztes Prefetching im Simulator testen

**I/O-Loss:**
```
L_IO = λ_B · Bytes_neu + λ_N · Transfers_neu + λ_S · Stallzeit
```

**Simulator-Konfiguration:**
- Virtueller Cache pro Sequenz
- Hub-Blöcke permanent resident
- Leiterbahn-Index: bekannte Tunnel werden vorab geladen
- Prefetch-Buffer: eine Iteration voraus
- Baseline-Vergleich: Layer-by-Layer-Offloading eines äquivalenten dichten Modells

**Erfolgskriterium:**
- Neu geladene Bytes pro Token < 5–10 % des äquivalenten Layer-Offloading-Aufwands

## 29. Phase 6 — Dynamisches Anhalten

**Kein Retraining des Backbones.** Zusätzlicher Halt-Kopf:

```
ΔQ̂_r = geschätzter Qualitätsgewinn durch Iteration r+1
Kosten(r+1) = Transfer-Kosten (neue Blöcke) + Rechenkosten
```

Stoppen, wenn `ΔQ̂_r < Kosten(r+1)`.

**Wichtig:** Wenn alle Blöcke der nächsten Iteration bereits im VRAM sind (bekannte
Leiterbahn), fallen keine Transfer-Kosten an — das Modell darf tiefer rechnen.

## 30. Phase 7 — Reales Streaming

Erst nach erfolgreicher Simulation (Phase 5 zeigt < 10 % des Layer-Offloading-Aufwands):
- Host-to-Device-Transfers asynchron
- Transfers und Berechnung überlappen (CUDA Streams)
- Leiterbahn-gestütztes Prefetching in Echtzeit
- Reale Tokens/Sekunde messen
- Vergleich gegen llama.cpp mit Layer-Offloading auf identischer Hardware

---

# Teil VI: Messgrößen

## 31. Transfer pro Token (primäre Streaming-Metrik)

```
D_bytes(t) = Σ_r Σ_{b ∈ 𝒮_{r+1} \ V_r} bytes(b)
```

**Vergleich gegen Baseline:**
```
Reduktion = 1 − D_bytes(t) / |gesamtes Modell in Bytes|
```

Bei 99 % Reduktion gegenüber Layer-Offloading: das Ziel ist erreicht.

## 32. Tunnel-Metriken

- **Jaccard-Overlap** zwischen aufeinanderfolgenden Iterationen: `J_r = |𝒮_r ∩ 𝒮_{r+1}| / |𝒮_r ∪ 𝒮_{r+1}|`
- **Tunnel-Breite**: Anzahl einzigartiger Blöcke über alle Iterationen eines Tokens
- **Tunnel-Divergenz**: Wie stark divergieren Tunnel bei ähnlichen Eingaben?
- **Eintrittskosten**: Blocks neu geladen in `r=0` vs. Folge-Iterationen

## 33. Leiterbahn-Metriken

- **Abdeckungsrate**: Anteil der Token, für die eine Leiterbahn bekannt ist
- **Prefetch-Precision**: Wie viele vorgeladenen Blöcke wurden tatsächlich genutzt?
- **Prefetch-Recall**: Wie viele benötigte Blöcke waren durch Leiterbahn vorab bekannt?
- **Leiterbahn-Stabilität**: Varianz der Tunnel innerhalb einer Leiterbahn-Klasse

## 34. Sprachqualität

- Cross-Entropy-Loss pro Iteration: `L(r)`
- Anytime-Kurve: `L(1), L(2), …, L(R)`
- Per-Regime-Analyse: `L_{q,r}` (nicht Aggregat — Aggregat verbirgt Tiefenstruktur)
- Korrelation optimale Stopptiefe mit Eingabeschwierigkeit: `corr(r*(q), schwierigkeit(q))`

## 35. Routing-Gesundheit

- Router-Entropie (Gleichverteilung der Blockaktivierungen)
- Maximaler Blockanteil (kein dominanter Block)
- Tote Blöcke (nie aktiviert)
- MI_norm zwischen Eingabetyp und Blockwahl

## 36. Arithmetische Intensität (Streaming-Physik)

```
AI_block = FLOPs(F_b) / bytes(W_b)    [FLOPs / Byte]
```

Mindest-AI für Streaming-Viabilität bei PCIe 16 GB/s:
```
AI_min ≈ GPU_peak_FLOPs / Bandbreite = ~290 FLOPs/Byte (RTX 2060)
```

MLP-Blöcke (d=128, h=256): AI ≈ 0,25 FLOPs/Byte — zu niedrig.
**Attention-Blöcke (seq=512, d=256)**: AI ≈ 128 FLOPs/Byte — deutlich besser.
int8-Quantisierung: verdoppelt AI (halbe Bytes, gleiche FLOPs).

---

# Teil VII: Datensätze

## 37. Stufe A — Synthetische Daten

Vier mechanistisch verschiedene Regimes (REPEAT, INCREMENT, FIB, ALTERNATE).
Ziel: Routing, Spezialisierung, Tunnel-Entstehung, Tiefennutzen validieren.
**Status: Abgeschlossen.**

Depth-Walk-Task: Bekannte wahre Tiefe `d`, modulare Permutation, kein Layer-Offloading-Vergleich
nötig — reine Mechanismus-Analyse.

## 38. Stufe B — TinyStories

Echte Sprache, kleine Domäne. Hier sollten erste Tunnel auf natürlichem Text sichtbar werden:
Tauchen grammatikalische Leiterbahnen auf? Haben häufige Phrasen stabile Tunnel?

## 39. Stufe C — Wikipedia-Ausschnitt

Wissensvielfalt. Testen: Entstehen domänenspezifische Tunnel-Regionen im 3D-Raum?
Kann der Leiterbahn-Index thematisch organisierte Blöcke identifizieren?

## 40. Stufe D — Instruction-Tuning

Erst nach funktionsfähigem Tunnel- und Leiterbahn-System.

---

# Teil VIII: Minimalprototyp

## 41. Konfiguration Phase 1 (Demo)

```
Zustandsdimension:      128 (Demo) / 256 (Spec)
Vokabular:              8.000–16.000 Tokens
Anzahl Blöcke:          24 (Demo) / 64–256 (Spec)
Aktive Blöcke:          k = 4
Iterationen:            R = 4–8
Blockoperator:          MLP (Demo), Attention anstreben (Spec)
Routing:                Noisy Top-k, weiches Softmax
Koordinaten:            fest (Phase 1), trainierbar ab Phase 3
Output-Head:            nach jeder Iteration (Deep Supervision)
Loss-Gewichtung:        end-lastig (0,7 auf letzte Iteration)
```

## 42. Konfiguration ab Phase 3

```
Blöcke:                 128–512
Aktive Blöcke:          k = 4–8
Routing:                Gumbel-Softmax → ST-Top-k
Koordinaten:            trainierbar mit Schattenkoordinaten
Cache:                  simuliert mit Leiterbahn-Index
Quantisierung:          int8 ab Phase 5
```

---

# Teil IX: To-Do-Liste

## 43. Abgeschlossen

- [x] Projektstruktur, Konfiguration, Checkpointing, Seeds
- [x] Synthetische Datenpipeline (4 Regimes + Depth-Walk-Task)
- [x] Modelle A, B, C implementiert und verglichen
- [x] Routing-Metriken: Entropie, Jaccard, tote Blöcke, Ablation, MI_norm
- [x] Deep Supervision mit konfigurierbarer Gewichtung
- [x] Anytime-Kurve und Loss-Varianten (Exp1a)
- [x] Virtueller Cache-Simulator, Hub-Analyse, Break-Even-Berechnung (Exp2)
- [x] In-Loop-Attention (`recurrent_read`) für Tiefen-Experiment
- [x] Depth-Walk-Harness mit Kontrollen (oracle, state_reset, shuffle_route)
- [x] Tunnel-Verhalten gemessen (Jaccard, Eintrittsphase, Hub-Struktur)
- [x] Leiterbahn-Konzept definiert (dieser Plan)

## 44. Nächste Schritte — Phase 2

- [ ] Temperatur-Parameter einführen
- [ ] Gumbel-Softmax implementieren
- [ ] ST-Top-k implementieren
- [ ] Nutzungsstatistiken je Block live tracken
- [ ] Spezialisierung nach Phase-2-Routing messen (MI_norm sollte steigen)

## 45. Nächste Schritte — Phase 3

- [ ] Koordinaten trainierbar machen
- [ ] Schattenkoordinaten-Update implementieren
- [ ] Co-Activation-Anziehung + Repulsion-Loss
- [ ] Distanzterm `β·d(μ_r, c_b)` im Router aktivieren
- [ ] 3D-Clustervisualisierung
- [ ] Neighborhood-Kandidaten-Einschränkung aktivieren

## 46. Nächste Schritte — Phase 4 (Leiterbahn)

- [ ] Routing-Trace-Analyse auf großem Validierungsset
- [ ] Tunnel-Clustering-Algorithmus (ähnliche Eintritt+Pfad → gleiche Leiterbahn)
- [ ] Leiterbahn-Index-Struktur implementieren
- [ ] Abdeckungsrate messen
- [ ] Prefetch-Simulation mit Leiterbahn-Index

## 47. Nächste Schritte — Phase 5

- [ ] I/O-Loss implementieren
- [ ] Simulator: Layer-Offloading als Baseline einbauen
- [ ] Simulator: Leiterbahn-gestütztes Prefetching testen
- [ ] int8-Quantisierung der Blockgewichte testen
- [ ] Zielmetrik: < 10 % Transfer vs. Layer-Offloading

## 48. Nächste Schritte — Phase 6 + 7

- [ ] Halt-Modul implementieren
- [ ] Dynamische Tiefe testen (einfache Tokens stoppen früher)
- [ ] CUDA Streams für asynchronen Transfer
- [ ] Reale Tokens/Sekunde gegen llama.cpp-Baseline messen
- [ ] Leiterbahn-Prefetch in Echtzeit

## 49. Offen / nicht begonnen

- [ ] TinyStories-Lauf (Netzwerk für Daten nötig)
- [ ] Attention-Blöcke als Alternative zu MLP-Blöcken (höhere arith. Intensität)
- [ ] Korrigierbarkeit: falsche Eintrittsregion in Iter 1 — verlässt das Modell sie?
- [ ] Curriculum für Depth-Walk (D_max=2 → 4 → 6 ohne Ablenker)
- [ ] Width-Depth-Grid konvergiert auf GPU

---

# Teil X: Go/No-Go-Kriterien

## 50. Go-Kriterium 1 — Tunnel-Lokalität

**Weiterarbeiten,** wenn aufeinanderfolgende Iterationen einen Block-Overlap > 0,8 zeigen
(nach Iteration 1) und die Transfermenge pro Token messbar unter Layer-Offloading liegt.

**Stopp,** wenn jede Iteration weitgehend neue Blöcke benötigt und der Transfer-Vorteil
gegenüber Layer-Offloading nicht nachweisbar ist.

**Status: GO.** Jaccard 0,93–0,95, Hub-Struktur nachgewiesen.

## 51. Go-Kriterium 2 — Block-Spezialisierung

**Weiterarbeiten,** wenn verschiedene Blöcke reproduzierbar verschiedene Funktionen lernen
(messbar durch MI_norm und Ablation).

**Status: GO.** MI_norm = 0,197 ± 0,030, Ablations-Effekte signifikant.

## 52. Go-Kriterium 3 — Breitenbegrenzung

**Weiterarbeiten,** wenn ein Modell mit k aktiven Blöcken eine Vorhersagequalität erreicht,
die deutlich besser als Zufall ist und mit mehr Iterationen verbessert werden kann.

**Status: GO.** C (4/24) schlägt A (16/16) bei gleichem Compute.

## 53. Go-Kriterium 4 — Transfer-Reduktion vs. Layer-Offloading

**Weiterarbeiten,** wenn das sparse Routing nachweislich weniger als X % der Gesamtgewichte
pro Token lädt, verglichen mit Layer-Offloading (= 100 %).

**(Dieser Vergleich ersetzt die frühere fehlerhafte Break-Even-Berechnung.)**

Bei aktueller Demo (24 Blöcke): ~29 % der Blöcke pro Token.
Bei skalierter Bank (256 Blöcke): ~4 % (10 einzigartige Blöcke von 256).

**Status: Bedingtes GO.** Demo-Maßstab zeigt 3,4× Reduktion. Skalierter Maßstab
projiziert 25–100×. Vollständiger Nachweis auf TinyStories mit realistischer Blockgröße
steht aus.

## 54. Go-Kriterium 5 — Leiterbahn-Index funktioniert

**Weiterarbeiten** mit Leiterbahn-Optimierungen, wenn der Index einen signifikanten Anteil
der Token mit bekannten Tunneln abdeckt (>20 %) und die Prefetch-Precision > 0,7 ist.

**Status: Offen.** Index noch nicht aufgebaut — dieser Schritt kommt nach Phase 3.

## 55. Go-Kriterium 6 — Reale Beschleunigung

Das Gesamtsystem gilt als validiert, wenn auf realer Hardware (GPU + PCIe) die Tokens/Sekunde
mit dem Sparse-Streaming-Modell nachweislich höher liegen als mit Layer-Offloading eines
äquivalent großen dichten Modells — oder wenn bei gleicher Geschwindigkeit ein deutlich
größeres Modell betrieben werden kann.

**Status: Offen.** Phase 7.

---

# Teil XI: Zulässige Aussagen nach aktuellem Stand

Nach Phase 1 + Exp1a + Exp2:

> Ein rekurrentes, blockselektives Modell kann mit 4 von 24 aktiven Blöcken eine
> Vorhersagequalität erreichen, die ein dicht ausgeführtes Modell gleichen Compute-Budgets
> übertrifft. Das Routing zeigt zeitliche Lokalität: nach der Eintrittsiteration werden
> 93–95 % der Blöcke wiederverwendet. Eine Hub-Struktur (wenige Blöcke tragen den Großteil
> der Aktivierungen) ist nachweisbar und durch permanentes Resident-Halten nutzbar. Diese
> Eigenschaften sind die Voraussetzungen für prädiktives Parameter-Streaming.

> Die Anytime-Kurve ist durch die Loss-Gewichtung steuerbar. End-lastige Gewichtung erhöht
> den Tiefengewinn für schwierige Muster um Faktor 10 gegenüber gleichmäßiger Gewichtung.
> Dieser Effekt muss bei jedem tief-supervidierten Modell berücksichtigt werden.

> Rekursive Tiefe leistet nützliche sequenzielle Berechnung in isolierter Messung
> (ablenkfreier Depth-Walk): 2-Schritt-Komposition benötigt 3 Iterationen, State-Reset
> zerstört den Gewinn nachweisbar. Das Eintritts-Problem (Jaccard ~0,13 in Iteration 0)
> ist die wichtigste offene Ingenieur-Frage für das Streaming-System.

Nach vollständiger Validierung (Phasen 3–7) wäre diese Aussage das Ziel:

> Ein Sprachmodell mit einer Gesamtparameterzahl, die den verfügbaren VRAM um Faktor N
> übersteigt, kann durch rekursive blockselektive Ausführung und Leiterbahn-gestütztes
> Prefetching effizienter betrieben werden als durch vollständiges Layer-by-Layer-Offloading
> — weil pro Token weniger als 5 % der Gesamtgewichte transferiert werden müssen.
