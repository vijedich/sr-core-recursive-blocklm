# Experiment 2 — Tunnel-Charakterisierung und Transfer-Analyse

Tunnel-Charakterisierung auf den realen Routing-Traces des trainierten C-Modells (Seed 0).
**Kein Retraining.** Physikmodell mit voller Blockgröße (d=256, h=512 → fp16 ≈ 527 KB/Block).

> **Ehrliche Reichweite:** Die Demo-Bank (24 Blöcke ≈ 12 MB) passt in den VRAM —
> reales Streaming ist hier physisch nicht nötig. Der Wert dieses Experiments liegt in der
> **Tunnel-Charakterisierung**: Wie lokal sind die Tunnel? Wo liegen die Transferkosten?
> Ist eine Hub-Struktur vorhanden? Diese Eigenschaften skalieren auf große Bänke.
>
> **Korrekte Vergleichs-Baseline:** Der Transfer-Aufwand wird gegen **Layer-by-Layer-
> Offloading** eines äquivalent großen dichten Modells gemessen, nicht gegen null.
> Ein 24-Block-Modell mit vollständigem Offloading lädt alle 24 Blöcke pro Token.
> Das sparse Modell lädt ~7 einzigartige Blöcke — eine 3,4× Reduktion auf Demo-Skala,
> die bei 256 Blöcken auf ~26× und bei 1000 Blöcken auf ~100× skaliert.

## Befund 1 — Die Lokalität ist echt (entscheidender Test)

Gelerntes Routing gegen eine **Random-Routing-Kontrolle** mit identischer Last,
Miss-Rate unter Cache-Druck (Single-Stream):

| Cache-Kapazität (von 24) | gelernt | random |
|---|---|---|
| 8 | **0,174** | 0,716 |
| 12 | **0,082** | 0,540 |
| 16 | **0,033** | 0,367 |

Bei halber Bank (cap=12) verursacht gelerntes Routing **6,5×** weniger Misses als
Zufall. Die Aktivierungen sind also nicht nur intra-Token kohärent (Jaccard),
sondern über den **Token-Strom** hinweg konzentriert. Das ist die Voraussetzung,
die Streaming überhaupt erst plausibel macht — und sie ist erfüllt.

Reuse-Distanz (gelernt): **p50 = 1**, p90 = 8 Mikroschritte. Ein angefragter
Block wird meist im unmittelbar nächsten Mikroschritt erneut gebraucht → kleiner
Cache genügt für die Inner-Loop-Wiederverwendung.

## Befund 2 — Eintrittsphase dominiert die Transferkosten

**66 %** aller Blockladungen fallen in der Eintritts-Iteration `r=0` an, nur 34 %
in den Folge-Iterationen. Das bestätigt das Zwei-Phasen-Tunnelbild quantitativ
in Ladevorgängen (nicht nur in Overlap): beweglicher Einstieg, dann lokale
Verfeinerung. Prefetching muss vor allem den Übergang `r=0 → r=1` treffen.

## Befund 3 — Hub-Struktur ist real und pinbar

Blockfrequenz ist steil: Top-3-Blöcke ~24–26k Nutzungen, danach Abfall auf ~6k.
Pinnen der heißesten Blöcke (gebatchter Stresstest, cap=10) senkt die Miss-Rate
gelernt von 0,845 (H=0) auf 0,521 (H=8); bei Random bleibt sie bei 0,667. Ein
statischer Resident-Satz der wenigen Hub-Blöcke ist also wirksam — aber nur, weil
die Hub-Struktur existiert.

## Befund 4 — Arithmetische Intensität als Engineeringproblem

Bei dieser Block-Größe (MLP, d=128/256) ist die arithmetische Intensität niedrig
(~0,25 FLOPs/Byte), was bedeutet dass bei Messung gegen *keine Berechnung* der Transfer
dominiert. **Das ist die falsche Baseline für das eigentliche Projektziel.**

Gegen die richtige Baseline (Layer-Offloading = alle Blöcke pro Token):
- Demo (24 Blöcke): ~7 einzigartige geladen statt 24 → **3,4× weniger Transfer**
- 256 Blöcke: ~10 einzigartige geladen statt 256 → **26× weniger Transfer**
- 1000 Blöcke: ~10 einzigartige geladen statt 1000 → **100× weniger Transfer**

Dieser Vorteil wächst mit der Modellgröße. Das ist der eigentliche Wert des Tunnelmechanismus.

Die folgende Tabelle zeigt die absolute Compute-vs-Transfer-Relation (für Prefetch-Overlap-
Planung, nicht als Viabilität-Urteil):

| Bandbreite | Transfer (gebatcht, 48) | Compute | Status |
|---|---|---|---|
| 8 GB/s | 320 ms | 1,55 ms | transfer-bound |
| 16 GB/s | 172 ms | 1,55 ms | transfer-bound |
| 32 GB/s | 97 ms | 1,55 ms | transfer-bound |

Batching über 48 gleichzeitige Ströme amortisiert (695 → 259 KB/Token), aber bei
weitem nicht genug: Transfer übersteigt Compute um zwei Größenordnungen.

**Damit ist die Frage aus deiner Analyse beantwortet:** Der hohe Overlap erzeugt
sehr wohl einen *relativen* Transfervorteil (6,5× weniger Misses als Zufall) —
aber der *absolute* Transfer dominiert trotzdem. Jaccall/Overlap allein ist keine
hinreichende Streaming-Metrik; die **arithmetische Intensität** entscheidet.

### Break-Even: Wiederverwendungen pro Block-Ladung

Damit Transfer hinter Compute versteckt werden kann, muss jeder geladene Block
mindestens so oft genutzt werden:

| dtype | 8 GB/s | 16 GB/s | 32 GB/s |
|---|---|---|---|
| fp16 | 6 281 | **3 140** | 1 570 |
| int8 | 3 140 | 1 570 | **785** |

Lies das als harte Design-Schranke: Ein MLP-Block dieser Größe muss **~3 000-mal
(fp16, 16 GB/s) bzw. ~800-mal (int8, 32 GB/s) wiederverwendet werden, bevor er
evictet wird**, sonst lohnt Streaming nicht. Hebel dafür:

- **Residenz-Stickiness / Hot-Block-Pinning** (Befund 3) — heiße Blöcke gar nicht erst evicten.
- **Hohe Nebenläufigkeit** (großer Decode-Batch / geteilter Kontext) — ein residenter Block bedient viele Tokens gleichzeitig.
- **Quantisierung** (int8/int4) — halbiert/viertelt die Bytes und damit den Break-Even.
- **Größere arithmetische Intensität pro Block** (z. B. Attention-/SSM-Operatoren statt schmaler MLPs) — mehr FLOPs je geladenem Byte.
- **Seltener wechseln**: dynamische Tiefe so steuern, dass Folge-Iterationen in der residenten Neighborhood bleiben.

## Zulässige Kernaussage (Experiment 2)

> Das gelernte Routing erzeugt echte Tunnel: zeitlich konzentrierte Aktivierungspfade
> durch den Parameterraum, mit 6–7× weniger Cache-Misses als Zufallsrouting, sehr
> kurzen Reuse-Distanzen, einer dominanten Eintrittsphase und einer pinbaren Hub-Struktur.
> Gegen die richtige Baseline (Layer-Offloading = alle Blöcke) lädt das Modell auf
> Demo-Skala 3,4× weniger Blöcke pro Token; dieser Faktor wächst auf ~100× bei
> 1000-Block-Bänken, weil k (aktive Blöcke) mit der Bankgröße nicht skaliert.
> Die Eintrittsphase (r=0, Jaccard ~0,13) ist das verbleibende Hauptproblem —
> genau hierfür ist der Leiterbahn-Index vorgesehen.

## Was als Nächstes

1. **Phase 3** (räumliche Topologie): Koordinaten trainierbar machen, Co-Activation-Loss,
   Distanzterm im Router aktivieren. Erst dann ist der Parameterraum kohärent genug,
   um den Leiterbahn-Index sinnvoll aufzubauen.
2. **Leiterbahn-Index** (Phase 4): Routing-Traces auf Validierungsset aufzeichnen,
   häufige Tunnel clustern, Prefetch-Plan für bekannte Tunnel erzeugen. Das adressiert
   direkt das Eintritts-Problem.
3. **Simulator erweitern** (Phase 5): Baseline = Layer-Offloading, Leiterbahn-gestütztes
   Prefetching einschalten, Zielmetrik = < 5 % Transfer vs. Baseline.
4. Erst nach Simulator-GO: reales CUDA-Streaming (Phase 7).
