# td3-sa-pilot: łańcuch SA z checkpointów TD3 w fazach plateau

- **Status:** prerejestracja 2026-11-16 (7bdf3b5), wyniki 2026-11-16.
- **Charakter:** offline, bez treningu i bez nowych checkpointów, lokalnie na CPU (bez grantu).
- **Code:** branch `td3-sa-pilot` (z `local-search-v1`).
- **Spec:** `experiment.toml` (`[chain]`, `[evaluation]`, `[compare]`, `[rule]`). Checkpointy, σ i krzywe TD3: `critic-degeneracy-v1`. Epizody do T0: `local-search-v1`.
- **Bez decyzji o metodzie:** wynik wraca do uzgodnienia.

## Pytanie

- Czy w fazach, w których TD3 stoi, łańcuch SA na aktorze TD3 (k = 3 parowane) zyskuje przez 100k kroków więcej niż ten sam run TD3 przez kolejne 100k kroków?

## Kontekst: zgodność odtworzeń local-search-v1

- Rozbieżności (`validation_max_abs_diff` do 9614) w 8/18 checkpointów: v1 liczyło walidację w batchu 11-15 głów, local-search-v1 w batchu 81. XLA na CPU liczy matmul z < 16 wierszami inną ścieżką (|Δa| ≈ 1e-7 dla n < 16, bitowa równość dla n ≥ 16, sprawdzone n = 8..40 w 3 envach). Chaotyczna dynamika rozwija to w każdy epizod.
- Powtórka tych 8 checkpointów w oryginalnym batchu: bitowo zgodna, więc zaburzenia są te same.
- `reproduction_checked = false` w 12/18 = brak lokalnego `buffer.npz` (pobrany tylko dla 300k). Powtórka zastępuje to sprawdzenie.
- Prawda w Walker2d/Swimmer: komórki z tego pilota bez rozbieżności poza Swimmer s0 600k (zysk zaburzeń przesunięty maks. o 1.39 = 2.4% zwrotu aktora) i HalfCheetah 600k. Prawda local-search-v1 liczona w jednym batchu dla wszystkich głów, więc spójna.
- Wniosek dla pilota: zwrot epizodu przy stałym seedzie resetu zależy od ścieżki numerycznej. Ocena końcowa liczy aktora startowego i wszystkie snapshoty w jednym batchu (61 głów) na tych samych seedach.

## Komórki

- Swimmer 300k, 600k, 1M, Walker2d 600k, seedy 0-1. HalfCheetah 600k, seedy 0-1, jako kontrola braku szkody. Razem 10 komórek.
- Zwrot aktora startowego (30 epizodów local-search-v1): Swimmer 53 / 48 (300k), 57 / 127 (600k), 64 / 133 (1M), Walker2d 600k 944 / 5060, HalfCheetah 600k 10010 / 8360 (s0 / s1).

## Łańcuch

- Start: aktor TD3 z checkpointu (embedding stały, łańcuch zmienia tylko głowę).
- Iteracja: kandydat θ' = θ + σ·std(W)·N(0, 1) per liść bieżącej głowy, σ = skalibrowany poziom komórki (5% lub 20% spadku zwrotu).
- Ocena: 3 świeże seedy resetu, wspólne dla θ i θ' (parowanie), Δ̂ = średnia R(θ') − R(θ). Bieżąca polityka oceniana od nowa w każdej iteracji.
- Akceptacja: Δ̂ > 0, inaczej z prawd. exp(Δ̂ / T).
  - `anneal`: T = T0·(1 − użyte / 100k), liniowo do 0.
  - `greedy`: T = 0.
  - T0 per komórka i σ = mediana po 40 zaburzeniach sd(różnicy parowanej na 30 epizodach) / √3 z local-search-v1, czyli szum Δ̂.
- Koszt = rzeczywiste kroki 6 epizodów. Iteracja startuje, gdy użyte < 100k.
- Warianty: 2 σ × 2 harmonogramy = 4, po 5 niezależnych łańcuchów (40 łańcuchów na env × seed × faza, 200 razem). Seedy resetu łańcucha c: 3 000 000 + 100 000·c + 3i + j, te same dla wszystkich wariantów.
- Snapshoty polityki przy 25k, 50k, 100k użytych kroków. Liczba akceptacji per łańcuch.

## Ocena i porównanie

- Ocena końcowa poza budżetem: aktor startowy + 60 snapshotów na 10 świeżych seedach (4 000 000-4 000 009), jeden batch.
- Zysk SA (komórka, wariant, snapshot) = średnia po 5 łańcuchach z średniej różnicy R(snapshot) − R(start) na 10 epizodach.
- Zysk TD3 (komórka) = średnia `perf/eval_rl` w [t + 90k, t + 110k] − średnia w [t − 10k, t + 10k] (5 punktów × 5 epizodów), ten sam run. Dla 1M brak: tylko zysk SA vs aktor startowy.

## Reguła (zamknięta)

- **SA bije TD3 w plateau**, jeśli istnieje wariant, który przy snapshocie 100k ma zysk SA > 0 i > zysku TD3:
  - w ≥ 3 z 4 par Swimmer (300k, 600k × seedy 0, 1),
  - i w Walker2d 600k na seedzie, który utknął (s0: aktor upada po ok. 297 krokach, zwrot 944 vs 5060 na s1),
  - i ma |zysk SA| < 2% zwrotu aktora startowego w obu komórkach HalfCheetah 600k.
- Doprecyzowanie „> 0” (jak w local-search-v1): przy ujemnym zysku TD3 łańcuch, który nic nie zyskał, nie przechodzi.
- Poza regułą, raportowane: Walker2d 600k s1, Swimmer 1M, snapshoty 25k i 50k (czy łańcuch traci impet), akceptacje, rozrzut między łańcuchami.
- Wielokrotność: 4 warianty, 5 łańcuchów na wariant, 10 epizodów oceny. Pilot: wynik to wskazanie, nie dowód.

## Raport

- Zysk po 25k, 50k, 100k per komórka × wariant (średnia i rozrzut po łańcuchach) vs zysk TD3, liczba akceptacji, ocena końcowa na 10 świeżych epizodach, werdykt reguły.
- `python -m analysis.sa_pilot ../experiments/td3-sa-pilot/experiment.toml` (z `src/`) → `sa_pilot.{json,html}`.

## Wyniki (2026-11-16)

- **Raport:** https://claude.ai/code/artifact/d014ff24-b872-465d-a175-37cb78b31b00 (`sa_pilot.{json,html}`).
- **Werdykt: reguła niespełniona.** Żaden z 4 wariantów nie bije TD3 w plateau.

| wariant | Swimmer (pary) | Walker2d 600k s0: SA vs TD3 | HalfCheetah 600k s0 / s1 |
|---|---|---|---|
| σ 5%, T0 → 0 | 0/4 | +10.7 vs +57.6 | −1.42% / −0.82% |
| σ 5%, T = 0 | 2/4 | −120.2 vs +57.6 | +0.24% / 0.00% |
| σ 20%, T0 → 0 | 1/4 | −306.0 vs +57.6 | 0.00% / 0.00% |
| σ 20%, T = 0 | 1/4 | −136.1 vs +57.6 | 0.00% / 0.00% |

- Swimmer: TD3 też stoi (+3.1, +2.8, +0.9, −2.5 przez 100k). Zysk SA po 100k od −20 do +11.7, rozrzut między łańcuchami większy niż średnia (np. σ 20% s0 300k: −30 … +44). Nie da się odróżnić od szumu przy 5 łańcuchach i 10 epizodach.
- Walker2d 600k s0 (utknięty): impet traci się po 50k. σ 5% przy 50k: wszystkie 10 łańcuchów > 0 (średnio +169 / +178), przy 100k 2/5 łańcuchów w każdym harmonogramie spada do ok. −600. σ 20% szkodzi już od 25k.
- Poza regułą: Walker2d 600k s1 σ 5% +310 / +285, wszystkie 10 łańcuchów > 0 (+117 … +565), TD3 −28. σ 20% tam −809 … −1519. Swimmer 1M σ 20% +21 / +17 (s0), σ 5% +8 / +11 (s1).
- HalfCheetah: brak szkody, bo prawie nic nie jest akceptowane (σ 20%: 0 akceptacji, σ 5%: ≤ 1.4 średnio). Kontrola przechodzi trywialnie.
- Akceptacje: σ 5% ok. 25%, σ 20% ok. 18% kandydatów. Swimmer i HalfCheetah 17 iteracji na 100k (6 × 1000 kroków), Walker2d s0 44-58 (upadki skracają epizody).
- Wniosek wstępny: k = 3 przy σ z kalibracji to za mało, by łańcuch szedł w górę dłużej niż ok. 50k. Akceptacje na szumie (winner's curse) zjadają wcześniejszy zysk. Jedyny czysty sygnał (Walker2d s1 σ 5%) jest poza regułą i wymaga potwierdzenia.

### Odstępstwa

- `[analysis] chains_cache` i `[report] url` dopisane do toml po prerejestracji (ścieżki, bez zmian w `[chain]`, `[compare]`, `[rule]`).
- Pierwsze uruchomienie łańcuchów padło przy zapisie JSON (`np.bool_` w trace), poprawka `bool(...)`, ponowne uruchomienie. Łańcuchy są deterministyczne (te same akceptacje w logach obu przebiegów).
- Snapshoty biorą pierwszą iterację z użytymi ≥ 25k / 50k / 100k (przekroczenie < 1 iteracji, w Swimmer/HalfCheetah do 2k kroków). Akceptacje liczone do tej iteracji.
