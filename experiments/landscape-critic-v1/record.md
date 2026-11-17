# landscape-critic-v1: czy krajobraz zaburzeń uczy krytyka kierunku poprawy, offline

- **Status:** prerejestracja 2026-11-17, przed rolloutami krajobrazu, dotrenowaniem krytyka i krokami.
- **Charakter:** offline, lokalnie na CPU (bez grantu). Bufory 600k/1M skopiowane z PD (`critic-degeneracy-v1/ckpt`, tylko odczyt).
- **Code:** branch `critic-curvature-es-v1`.
- **Spec:** `experiment.toml` (`[landscape]`, `[critic]`, `[direction]`, `[steps]`, `[evaluation]`, `[[hypotheses]]`). Checkpointy, σ i klucz zaburzeń: `critic-degeneracy-v1`. R_i: `local-search-v1`. Stany, dist_H, losowania k: `critic-curvature-es-v1`.
- **Bez decyzji o metodzie:** wynik wraca do uzgodnienia.

## Pytanie

- Czy krytyk TD3 dotrenowany na przejściach z krajobrazu zaburzeń głowy aktora uczy się kierunku poprawy (gradient ES oracle), i czy krok po jego gradiencie daje więcej niż to samo dotrenowanie na samym buforze?

## Dane

- Komórka = env × seed × checkpoint (300k, 600k, 1M): 18 komórek, 6 w Walker2d (hipotezy), 6 HalfCheetah (kontrola), 6 Swimmer (opisowo).
- Krajobraz: 80 zaburzeń critic-degeneracy-v1 (40 na poziom σ: małe = spadek 5%, duże = 20%), 1 epizod każde, seed 7 000 000 + i (to samo i w obu poziomach). Zapisane wszystkie żywe przejścia (s, a, r, s′, terminated), jak w buforze TD3. ≤ 80k przejść na komórkę.
- Głowa aktora P = 1542 (Walker2d, HalfCheetah), 514 (Swimmer).

## Krytyk

- Start z krytyków checkpointu, target = kopia online, target aktor i embedding zamrożone (a′ = aktor(s′) + obcięty szum, jak TD3). Aktualizacja krytyka TD3, świeży Adam lr 1e-3, batch 256, γ 0.99, τ 0.005.
- Warianty: A = bufor + krajobraz σ małe, B = bufor + σ duże, C = bufor + oba, D = sam bufor (kontrola liczby aktualizacji).
- Batch: połowa z bufora, połowa z przejść krajobrazu wariantu. D: cały z bufora.
- K = 5k i 20k: snapshot po 5k tego samego przebiegu. Ten sam klucz dla wszystkich wariantów.

## Kierunek

- Współrzędne u jak w critic-curvature-es-v1: δθ = σ·S·ε.
- g_kryt = S·∇_θ średnia_s Q1(s, π_θ(s)) w aktorze, 2000 stanów on-policy z critic-curvature-es-v1 (Q1 jak w stracie aktora TD3).
- g_ES = Σ_i (R_i − R_aktor)·ε_i / (N·σ) per poziom, R z 30 epizodów local-search-v1.
- **Odstępstwo od specyfikacji (miara L1):** g_ES leży w rozpiętości 40 wektorów ε. Surowy cos(g_kryt, g_ES) ≤ ok. √(N/(N+P)) nawet dla idealnego krytyka: 0.16 w Walker2d i HalfCheetah, 0.27 w Swimmer. Próg 0.3 na surowym cos jest więc nieosiągalny z konstrukcji.
  - Sonda przed prerejestracją (oryginalny krytyk, 5 komórek): surowy cos −0.06 do +0.07, split-half cos g_ES (epizody 0-14 vs 15-29) 0.90-0.998.
  - Miara główna = sketch cos: cos(Σ_i (ε_i·g_kryt)·ε_i / N, g_ES). To gradient ES, jaki dałaby liniowa predykcja zwrotu przez krytyka na tych samych 40 ε. Idealny krytyk daje 1, losowy kierunek ok. 0 ± 0.16. Sonda: oryginalny krytyk −0.44 do +0.43.
  - Per krytyk: średnia po dwóch poziomach σ. Raportowane też: surowy cos z sufitem, podział na poziom „w próbie” (A: σ małe, B: σ duże) i drugi poziom.
- Progi L1 (0.3, +0.2) bez zmian, na sketch cos.

## Kroki

- d = S·g_kryt (jak d_ES = S·g_u), norma 1. α z bisekcji: dist_H oryginalnego krytyka = 1× odniesienia (średnia dist_H 40 zaburzeń σ małe, te same stany i H_a co w critic-curvature-es-v1). Wszystkie kroki tej samej wielkości w tej samej metryce.
- Porównanie: krok ES oracle 1× i krok ES k = 1 1× (10 losowań critic-curvature-es-v1) per poziom, krok oryginalnego krytyka, kroki A-D × K.
- Koszt: krajobraz = suma długości 80 epizodów + K aktualizacji krytyka. ES k = 1 = 41 epizodów jednego poziomu. Raportowane.

## Ocena

- Jeden batch na checkpoint (32 głowy ≥ 16 wierszy, ta sama ścieżka numeryczna), 30 świeżych seedów 8 000 000-029, wspólnych dla wszystkich głów.
- Zysk = średnia po epizodach R(głowa) − R(aktor), parowana.

## Hipotezy (zamknięte)

- X* = krytyk z A-C × K o najwyższym sketch cos (wybór po cos, nie po zysku). D* = D przy tym samym K.
- **L1:** sketch cos X* ≥ 0.3 i ≥ najlepszy sketch cos D (po K) + 0.2 w ≥ 4 z 6 komórek Walker2d.
- **L2:** zysk kroku X* > 0 i > zysku kroku D* w ≥ 4 z 6 komórek Walker2d.
- **HC (kontrola):** mediana po 6 komórkach HalfCheetah zysku kroku X* / |R_aktor| ≥ −2%.
- Swimmer: opisowo.
- **Reguła stopu:** L1 nie przechodzi → krajobraz nie uczy krytyka kierunku. Zamykamy kierunek metody i spisujemy wynik analityczny.
- Wielokrotność: X* to maksimum po 6 krytykach, a D to maksimum po 2. Margines 0.2 jest większy niż sd sketch cos losowego kierunku (ok. 0.16), ale samo maksimum po 6 przesuwa X* w górę.

## Raport

- Per komórka: sketch cos i surowy cos (przed, A-D × K, per poziom), strata TD, zyski kroków vs ES oracle, ES k = 1 i krok oryginalnego krytyka, koszt, werdykty L1, L2, HC.
- `python -m analysis.landscape_critic ../experiments/landscape-critic-v1/experiment.toml` (z `src/`) → `landscape_critic.{json,html}`.
