# paper-v2 etap 2 rozszerzony: środowiska trudne dla gradientu

- **Status:** krok A done (75/75 COMPLETED, 2026-11-14). Krok B (zmodyfikowany: tylko Swimmer) i kontrola konfiguracji wysłane 2026-11-15 (commit `9515679`).
- **Account:** `hpc-jakgil6519-1780267238` (SC-ERL)
- **Prerejestracja:** 2026-11-13, przed pierwszym runem etapu 2
- **Code:** git@github.com:jgilewicz/usc_erl_jax.git, branch `paper-v2`, commit `c626aaf` (kod jak `paper-core-v1` + `rl_only`)
- **Spec:** `experiment.toml`
- Zastępuje etapy 2a-2c z `../paper-v2/record.md` (Swimmer + Humanoid, 6 warunków ERL), z których nic nie zostało uruchomione.

## Pytanie

- Czy w środowiskach, w których TD3 jest znany jako słaby, któryś wariant ERL wygrywa z TD3 (H6)?
- Czy tam tańsza ocena wygrywa z pełnym ERL (H7)?

## Środowiska (lista zamknięta, uzasadnienie a priori)

| env | dlaczego TD3 słaby | wymiar akcji |
|---|---|---|
| Swimmer-v5 | γ=0.99 za krótkie dla nagrody Swimmera | 2 |
| Humanoid-v5 | wysoki wymiar akcji | 17 |
| dog-walk, dog-trot, dog-run (DMC) | wysoki wymiar akcji | 38 |

- Bez dokładania envów po wynikach.
- Grupa łatwa (HalfCheetah, Hopper, Walker2d, Ant): komórki już są (`paper-core-v1` + etap 1 `paper-v2`), bez nowych runów.

## Warunki

`td3`, `erl`, `sc_erl-always`, `sc_erl-random`, `prefix-h200`. Bez `prefix-td` i `hboot-td`.

- Kod i logowanie jak `paper-core-v1`, bez nowych metryk. Tag `paper-v2`, nazwy `<warunek>_<env>_s<seed>`.
- Mapowanie na `slurm_run_array.sh` (task = warunek × 10 + seed):
  - `SUITE=core`: erl 0-9, sc_erl-random 10-19, prefix-h200 20-29
  - `SUITE=controls`: td3 0-9, sc_erl-always 10-19

## Kroki

| krok | envy | seedy | runy |
|---|---|---|---|
| A0 (czas) | Humanoid, dog-walk, dog-trot, dog-run | 0 | 20 |
| A | wszystkie 5 trudnych | 0-2 | 75 (razem z A0) |
| B | tylko envy po bramce | 3-9 | ≤ 175 |

- Przed resztą A: szacunek całości z czasów A0.
- Czasy A0 (seed 0, h; trwające rzutowane liniowo): Humanoid erl 0.57, prefix 1.12, sc_erl-random 1.91, td3 ~2.6, sc_erl-always ~2.7 (suma ~8.9). dog-* erl 0.98, prefix ~1.9, sc_erl-random ~3.0, td3 ~3.1, sc_erl-always ~6.2 (suma ~15.2).
- Szacunek z A0: A0 ~55 GPU-h, reszta A ~135 GPU-h (~540 CPU-h), B najgorszy przypadek ~440 GPU-h (~1770 CPU-h). Razem ~630 GPU-h, ~2530 CPU-h.

## Bramka A → B (per env, seedy 0-2)

- Wynik końcowy: średnia `perf/eval_rl` z ostatnich 10% budżetu.
- Env przechodzi, jeśli któryś wariant ERL v spełnia oba warunki:
  - średnia(v) ≥ średnia(td3) + 0.10 · |średnia(td3)|
  - v > td3 w ≥ 2/3 par seedów (ten sam seed)

## Hipotezy

| | porównanie | kryterium |
|---|---|---|
| H6 | wariant ERL vs td3, envy po bramce | w ≥ 1 env CI różnicy IQM (v − td3) > 0 przy 10 seedach (Bonferroni) |
| H7 | prefix-h200, sc_erl-random, sc_erl-always vs erl, envy z H6 | per wariant: CI (v − erl) > 0 w każdym env z H6 (Bonferroni) |

- CI: sparowany bootstrap po seedach (10k), IQM wyniku końcowego, per env.
- H6, korekta Bonferroniego: CI na poziomie 1 − 0.05/k, k = liczba testowanych par (env po bramce × 4 warianty ERL). Seedy 0-9.
  - Sprawdzenie (opisowo): ten sam test tylko na seedach 3-9, niezależnych od bramki.
- H7, werdykt osobno dla każdego z 3 tańszych wariantów: PASS, jeśli CI różnicy IQM (v − erl) > 0 w każdym env z H6. Bonferroni w obrębie H7 (k = 3 × liczba envów z H6).

## Raport

- Macierz jak w `paper-v2` (summary): env × warunek, ranking IQM, krzywe.
- td3 w normalizacji. Normalizacja per env po 5 warunkach.
- Osobno grupa łatwa i trudna (ranking IQM liczony w każdej grupie osobno).

## Budżet

Saldo 2026-11-13 (`preflight.sh`): **5258.5 CPU-h, 1534.2 GPU-h**.

- Czasy z etapu 1 i `paper-core-v1` (łatwe envy), suma 5 warunków na (env, seed): 8.6 h średnio, 11.1 h max.
- Symulacja (lokalnie, losowe akcje): HalfCheetah 0.03 ms/krok, Swimmer 0.04, Humanoid 0.13, dog-walk 0.38. Przy 1M kroków to < 0.2 h, czas runu zależy głównie od GPU.
- Założenie przed A0: trudne envy ~1.5× łatwe (brak pomiaru).

| krok | runy | GPU-h | CPU-h (×4) |
|---|---|---|---|
| A0 | 20 | ~52 (limit `--time=12h`: ≤ 240) | ~210 |
| A (bez A0) | 55 | ~130 | ~520 |
| B (wszystkie 5 po bramce) | 175 | ~420 | ~1690 |
| razem, najgorszy przypadek | 250 | ~600 | ~2420 |

- `$HOME` na klastrze: 867 MB wolne. Logi SLURM na PD (`/lustre/pd03/hpc-jakgil6519-1780267238/logs`), nie w `logs/`.

## Runy

| Job ID | Data | Krok | Env | Array | Commit |
|---|---|---|---|---|---|
| 6310811 | 2026-11-13 | A0 | Humanoid-v5 | core 0,10,20, --time=12h | c626aaf |
| 6310812 | 2026-11-13 | A0 | Humanoid-v5 | controls 0,10, --time=12h | c626aaf |
| 6310813 | 2026-11-13 | A0 | dog-walk | core 0,10,20, --time=12h | c626aaf |
| 6310814 | 2026-11-13 | A0 | dog-walk | controls 0,10, --time=12h | c626aaf |
| 6310815 | 2026-11-13 | A0 | dog-trot | core 0,10,20, --time=12h | c626aaf |
| 6310816 | 2026-11-13 | A0 | dog-trot | controls 0,10, --time=12h | c626aaf |
| 6310817 | 2026-11-13 | A0 | dog-run | core 0,10,20, --time=12h | c626aaf |
| 6310818 | 2026-11-13 | A0 | dog-run | controls 0,10, --time=12h | c626aaf |
| 6313907 | 2026-11-13 | A | Humanoid-v5 | core 1,2,11,12,21,22, --time=12h | c626aaf |
| 6313908 | 2026-11-13 | A | Humanoid-v5 | controls 1,2,11,12, --time=12h | c626aaf |
| 6313909 | 2026-11-13 | A | dog-walk | core 1,2,11,12,21,22, --time=12h | c626aaf |
| 6313910 | 2026-11-13 | A | dog-walk | controls 1,2,11,12, --time=12h | c626aaf |
| 6313911 | 2026-11-13 | A | dog-trot | core 1,2,11,12,21,22, --time=12h | c626aaf |
| 6313912 | 2026-11-13 | A | dog-trot | controls 1,2,11,12, --time=12h | c626aaf |
| 6313913 | 2026-11-13 | A | dog-run | core 1,2,11,12,21,22, --time=12h | c626aaf |
| 6313914 | 2026-11-13 | A | dog-run | controls 1,2,11,12, --time=12h | c626aaf |
| 6313915 | 2026-11-13 | A | Swimmer-v5 | core 0-2,10-12,20-22, --time=12h | c626aaf |
| 6313916 | 2026-11-13 | A | Swimmer-v5 | controls 0-2,10-12, --time=12h | c626aaf |
| 6329266 | 2026-11-15 | B | Swimmer-v5 | core 3-9,13-19,23-29, --time=12h | 9515679 |
| 6329267 | 2026-11-15 | B | Swimmer-v5 | controls 3-9,13-19, --time=12h | 9515679 |
| 6329268 | 2026-11-15 | config-check | Swimmer-v5 | config-check 0-4, tag paper-v2-config, --time=12h | 9515679 |

- Logi SLURM: `/lustre/pd03/hpc-jakgil6519-1780267238/logs/slurm-<job>_<task>.{out,err}`. Partycja wybrana automatycznie: `lem-gpu-short`.

## Wyniki kroku A (2026-11-14)

- 75/75 runów COMPLETED, w wandb bez duplikatów. Raport: https://claude.ai/code/artifact/90ece3b6-f156-4bc0-a9c4-f56dbff5630e (`summary.py`, `summary_template.html`).
- Bramka (seedy 0-2, wynik końcowy):

| env | td3 | najlepszy wariant ERL | zysk | wygrane | bramka |
|---|---|---|---|---|---|
| Swimmer-v5 | 105.3 | erl 139.0 | +32% | 3/3 | PASS |
| Humanoid-v5 | 3918 | sc_erl-random 2824 | −28% | 1/3 | FAIL |
| dog-walk | 204.2 | erl 205.0 | +0% | 1/3 | FAIL |
| dog-trot | 119.5 | prefix-h200 144.6 | +21% | 3/3 | PASS |
| dog-run | 128.9 | erl 116.2 | −10% | 1/3 | FAIL |

- dog-trot: erl +16% (1/3), sc_erl-always +9% (3/3), oba blisko progu, ale nie spełniają obu warunków.
- Ranking IQM grupy trudnej (opis, 3 seedy): td3 0.75 [0.55, 0.97], reszta 0.20-0.48 z szerokimi CI.
- Koszt kroku A: ~193 GPU-h (suma `_runtime`), ~770 CPU-h.
- Krok B (propozycja): Swimmer + dog-trot, seedy 3-9, 5 warunków = 70 runów. Czasy z kroku A: Swimmer 9.0 h, dog-trot 15.3 h na seed (suma 5 warunków), razem ~170 GPU-h, ~680 CPU-h. Najdłuższy run ~6.5 h, mieści się w `--time=12h`.

## Kontrola implementacji (2026-11-14, bez nowych runów)

- Kod porównany z ERL-Re2 (commit `1ca10cf`) i EvoRainbow (commit `5fb900e`) oraz z ich papierami. Krok B wstrzymany do decyzji po kontroli.
- Metryka: oba repo raportują `best_reward = max(aktor RL, champion populacji)` (ERL-Re2 `run_re2.py:160`, EvoRainbow `run_EvoRainbow.py:173`). Nasz `perf/eval_rl` ocenia tylko aktora TD3 (`src/algos/erl.py:297-311`). Wyniki z literatury nie są porównywalne z naszymi 1:1.
- Najlepszy fitness populacji nie jest logowany (tylko `screen/cem_std`). Krzywej championa nie da się odtworzyć bez nowych runów.
- Swimmer erl s0-2: 90 generacji CEM na 1M kroków (pop 10 × 1000 + epizod RL). EvoRainbow (pop 5, H 200, p 0.4): ~280 generacji.
- `screen/cem_std`: 0.034 na starcie (√1e-3), skoki do 0.07-0.095 przy wstrzyknięciu aktora TD3, 0.005 na końcu (blisko `damp_limit`).
- γ: ERL-Re2 0.999 dla Swimmera (wszystkie baseline), EvoRainbow 0.99 w `run.sh` (EvoRainbow-Exp 0.999). My 0.99.
- Różnica w sieci: `SharedStateEmbedding` bez tanh po pierwszym LayerNorm (`src/modules/deep_modules.py:116-119`), u nich tanh po obu warstwach.

## Odstępstwa od prerejestracji (2026-11-15, przed krokiem B)

- **Krok B tylko Swimmer-v5**, seedy 3-9, 5 warunków (td3, erl, sc_erl-always, sc_erl-random, prefix-h200) = 35 runów.
  - dog-trot przeszedł bramkę, ale odpada: wszystkie metody osiągają ~10-15% maksimum nagrody DMC, sygnał nieinformatywny.
  - Decyzja po wynikach kroku A, więc H6 i H7 testowane tylko na Swimmerze (k mniejsze, Bonferroni po 1 env).
- **Metryka:** krok B na kodzie `da290cd`, loguje też `perf/eval_champion` i `pop/champion_fitness` (`../paper-v2-metric-pilot/record.md`). `eval_rl` bez zmian (test zgodności bitowy), wynik główny dalej `eval_rl`. Seedy 0-2 kroku A nie mają championa.
- **Statystyka:** IQM z bootstrapem bez parowania po seedach (resampling seedów niezależnie per warunek), P(A>B) jako porównanie niezależnych realizacji (wszystkie pary seedów).
  - Powód: seed nie ustala trajektorii na GPU. Ten sam seed i kod dał erl Swimmer `eval_rl` 116.9 vs 139.0, erl Ant 2840 vs 3417 (pilot vs wcześniejsze runy).
  - Zastępuje „sparowany bootstrap” w H6, H7 i „pary seedów” w bramce. Bramki nie liczę ponownie.
- **Częstość ewaluacji (bez zmian w kodzie):** `maybe_evaluate` (`src/algos/erl.py:300-307`) liczy poprzedni krok tak, jakby cała populacja szła przez środowisko. W sc_erl generacje są krótsze, więc ewaluacja jest częstsza niż co 5000 kroków (Swimmer: sc_erl-random ~313, sc_erl-always ~990 punktów, erl 91). Wynik końcowy to średnia z okna 900k-1M, więc różnica jest tylko w liczbie punktów. Kosztuje czas: champion ~16 s na ewaluację.

## Kontrola konfiguracji: erl-evorainbow (poza protokołem)

- Oddzielny warunek, nie wchodzi do H6, H7 ani rankingu. Pytanie: czy champion zbliża się do ~350 z literatury na Swimmerze.
- `erl-evorainbow` = erl (pełny epizod H 1000) z `pop_size=5`, `ea_tau=0.3`, `rl_to_ea_sync_period=1`. γ 0.99 jak td3. Reszta bez zmian (sigma_init 1e-3, CEM na ActorHead, bez tanh).
  - Nie odtwarza EvoRainbow w całości: tam H 200 i surogat krytyka (theta 0.4), u nas pełny epizod.
- `slurm_run_array.sh`: nowy `SUITE=config-check`, task = seed. Tag `paper-v2-config`. Seedy 0-4.
- Smoke lokalny (24k kroków): 6000 kroków na generację, overrides w `.hydra/config.yaml`.

## Krok B i kontrola: plan submitu

| array | SUITE | tag | task | runy | h na run (szacunek) |
|---|---|---|---|---|---|
| Swimmer core | core | paper-v2 | 3-9,13-19,23-29 | 21 | erl 1.0, sc_erl-random 3.1, prefix-h200 2.0 |
| Swimmer controls | controls | paper-v2 | 3-9,13-19 | 14 | td3 1.9, sc_erl-always ~8.3 |
| erl-evorainbow | config-check | paper-v2-config | 0-4 | 5 | ~1.4 |

- Czasy: pilot (z championem) i krok A. sc_erl-always = 3.8 h (krok A) + ~990 × 16 s championa. erl-evorainbow: ~167 generacji, każda z ewaluacją.
- Razem ~121 GPU-h, ~485 CPU-h (`-c 4`). Najdłuższy run ~8.5 h przy `--time=12h`.

## Raport kosztu (bez nowych runów, 2026-11-15)

- Artefakt: https://claude.ai/code/artifact/a2b60a78-f45c-4295-803d-550ca8b32088 (generator `cost.py` + `cost_template.html`)
- Łatwe MuJoCo (4 envy × 10 seedów), znormalizowany IQM: td3 0.96, sc_erl-always 1.00, sc_erl-random 0.80, prefix-h200 0.72, erl 0.08
- Łatwe: każdy tańszy wariant > erl (P 0.76-0.83, CI nad 0.5), żaden > td3 (max P 0.48, sc_erl-always)
- Swimmer (seedy 0-2): erl najwyżej (IQM 143), sc_erl-always najniżej (53), żadne CI P(A>B) nie wyklucza 0.5
- Dominacja (pilot): champion > RL w 53-91% ewaluacji tylko w Swimmer erl, poza tym 20-67% (Ant 34-49%)
- Do odświeżenia po kroku B (Swimmer 10 seedów)

## Wyniki kroku B i kontroli konfiguracji (2026-11-15)

- 40/40 tasków COMPLETED (6329266, 6329267, 6329268), Swimmer-v5 seedy 0-9 kompletne dla 5 warunków
- IQM wyniku końcowego (eval_rl): prefix-h200 138.1, erl 129.3, td3 120.3, sc_erl-random 119.0, sc_erl-always 99.9
- H6 (wariant − td3, różnica IQM, bootstrap bez parowania, Bonferroni k = 4): FAIL dla wszystkich
  - prefix-h200 +17.8 [−26.5, +66.7], erl +9.0 [−31.0, +58.0], sc_erl-random −1.3 [−52.2, +50.5], sc_erl-always −20.4 [−74.7, +40.6]
- H7 (warunkowa na H6): nie dotyczy. Dla informacji vs erl: prefix-h200 +8.8 [−34.6, +45.1], sc_erl-random −10.3, sc_erl-always −29.4, wszystkie CI zawierają 0
- P(A>B) bez korekty: prefix-h200 > td3 0.80 [0.56, 1.00], erl > td3 0.68 [0.42, 0.90]
- Champion w kroku B (seedy 3-9, okno 900k-1M): ≈ aktor RL we wszystkich warunkach (erl 115.4 vs 116.7), champion > RL w 48-57% ewaluacji. Duża przewaga championa z pilota (erl s1: 264.8) się nie powtórzyła
- erl-evorainbow (seedy 0-4): champion 126.5 (max 160.2), eval_rl 121.6, champion > RL w 65% ewaluacji. Daleko od ~350 z literatury, więc konfiguracja gene flow nie tłumaczy luki
- Raport kosztu odświeżony (Swimmer 10 seedów): https://claude.ai/code/artifact/a2b60a78-f45c-4295-803d-550ca8b32088
- Analiza: `stepb.py` (cache w `cache/stepb.json`, `cache/champion/`). Od refaktoru B: dominacja championa w `python -m analysis.cost`, H6/H7 w `python -m analysis.compare` (Holm zamiast Bonferroniego, te same werdykty)
