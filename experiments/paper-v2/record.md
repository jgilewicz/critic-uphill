# paper-v2: kontrole i 6 środowisk

- **Status:** etap 1 done, etap 2 przeniesiony do `../paper-v2-stage2/`
- **Account:** `hpc-jakgil6519-1780267238` (SC-ERL)
- **Started:** 2026-11-12
- **Code:** git@github.com:jgilewicz/usc_erl_jax.git, branch `paper-v2`
- **Spec:** `experiment.toml` (siatka, metryka, porównania)

## Pytanie

- Czy populacja pomaga ponad sam TD3 (H1)?
- Czy zysk prefiksu bierze się z podziału budżetu, a nie z jakości oceny (H2)?
- Czy wynik prefiks > ERL przenosi się na Swimmer i Humanoid (H3)?

## Zasady

- Kod i logowanie jak `paper-core-v1` (commit `67c7b58`), bez nowych metryk.
- Jedyna zmiana kodu: tryb `eval_mode=rl_only` (warunek `td3`), commit `0ce868b`.
  - `train_and_merge` rozdzielone na `train_td3` + merge, kolejność kluczy RNG bez zmian.
  - `maybe_evaluate`: okno ewaluacji dla `rl_only` = sam epizod RL, dla pozostałych trybów bez zmian.
- CORE na HalfCheetah, Hopper, Walker2d, Ant nie jest powtarzany: komórki z tagu `paper-core-v1` (`join` w toml).
- Nowe runy: tag `paper-v2`, nazwy `<warunek>_<env>_s<seed>`, `WANDB_MODE=online`.

## Odstępstwa od prerejestracji

- 2026-11-13, po etapie 1 (decyzja autora): porównania tylko w obrębie ERL. `td3` usunięty z `conditions` i z porównań, H1 odłożone.
  - Wynik H1 (FAIL) zostaje zapisany niżej. Runy `td3` zostają w wandb pod tagiem `paper-v2`.
  - Normalizacja po 6 warunkach ERL. Etap 2 bez `td3` (6 warunków).

## Hipotezy (prerejestracja)

| | etap | porównanie | kryterium | FAIL znaczy |
|---|---|---|---|---|
| H1 | 1 | prefix-h200 vs td3 | 95% CI różnicy IQM (wynik końcowy, znormalizowany) > 0 | populacja nie pomaga |
| H2 | 1 | prefix-h200 vs sc_erl-always | \|ΔIQM\| < 0.1 | różnica jakości oceny, nie podziału budżetu |
| H3 | 2 | prefix-h200 vs erl | wyższa średnia i ≥ 7/10 wygranych seedów w Swimmer i w Humanoid | prefiks nie przenosi się |

- Prefiks vs SC-ERL w etapie 2: opisowo, bez testu.
- H1 i H2 liczone na 4 envach etapu 1, po etapie 2 powtórzone na 6.

## Statystyka

- Wynik końcowy: średnia `perf/eval_rl` z ostatnich 10% budżetu (1M `env_steps`).
- AUC: średnia `perf/eval_rl` po wszystkich ewaluacjach (pole pod krzywą / budżet).
- Normalizacja per env: 0 = najgorsza średnia warunku, 1 = najlepsza, po wszystkich 7 warunkach (liczby nie są wprost porównywalne z raportem `paper-core-v1`, który normalizował po 5).
- IQM z 95% CI ze stratyfikowanego bootstrapu po (env, seed), dla wyniku końcowego i AUC.
- P(A>B) po parach seedów (ten sam env, ten sam seed).
- Po etapie 2: Wilcoxon po 6 envach + Holm po H1, H2, H3.

## Etapy

| etap | envy | warunki | runy | stan |
|---|---|---|---|---|
| 0 | HalfCheetah | prefix-h200, seed 0, 100k kroków (regresja) | 1 lokalnie | PASS |
| 1 | HalfCheetah, Hopper, Walker2d, Ant | td3, sc_erl-always × seedy 0-9 | 80 | done 2026-11-13, H1 FAIL, H2 FAIL |
| 2a | Humanoid | 6 warunków ERL × seed 0 (pilot: czas, czy cokolwiek się uczy w 1M) | 6 | zastąpiony, 0 runów |
| 2b | Swimmer | 6 warunków ERL × seedy 0-9 | 60 | zastąpiony, 0 runów |
| 2c | Humanoid | 6 warunków ERL × seedy 1-9 | 54 | zastąpiony, 0 runów |

- 2026-11-13: etapy 2a-2c zastąpione przed jakimkolwiek runem przez `../paper-v2-stage2/` (5 trudnych envów, 5 warunków z td3, H6/H7). H3 bez testu.
- Swimmer `6256649` (stary CORE): już `CANCELLED`, nigdy nie wystartował, nic do wycofania.

## Budżet

Saldo SC-ERL 2026-11-12 (`discover.sh`): **6110.9 CPU-h, 1747.2 GPU-h**.
Czas na run z `paper-core-v1` (200 runów, lem-gpu, `-c 4`, 1 H100):

| warunek | średnio | max |
|---|---|---|
| erl | 0.52 h | 0.66 h |
| sc_erl-random | 1.69 h | 2.42 h |
| prefix-h200 | 1.07 h | 1.32 h |
| prefix-td | 0.68 h | 0.84 h |
| hboot-td | 0.68 h | 0.86 h |

| etap | runy | założenie | GPU-h | CPU-h (×4) |
|---|---|---|---|---|
| 1 | 80 | td3 ~1.0 h (brak pomiaru), sc_erl-always ~1.7 h | ~110 (max ~160) | ~430 (max ~640) |
| 2a | 7 | Humanoid ~3 h (brak pomiaru) | ~20 | ~85 |
| 2b | 70 | średnia warunków ~1.05 h | ~75 | ~300 |
| 2c | 63 | z pilota | ~190 przy 3 h | ~760 przy 3 h |
| razem | 220 | | ~395 | ~1575 |

- Zapas: ok. 4× w CPU-h, 4× w GPU-h. Przed 2c przeliczyć z czasów pilota.

## Runy

| Job ID | Data | Etap | Env | Array | Commit |
|---|---|---|---|---|---|
| 6306493 | 2026-11-12 | 1 | HalfCheetah-v5 | controls 0-19, --time=6h | c626aaf |
| 6306494 | 2026-11-12 | 1 | Hopper-v5 | controls 0-19, --time=6h | c626aaf |
| 6306495 | 2026-11-12 | 1 | Walker2d-v5 | controls 0-19, --time=6h | c626aaf |
| 6306496 | 2026-11-12 | 1 | Ant-v5 | controls 0-19, --time=6h | c626aaf |

## Wyniki

### Etap 0: regresja (PASS)

- Commit kodu `0ce868b`, lokalnie (Mac M, CPU, JAX), `prefix-h200` HalfCheetah-v5 seed 0, 100k `env_steps`, 31 generacji.
- Porównanie z 10 seedami `prefix-h200` HalfCheetah z `paper-core-v1` (H100), na tych samych krokach ewaluacji.
- Kroki ewaluacji identyczne jak w CORE (11k, 17k, 20k, ..., 101k), więc licznik kroków i wyzwalacz ewaluacji bez zmian.
- 16/19 punktów w kopercie min–max CORE. 3 punkty nad max o ≤ 50 (47k: 3403 vs 3354, 62k: 4163 vs 4153, 77k: 4718 vs 4700). Żaden poniżej min.
- Przy 101k: 4848 (CORE 3356–5170, seed 0 w CORE 4445). Bitowej zgodności z CORE s0 nie ma (CPU vs GPU).

| krok | CORE min | CORE max | lokalnie |
|---|---|---|---|
| 20k | -809 | 595 | 307 |
| 41k | 1417 | 3305 | 3161 |
| 62k | 1633 | 4153 | 4163 |
| 80k | 2128 | 4963 | 4466 |
| 101k | 3356 | 5170 | 4848 |

### Etap 1: kontrole (H1 FAIL, H2 FAIL)

- Raport: https://claude.ai/code/artifact/09137686-778a-4d46-a1fa-811dd9ab5ea6
- Statystyki: `uv run python experiments/paper-v2/analyze.py experiments/paper-v2/experiment.toml` (10k paired stratified bootstrap).
- 80/80 `COMPLETED`, 0 failed. Koszt: **213.1 GPU-h, 852 CPU-h** (szacunek był ~110 / ~430).
  - td3: 1.96 h średnio (max 2.26), sc_erl-always: 3.36 h (max 4.39). Oba poniżej `--time=6h`.
- Kontrola td3 sprawdzona: rollouty populacji też idą do bufora, więc oba warunki mają 1M przejść i 1M aktualizacji. Wyniki td3 w zakresie typowym dla TD3 w MuJoCo.

IQM znormalizowanego wyniku (7 warunków, 4 envy × 10 seedów):

| warunek | final IQM [95% CI] | AUC IQM [95% CI] |
|---|---|---|
| sc_erl-always | 1.00 [0.75, 1.13] | 0.87 [0.69, 1.05] |
| td3 | 0.96 [0.81, 1.10] | 1.04 [0.86, 1.18] |
| sc_erl-random | 0.80 [0.61, 0.94] | 0.67 [0.54, 0.78] |
| prefix-h200 | 0.72 [0.57, 0.83] | 0.65 [0.50, 0.76] |
| prefix-td | 0.72 [0.55, 0.86] | 0.49 [0.37, 0.60] |
| hboot-td | 0.65 [0.48, 0.81] | 0.51 [0.43, 0.60] |
| erl | 0.08 [-0.18, 0.34] | -0.01 [-0.11, 0.09] |

- **H1 FAIL:** prefix-h200 − td3 = −0.24 [−0.44, −0.04] (final), −0.38 [−0.59, −0.17] (AUC). P(A>B) 0.28. td3 lepszy na 4/4 envach, prefiks wygrywa 2-3/10 seedów.
- **H2 FAIL:** prefix-h200 − sc_erl-always = −0.28 [−0.49, +0.01] (final), −0.22 [−0.47, +0.01] (AUC). P(A>B) 0.33. Kierunek: surrogat na wszystkich > prefiks (CI dotyka 0).

| env | prefix-h200 | td3 | wygrane prefiks | sc_erl-always | wygrane prefiks |
|---|---|---|---|---|---|
| HalfCheetah-v5 | 8834 ± 1078 | 9517 ± 1558 | 3/10 | 9665 ± 1358 | 3/10 |
| Hopper-v5 | 2797 ± 529 | 3086 ± 282 | 3/10 | 2752 ± 899 | 4/10 |
| Walker2d-v5 | 3435 ± 1080 | 4180 ± 486 | 2/10 | 3507 ± 1436 | 3/10 |
| Ant-v5 | 4243 ± 632 | 4783 ± 912 | 3/10 | 5049 ± 904 | 3/10 |

- Normalizacja po 7 warunkach: IQM prefix-h200 spadło z 0.84 (`paper-core-v1`, 5 warunków) do 0.72, bo td3 i sc_erl-always podniosły górną granicę.

#### Etap 1 bez td3 (normalizacja po 6 warunkach ERL)

| warunek | final IQM [95% CI] | AUC IQM [95% CI] |
|---|---|---|
| sc_erl-always | 1.02 [0.76, 1.14] | 0.97 [0.76, 1.18] |
| sc_erl-random | 0.81 [0.63, 0.96] | 0.77 [0.62, 0.90] |
| prefix-h200 | 0.74 [0.58, 0.85] | 0.72 [0.55, 0.84] |
| prefix-td | 0.73 [0.56, 0.88] | 0.53 [0.41, 0.67] |
| hboot-td | 0.66 [0.49, 0.83] | 0.57 [0.47, 0.67] |
| erl | 0.08 [-0.18, 0.35] | -0.01 [-0.13, 0.10] |

- H2 FAIL bez zmian: −0.29 [−0.50, +0.01] (final), −0.25 [−0.53, +0.01] (AUC), P(A>B) 0.33.
- prefix-h200 vs erl na 4 envach (opisowo, H3 dotyczy Swimmer i Humanoid): +0.65 [+0.35, +0.94], P(A>B) 0.80, wygrane seedy 7, 8, 8, 9 /10.
