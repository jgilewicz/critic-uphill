# critic-degeneracy-v1: czy krytyk TD3 rankuje sąsiadów aktora tylko przez odległość

- **Status:** zakończone 2026-11-16, decyzja **H1-noH3**.
- **Account:** `hpc-jakgil6519-1780267238` (SC-ERL)
- **Prerejestracja:** 2026-11-16, przed pierwszym runem
- **Code:** git@github.com:jgilewicz/usc_erl_jax.git, branch `critic-degeneracy-v1` (z `refactor-b`, protokół 3). Zmiana w treningu: tylko zapis checkpointów (`checkpoint.dir`, `checkpoint.at`), algorytm bez zmian.
- **Spec:** `experiment.toml` (`[offline]`, `[[hypotheses]]`, `[decision]`)

## Pytanie

- Czy ocena krytyka TD3 dla zaburzeń aktora redukuje się do (ważonej hesjanem) odległości od aktora, bo aktor stoi w maksimum krytyka?
- Czy krytyk niezależny od aktora (FQE) niesie informację o kierunku poprawy?

## Źródło polityk

- `td3` (`eval.mode=rl_only`), HalfCheetah-v5, Walker2d-v5, Swimmer-v5, seedy 0-1, 1M kroków.
- Checkpointy przy 30%, 60%, 100% budżetu (kroki 300k, 600k, 1M): embedding + aktor, obie głowy krytyka (online), bufor powtórek (wypełniona część + `ptr`). 18 checkpointów.
- Bufor 1M bez zawijania przy 1M kroków.
- Zapis: `$PD/critic-degeneracy-v1/ckpt/<env>_s<seed>/step_<N>/{nets.eqx,buffer.npz,meta.json}`.

## Populacje kontrolowane (na checkpoint)

- Zaburzenia głowy aktora (`ActorHead`, jak CEM w ERL), per liść (waga, bias): δ ~ N(0, (σ·std(liścia))²). Embedding wspólny, niezaburzony.
- 2 poziomy σ × 40 zaburzeń.
- **Kalibracja σ** (per env, raz): seed 0, checkpoint 30%.
  - Siatka σ ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}, 20 zaburzeń × 3 epizody na σ.
  - Spadek = 1 − średni zwrot zaburzeń / zwrot aktora.
  - σ = pierwsze przecięcie 0.05 i 0.20, interpolacja liniowa w log σ.
  - Potem σ stałe dla 6 checkpointów envu (`sigma.json`, kalibracja nie nadpisuje).
  - Błąd zamiast zgadywania: zwrot aktora ≤ 0 albo cel poza siatką → rozszerzyć siatkę.
- **Prawda:** średnia z 10 pełnych epizodów (H = 1000) na zaburzenie.
  - Seedy resetu 1 000 000-1 000 009, poza seedami ewaluacji treningu (seed + 10 000), wspólne dla wszystkich zaburzeń (common random numbers: każda polityka startuje z tych samych stanów).
- **Sufit szumu:** Spearman połówek (epizody 0-4 vs 5-9), Spearman-Brown 2r/(1+r). Maksymalny osiągalny ρ z prawdą ≈ √sufit.
- **Prefiks:** 1 osobny epizod (seed 1 000 010), zwrot z pierwszych 200 kroków. Polityka i MuJoCo deterministyczne, więc rollout 200 kroków = pierwsze 200 kroków pełnego epizodu.

## Krytyk niezależny (FQE)

- Ta sama architektura co krytyk TD3 (twin, 400-300, LayerNorm), od zera.
- Cel TD z a' = clip(aktor(s') + clip(N(0, 0.2²), ±0.5)), min z dwóch celów, γ = 0.99, τ = 0.005, Adam 1e-3, batch 256, 100k kroków gradientu, bufor z checkpointu.
- Aktor zamrożony (aktor z checkpointu = aktor celu). Implementacja: `_critic_update` TD3 bez kroku aktora (`make_fqe_train`).

## Miary (2000 różnych stanów z ostatnich 100k przejść bufora)

- δ_a(s) = π'(s) − π(s), Q = min(Q1, Q2) (jak w TD3).
- score = średnia_s Q(s, π'(s)), osobno TD3 i FQE.
- dist_L2 = średnia_s ‖δ_a‖².
- g, H: gradient i hesjan Q po akcji w a = π(s) (autodiff, przez min, czyli głowa minimalna w a), per stan.
- t1 = średnia_s g·δ_a, t2 = średnia_s ½ δ_aᵀ H δ_a, osobno dla TD3 i FQE (każdy krytyk swoje g, H).
- dist_H = −t2_TD3 (dodatnie w maksimum), więc −dist_H = t2_TD3.
- Stosunek 1./2. rzędu w komórce = mediana |t1| / mediana |t2| po 40 zaburzeniach.

## Statystyki

- Komórka = env × seed × checkpoint × σ (36 komórek po 40 zaburzeń).
- Spearman z prawdą: score_TD3, score_FQE, −dist_L2, −dist_H, prefiks200.
- ρ(score_TD3, −dist_H), ρ(score_FQE, −dist_H).
- Cząstkowa ρ(score, prawda | dist_H) = (ρxy − ρxz·ρyz) / √((1 − ρxz²)(1 − ρyz²)), z rang Spearmana.

## Hipotezy i reguły (zamknięte)

| id | reguła |
|---|---|
| H1 degeneracja | mediana po komórkach ρ(score_TD3, −dist_H) ≥ 0.8 **i** mediana stosunku 1./2. rzędu TD3 ≤ 0.2 |
| H2 brak informacji ponad odległość | \|mediana cząstkowej ρ(score_TD3, prawda \| dist_H)\| ≤ 0.1 |
| H3 poprawka | per env mediana (cząstkowa FQE − cząstkowa TD3) po 12 komórkach ≥ 0.15 w ≥ 2 z 3 envów **i** mediana stosunku FQE ≥ 2 × mediana stosunku TD3 |

- „Wyraźnie większy” stosunek FQE z polecenia = ≥ 2× (ustalone tutaj, przed wynikami).
- Decyzja:
  - H1 + H3 → etap 2: surogat z krytyka niezależnego w ERL.
  - H1 bez H3 → degeneracja potwierdzona, poprawka nie działa: notatka analityczna i inny wariant poprawki (krytyk średniej populacji przy centrum ≠ aktor).
  - H1 nie przechodzi → hipoteza obalona, stop.
- H2 raportowana, nie wchodzi do decyzji.

## Raport

- `python -m analysis.degeneracy ../experiments/critic-degeneracy-v1/experiment.toml` (z `src/`) → tabela + `degeneracy.json`.
- Strona HTML po wynikach: tabela env × checkpoint × σ (wszystkie korelacje + sufit szumu), wykres score_TD3 vs dist_H z kolorem = prawda, stosunki 1./2. rzędu.

## Uruchomienie

```bash
./scripts/submit_degeneracy.sh HalfCheetah-v5 Walker2d-v5 Swimmer-v5
# po zakończeniu, lokalnie:
rsync -a wcss:/lustre/pd03/hpc-jakgil6519-1780267238/critic-degeneracy-v1/out/ \
  experiments/critic-degeneracy-v1/cache/offline/
```

- Per env: `SUITE=controls` array 0-1 (td3 seedy 0, 1, tag `critic-degeneracy-v1`) + `slurm_offline_degeneracy.sh` z `afterok` (kalibracja, potem 6 checkpointów).

## Koszt (szacunek przed startem)

| część | zadania | czas | CPU-h | GPU-h |
|---|---|---|---|---|
| trening td3 (-c 4, 1 H100) | 6 | ~1.7-2.1 h (zmierzone, paper-v2) | ~50 | ~12.5 |
| offline (-c 4, 1 H100) | 3 | ~0.5-1 h (limit 3 h) | ~6-12 (max 36) | ~1.5-3 (max 9) |
| **razem** | 9 | | **~56-62** | **~14-16** |

- Smoke lokalny (CPU): rollouty 81 polityk × 11 epizodów ≈ 38 s na checkpoint, FQE 10k kroków + miary ≈ 25 s.
- Saldo 2026-11-16: 3845 CPU-h, 1181 GPU-h.
- PD: checkpointy ≈ 1.6 GB (bufory 300k + 600k + 1M na run).

## Jobs

| job | co | stan |
|---|---|---|
| 6337576 | td3 HalfCheetah-v5 s0-1 + checkpointy | COMPLETED |
| 6337582 | offline HalfCheetah-v5 (afterok 6337576) | COMPLETED |
| 6337583 | td3 Walker2d-v5 s0-1 + checkpointy | COMPLETED |
| 6337584 | offline Walker2d-v5 (afterok 6337583) | COMPLETED |
| 6337585 | td3 Swimmer-v5 s0-1 + checkpointy | COMPLETED |
| 6337586 | offline Swimmer-v5 (afterok 6337585) | COMPLETED |

- Pierwszy `sbatch` (HalfCheetah, trening) zwrócił socket timeout, ale job powstał (sprawdzone w `squeue`). Offline HalfCheetah dosłany ręcznie z tą samą zależnością. Brak duplikatów.

## Wyniki (2026-11-16)

- Pokrycie 36/36 komórek. Wszystkie 9 jobów COMPLETED (trening 1:38-2:03 h, offline 7-11 min). Koszt ≈ 48 CPU-h, ≈ 12 GPU-h.
- σ (5%, 20% spadku): HalfCheetah 0.051, 0.114 · Walker2d 0.060, 0.123 · Swimmer 0.095, 0.494.
- Raport: https://claude.ai/code/artifact/f53b69d3-bc81-485a-9e42-31dfc160f13f (`degeneracy.html`, dane `degeneracy.json`).

| hipoteza | wartość (mediana po komórkach) | próg | wynik |
|---|---|---|---|
| H1 | ρ(score_TD3, −dist_H) = 0.86, stosunek TD3 = 0.095 | ≥ 0.8, ≤ 0.2 | przechodzi |
| H2 | cząstkowa ρ TD3 = 0.19 | \|·\| ≤ 0.1 | nie przechodzi |
| H3 | zysk FQE: HC −0.27, Walker2d 0.07, Swimmer −0.07, stosunek FQE/TD3 = 3.1× | ≥ 0.15 w ≥ 2/3, ≥ 2× | nie przechodzi |

- **Decyzja (z prerejestracji): H1-noH3.** Degeneracja potwierdzona, poprawka FQE nie działa. Dalej: notatka analityczna + krytyk średniej populacji (centrum ≠ aktor).
- Per env: degeneracja w HalfCheetah (ρ 0.93, stosunek 0.071) i Walker2d (0.85, 0.070). Swimmer nie spełnia żadnego progu (0.71, 1.32), H1 pooled przechodzi dzięki 2 z 3 envów.
- −dist_H przewiduje prawdę co najmniej tak dobrze jak score_TD3 (mediana ρ 0.43 vs 0.39).
- Krytyk TD3 zachowuje informację ponad odległość głównie w HalfCheetah (cząstkowa 0.41), Walker2d 0.19, Swimmer 0.02.
- FQE ma większy wyraz 1. rzędu (0.29 vs 0.095), ale cząstkowa ρ niższa (0.07 vs 0.19): gradient FQE nie wskazuje kierunku poprawy prawdziwego zwrotu.
- Sufit szumu prawdy: mediana 0.94, min 0.77. Niskie korelacje w Walker2d i Swimmer to słabość predyktorów, nie szum.
- Prefiks 200: dobry w HalfCheetah (0.78), słaby w Walker2d (0.06) i Swimmer (0.23).
- Uwaga: spadek przy stałym σ w Walker2d waha się między checkpointami od −0.06 do 0.62 (kalibracja na jednym checkpoincie słabo przenosi się na inne).

## Walidacja poprawy (prerejestracja 2026-11-16, przed rolloutami walidacji)

- Pytanie: czy zaburzenia z prawdą > aktor to realna poprawa, czy szum selekcji z 10 epizodów (klątwa zwycięzcy)?
- Te same 36 komórek i te same zaburzenia (odtworzone z `[offline].key`), bez nowych checkpointów i bez nowego treningu. Spec: `experiment.toml` `[validation]`.
- **Kandydaci na komórkę** (suma zbiorów): zaburzenia z prawdą > aktor, top-1 wg prawdy, top-3 wg score_TD3, top-3 wg prefiksu. Plus aktor.
  - Top-1 wg prawdy dodany do polecenia: bez niego komórki bez zaburzenia lepszego od aktora nie miałyby nieobciążonego „najlepszego”.
- **Ponowna ocena:** 20 pełnych epizodów, seedy resetu 2 000 000-2 000 019 (nowe, rozłączne z prawdą, prefiksem i kalibracją), wspólne dla aktora i kandydatów.
- **Różnica:** per epizod d_e = R_pert − R_aktor (te same stany startowe), średnia i dwustronny 95% przedział t (19 df).
- **Sprawdzenie odtworzenia:** na 6 checkpointach 300k (z buforem) dist_L2 wszystkich 80 zaburzeń musi się zgadzać z zapisanym (rtol 1e-3), inaczej stop. (zmienione na 5e-2, patrz wyniki walidacji)
- **Raport per env × σ** (6 komórek):
  - odsetek zaburzeń lepszych od aktora po ponownej ocenie (CI różnicy > 0) wśród 240 zaburzeń i wśród tych z prawdą > aktor,
  - zysk najlepszego po ponownej ocenie (max po kandydatach, optymistyczny: selekcja na tych samych danych),
  - zysk top-1 wg prawdy, top-1 wg krytyka TD3 i top-1 wg prefiksu po ponownej ocenie (względem aktora, %).
- **Reguła (zamknięta):** poprawa realna w envie, jeśli w ≥ 50% jego 12 komórek (≥ 6) co najmniej jedno zaburzenie ma 95% CI różnicy w całości powyżej zera.
- Raportowane obok, poza regułą: to samo z poprawką Bonferroniego w komórce (α / liczba kandydatów), bo przy wielu kandydatach jedno CI > 0 może wyjść przypadkiem.
- Uruchomienie lokalnie (CPU, bez grantu): `python -m offline.critic_degeneracy validate <toml> <ckpt_root> <out_dir> <env>`.

### Wyniki walidacji (2026-11-16)

- Raport: ta sama strona, sekcja „Walidacja”: https://claude.ai/code/artifact/f53b69d3-bc81-485a-9e42-31dfc160f13f
- 36/36 komórek, 386 kandydatów + 18 aktorów × 20 epizodów = 8080 epizodów. Lokalnie na CPU (M-series) ok. 6 min, bez grantu.
- **Odstępstwo od prerejestracji:** tolerancja sprawdzenia odtworzenia 1e-3 → 5e-2 (zmiana przed obejrzeniem jakichkolwiek wyników walidacji).
  - Przy 1e-3 sprawdzenie zatrzymało pierwszy checkpoint: maks. błąd względny dist_L2 1.0e-2.
  - Diagnoza na 6 checkpointach 300k: mediana błędu 1e-3 do 5e-3, maks. 2.2e-2, korelacja z zapisanym dist_L2 ≥ 0.99999. Zaburzenia z innym kluczem: mediana błędu 0.28 do 0.56.
  - Przyczyna: matmule GPU (TF32) przy ewaluacji vs CPU przy walidacji. Zaburzenia są te same. Reguła decyzyjna bez zmian.

| env | σ | lepsze / 240 | lepsze / prawda > aktor | komórki CI > 0 | Bonferroni | najlepszy (optym.) | top-1 prawda | top-1 TD3 | top-1 prefiks |
|---|---|---|---|---|---|---|---|---|---|
| HalfCheetah | 5% | 7 (2.9%) | 7/19 | 1/6 | 1/6 | +0.2% | +0.1% | −1.4% | −2.8% |
| HalfCheetah | 20% | 0 | 0/1 | 0/6 | 0/6 | −7.5% | −7.5% | −11.0% | −14.4% |
| Walker2d | 5% | 73 (30.4%) | 73/95 | 6/6 | 5/6 | +5.8% | +5.8% | −0.7% | −10.2% |
| Walker2d | 20% | 28 (11.7%) | 27/45 | 4/6 | 4/6 | +3.2% | +1.4% | +0.4% | −25.6% |
| Swimmer | 5% | 19 (7.9%) | 19/47 | 5/6 | 5/6 | +4.0% | +2.8% | −1.7% | −2.5% |
| Swimmer | 20% | 11 (4.6%) | 11/15 | 5/6 | 4/6 | +12.5% | +12.5% | −17.3% | −17.8% |

Zyski = mediany po 6 komórkach, względem zwrotu aktora na tych samych 20 epizodach.

- **Werdykt (reguła):** HalfCheetah poprawa nierealna (1/12 komórek), Walker2d realna (10/12, Bonferroni 9/12), Swimmer realna (10/12, Bonferroni 9/12).
- HalfCheetah: 7 z 19 „lepszych” się potwierdza, wszystkie przy σ 5% i o ok. 0.2%. Aktor TD3 jest tu lokalnym maksimum w skali tych zaburzeń, więc pierwotne „lepsze” to klątwa zwycięzcy.
- Walker2d: 30% wszystkich zaburzeń przy σ 5% jest naprawdę lepsze od aktora. Aktor TD3 nie jest w lokalnym maksimum, mała losowa perturbacja często pomaga.
- Wybór top-1 wg 10 epizodów prawdy daje realny zysk w Walker2d i Swimmer (+1.4% do +12.5%).
- Wybór top-1 wg krytyka TD3 daje ok. 0 (−1.7% do +0.4%) albo stratę (−11%, −17% przy σ 20%). Krytyk nie wskazuje lepszych zaburzeń, zgodnie z H1 (ranking ≈ odległość od aktora, więc top-1 TD3 ≈ najbliższe aktorowi).
- Wybór top-1 wg prefiksu 200 jest najgorszy wszędzie (do −26% w Walker2d).
- Wniosek dla tezy: w 2 z 3 envów populacja wokół aktora zawiera realnie lepsze polityki, ale znajduje je tylko pełna ocena. Ani krytyk TD3, ani prefiks 200 ich nie wybierają.
