# critic-degeneracy-v1: czy krytyk TD3 rankuje sąsiadów aktora tylko przez odległość

- **Status:** wysłane 2026-11-16 z commita `670468d`.
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
| 6337576 | td3 HalfCheetah-v5 s0-1 + checkpointy | PENDING |
| 6337582 | offline HalfCheetah-v5 (afterok 6337576) | PENDING |
| 6337583 | td3 Walker2d-v5 s0-1 + checkpointy | PENDING |
| 6337584 | offline Walker2d-v5 (afterok 6337583) | PENDING |
| 6337585 | td3 Swimmer-v5 s0-1 + checkpointy | PENDING |
| 6337586 | offline Swimmer-v5 (afterok 6337585) | PENDING |

- Pierwszy `sbatch` (HalfCheetah, trening) zwrócił socket timeout, ale job powstał (sprawdzone w `squeue`). Offline HalfCheetah dosłany ręcznie z tą samą zależnością. Brak duplikatów.
