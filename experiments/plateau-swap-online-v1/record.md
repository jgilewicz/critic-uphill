# plateau-swap-online-v1: czy podmiana aktora w plateau (G-d) pomaga TD3 online

- **Status:** prerejestracja 2026-11-17. Bieg zakończony 2026-11-18: brak sygnału.
- **Charakter:** eksploracyjnie, 5 seedów. Bez decyzji o metodzie: wynik wraca do uzgodnienia.
- **Code:** branch `plateau-swap-online-v1` (od `gibo-pilot-v1`).
- **Spec:** `experiment.toml`. Hiperparametry fazy czytane w runie z `../gibo-pilot-v1/experiment.toml` (ramię G-d), bez kopii.

## Pytanie

- Czy TD3 + faza podmiany aktora przy plateau daje lepszy wynik końcowy niż samo TD3 przy tym samym budżecie 1M kroków?

## Ramiona i zakres

- **td3:** obecny `rl_only`, bez zmian.
- **td3-swap:** td3 + faza podmiany (`algorithm.swap.enabled=true`).
- Walker2d-v5, Hopper-v5 × seedy 0-4 × 2 ramiona = 20 runów po 1M kroków. Oba ramiona świeże, pod jednym tagiem.

## Wyzwalacz (tylko przeszłość)

- Sprawdzany po każdym `perf/eval_rl` (co 5k kroków).
- t ≥ 200k. Nachylenie MNK eval_rl z ewaluacji w (t − 100k, t] ≤ 0.5% |średniej z tego okna| na 100k kroków.
- Karencja 100k kroków od końca poprzedniej fazy (zaakceptowanej lub nie). Bez warunku na krytyka.
- Faza nie startuje, gdy t + 88k > 1M (88k = maksymalny koszt fazy), żeby run kończył się na 1M łącznie.

## Faza (G-d z gibo-pilot-v1)

- Losowa ortonormalna podprzestrzeń d = 16 głowy aktora, nowa w każdej fazie. Mapa osi do 1× odniesienia dist_H.
- GIBO 8 rund × 4 zapytania, epizody parowane z aktorem. Line search 0.25/0.5/1×. Krok zastępczy przy ujemnej krzywiźnie jak w pilocie, oznaczany w logu.
- Weryfikacja sekwencyjna 5 → 20 epizodów, akceptacja tylko przy dolnej granicy 95% CI > 0.
- Wszystkie epizody fazy wliczane do budżetu (prawdziwa długość) i zapisywane w buforze. Kroki fazy nie dają aktualizacji TD3.

## Po akceptacji

- Głowa aktora online i docelowego ← wybrana głowa. 5k aktualizacji samego krytyka przy zamrożonym aktorze, potem normalny TD3.
- Cofnięcie: średnia 3 eval_rl po podmianie (pierwsza na końcu fazy) < średnia 3 eval_rl przed fazą → przywróć aktora sprzed fazy (embedding + głowa, online i docelowy, stan Adama). Krytyk zostaje.

## Miary (paper-v2, bez nowych metryk wyniku)

- Wynik końcowy: średnia eval_rl z ostatnich 10% (≥ 900k). AUC: średnia trapezowa eval_rl po env_steps.
- Per env: IQM ze stratyfikowanym bootstrapem bez parowania, AUC, dolny kwartyl. Łącznie: P(td3-swap > td3) (średnia po envach), dolny kwartyl wyniku / IQM td3 w danym env.
- Logi faz (opisowo): t, nachylenie, koszt w krokach, akceptacja, CI weryfikacji, kroki zastępcze, cofnięcie. Per run: liczba faz, odsetek kroków w fazach.

## Reguła decyzji (zamknięta)

- **SYGNAŁ:** łącznie P(td3-swap > td3) ≥ 0.65 i IQM td3-swap ≥ td3 w obu envach, oraz w żadnym env IQM nie gorszy o > 5%.
- Brak sygnału → kierunek metody zamknięty, bez kolejnych wariantów.
- Sygnał → nic nie twierdzimy, następny krok 10 seedów z tą samą regułą.
- Osobno: czy wyzwalacz w ogóle odpala. Jeśli w większości runów 0 faz, raport mówi to wprost.
- Uwaga: warunek 5% wynika już z IQM ≥ w obu envach. Zapisany jak w zadaniu.

## Symulacja wyzwalacza przed biegiem

- Reguła wyzwalacza puszczona na istniejących krzywych td3 (10 seedów na env, `paper-v2-stage2`, Walker2d s0-1 z `critic-degeneracy-v1`), z karencją 130k (≈ 30k fazy + 100k).
- Odpala w każdym runie: Walker2d 3-6 razy (średnio 4.3), Hopper 4-6 razy (średnio 4.8), zwykle zaraz po karencji. Szum eval_rl (5 epizodów) daje nachylenia ≤ 0.5% w większości okien po 200k.
- Wniosek przed biegiem: faza zużyje prawdopodobnie ok. 10-20% budżetu td3-swap. To koszt metody, nie błąd.

## Odstępstwa od pilota (ustalone przed biegiem)

- **σ dla Hopper:** brak kalibracji σ 5% w critic-degeneracy-v1 (tylko HalfCheetah, Walker2d, Swimmer). Hopper bierze σ Walker2d (0.0601). Kalibracja online kosztowałaby epizody, a σ wyznacza tylko skalę kroku (odniesienie dist_H).
- **Stany do H_a i dist_H:** 2000 różnych stanów z najnowszych 20k przejść bufora zamiast 20 epizodów aktora (te kosztowałyby do 20k kroków na fazę). Przejścia z ostatnich ok. 20 epizodów z szumem eksploracji 0.1.
- **Odniesienie dist_H:** 40 nowych zaburzeń σ aktora w każdej fazie (w pilocie: zaburzenia z critic-degeneracy-v1). Bez epizodów.
- **Seedy resetu:** świeże w każdej fazie (offset per seed runu i numer fazy). W pilocie jedna decyzja na komórkę.
- **Urządzenie:** faza w jobie treningowym na lem-gpu (pilot na CPU). Batche dopełnione do 16 wierszy jak w pilocie.
- **Budżet:** dodatkowy warunek t + 88k ≤ 1M (wyżej). Koszt fazy nie jest wielokrotnością 1000, więc ostatni epizod RL td3-swap może skończyć run do 999 kroków po 1M (td3 kończy dokładnie na 1M). Pętla treningowa bez zmian.
- **Cofnięcie:** „3 kolejne eval_rl” = 3 kolejne logowane `perf/eval_rl` po podmianie. Pierwsza wypada na końcu fazy, gdy faza przekracza granicę 5k (prawie zawsze). Podmiana bliżej końca niż 3 ewaluacje: bez decyzji, raport oznacza ją jako nierozstrzygniętą.

## Implementacja (przed biegiem)

- `src/algos/plateau_swap.py` (`algorithm.swap.*`), faza woła funkcje `offline.gibo` (G-d: `new_arm`, `run_rounds`, `line_search`, `verify`, `cost`) i `offline.curvature_es.curvature_on` bez zmian logiki. Hiperparametry fazy czytane z `gibo-pilot-v1/experiment.toml`, nadpisane tylko `seed_base` (świeże seedy na fazę).
- Ścieżka td3 (`swap.enabled=false`) bez zmian: dodatkowy kod nie wykonuje się.
- Smoke lokalnie (CPU, Walker2d, H = 100, progi zmniejszone tylko w smoke): faza przy 12k, koszt 5.8k kroków, akceptacja, 100 aktualizacji krytyka, decyzja o cofnięciu po 2 ewaluacjach, blokada t + max_phase_steps ≤ budżet działa. `restore_actor` sprawdzony osobno (aktor i Adam ze stanu sprzed fazy, krytyk bez zmian).

## Uruchomienia

| krok | runy | stan |
|---|---|---|
| smoke GPU, job 6348789_10 (207afce) | td3-swap Walker2d s0, 120k kroków, `swap.min_step=30000`, `swap.rel_slope=1000` (tylko smoke), tag `plateau-swap-smoke` | COMPLETED 16:48. Faza przy 30k: koszt 12 441 kroków, etap 2, odrzucona (CI [-83.6, 20.6]), bez błędów. `WANDB_MODE=disabled` nie zadziałał, run zsynchronizowany online pod tagiem smoke (poza analizą) |
| bieg, job 6351045 (207afce) | Walker2d-v5, td3 + td3-swap, s0-4, `--array=0-4,10-14`, 12 h, 1× H100, `-c 4` | COMPLETED 10/10 |
| bieg, job 6351046 (207afce) | Hopper-v5, td3 + td3-swap, s0-4, `--array=0-4,10-14`, 12 h, 1× H100, `-c 4` | COMPLETED 10/10 |

## Wyniki (2026-11-18)

- **Raport:** https://claude.ai/code/artifact/37c0e809-dfff-48bb-adb7-937e1faf6b1c (`plateau_swap.html`, dane w `plateau_swap.json`, `python -m analysis.plateau_swap`).
- **Werdykt: BRAK SYGNAŁU.** Kierunek metody zamknięty zgodnie z regułą, bez kolejnych wariantów. Nie spełnia żadnego z trzech warunków.
  - Łącznie P(td3-swap > td3) 0.500 [0.26, 0.74] (próg 0.65).
  - Walker2d: IQM td3-swap 3828.6 [3064, 4194] vs td3 4096.9 [3480, 5239], ×0.935, P 0.32. Spadek > 5%.
  - Hopper: IQM 3278.8 [3141, 3414] vs 3050.4 [1674, 3348], ×1.075, P 0.68. Szerokie CI td3 przez s3 (wynik 1163).
- **Opisowo (poza regułą):**
  - AUC: Walker2d 2304 vs 3094, Hopper 2523 vs 2048.
  - Dolny kwartyl: Walker2d 3503 vs 3600, Hopper 3158 vs 2650. Łącznie Q1 / IQM td3: 0.967 vs 0.871.
- **Wyzwalacz odpala w 10/10 runów td3-swap.**
  - 33 fazy (1-4 na run), 24 akceptacje, 6 z nich cofniętych, 0 nierozstrzygniętych, 1 krok zastępczy.
  - Koszt fazy 13-87k kroków, odsetek budżetu w fazach 1.3-28.5% na run (mediana 16.5%).
  - Na Hopper część akceptacji z dużym zyskiem weryfikacji wypada przy silnie ujemnym nachyleniu (dołki po załamaniu eval_rl). Hopper s3 przy 651k: akceptacja na etapie 2 mimo y line search −164, potem cofnięta.
- **Konfiguracja:** knoby `swap` wszystkich 20 runów zgodne z prerejestracją, wszystkie na commicie 207afce.
- **Koszt:** 35.6 GPU-h, 142.6 CPU-h (ze smoke).

## Odstępstwa i zastrzeżenia (po biegu)

- **Niedeterminizm GPU:** ramiona nie są parami po seedzie. Średnia eval_rl w (0, 200k], gdzie faza nie może wystartować, już się różni: Walker2d td3 479/1629/512/860/1428 vs td3-swap 349/1104/1912/378/698, Hopper 947/391/480/367/727 vs 437/674/751/899/653. Część różnic per env to szczęście seedów, nie faza. Kontrola opisowa, dodana po biegu, poza regułą.
- **Smoke w wandb:** run `td3-swap_Walker2d-v5_s0` z tagiem `plateau-swap-smoke` (id j4xl9hel) jest online, poza analizą. Nie usunięty.
