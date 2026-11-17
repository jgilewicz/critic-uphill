# plateau-swap-online-v1: czy podmiana aktora w plateau (G-d) pomaga TD3 online

- **Status:** prerejestracja 2026-11-17, przed jakimkolwiek runem.
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
