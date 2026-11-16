# local-search-v1 / krok 1: czy przeszukiwanie lokalne wokół aktora TD3 się opłaca

- **Status:** prerejestracja 2026-11-16, przed rolloutami i przed policzeniem nachyleń TD3.
- **Charakter:** offline, bez nowych checkpointów i bez treningu, lokalnie na CPU (bez grantu).
- **Code:** branch `local-search-v1` (z `critic-degeneracy-v1`).
- **Spec:** `experiment.toml` (`[analysis]`, `[rule]`). Źródło polityk i krzywych: `critic-degeneracy-v1`.
- **Bez decyzji o metodzie:** wynik wraca do uzgodnienia.

## Pytanie

- Czy akceptowanie zaburzeń aktora ocenionych na kilku epizodach (parowanych z aktorem) daje więcej zwrotu na krok środowiska niż dalszy trening TD3 w tej samej fazie?

## Dane

- 36 komórek v1 (3 envy × seedy 0-1 × checkpointy 300k/600k/1M × 2 poziomy σ), 40 zaburzeń na komórkę + aktor.
- Zapisane w v1: zwroty per epizod dla 10 seedów prawdy (wszystkie zaburzenia, GPU) i 20 seedów walidacji (tylko 404 z 1458 polityk, kandydaci walidacji), bez długości epizodów.
- Dlatego rollouty od nowa na CPU: wszystkie 81 głów na checkpoint × 30 seedów resetu (1 000 000-1 000 009 i 2 000 000-2 000 019), zwrot i długość per epizod. Ok. 44k epizodów.
  - Zaburzenia odtworzone z `[offline].key` v1, sprawdzenie dist_L2 na checkpointach 300k (rtol 5e-2, jak w walidacji v1).
  - Wszystkie 3 envy, także HalfCheetah: reguła wymaga odsetka fałszywych akceptacji w HalfCheetah na 30 epizodach. Odstępstwo od polecenia („tylko Walker2d i Swimmer”), koszt kilka minut CPU.
  - Kontrola zgodności: zwroty kandydatów na seedach walidacji = zapisane w v1 (ten sam kod na CPU), średnie z 10 seedów prawdy vs v1 (GPU, TF32): korelacja.
- Długości: HalfCheetah i Swimmer zawsze 1000 kroków, Walker2d kończy się upadkiem, koszt liczony z rzeczywistych długości.
- Krzywe TD3: `perf/eval_rl` (co 5000 kroków, 5 epizodów) z runów td3 v1, te same seedy.

## Prawda

- Zwrot = średnia z 30 epizodów. Prawdziwy zysk = średnia różnica parowana R_pert − R_aktor.
- „Lepszy” = 95% przedział t różnicy parowanej (29 df) w całości nad zerem.

## Procedury (1000 powtórzeń losowania epizodów, seed 20261117)

- **1. Pojedyncza decyzja** (krok SA / hill climbing), k ∈ {1, 3, 5}:
  - per zaburzenie k seedów bez zwracania z 30, parowanie z aktorem, akceptacja gdy średnia różnica > 0,
  - miary: P(akceptacja), P(słuszna decyzja) (akceptacja ⇔ lepszy), P(fałszywa akceptacja) (akceptacja i nie lepszy),
  - oczekiwany zysk jednej iteracji = średnia po zaburzeniach P(akceptacja) × prawdziwy zysk, koszt = kroki 2k epizodów.
- **2. Najlepszy z N**, N ∈ {10, 20, 40}, k ∈ {1, 3}:
  - N z 40 zaburzeń komórki bez zwracania, k epizodów każdy, zwycięzca = najwyższa średnia różnica do aktora,
  - akceptacja, gdy dolna granica 95% CI > 0. CI = dokładny bootstrap percentylowy (wszystkie k^k losowania) po k epizodach zwycięzcy. Dla k = 1 to po prostu różnica > 0, dla k = 3 przedział jest wąski (antykonserwatywny), co mierzy fałszywa akceptacja.
  - parowany: te same k seedów dla N kandydatów i aktora. Bez parowania (porównanie): osobne seedy dla każdego kandydata i aktora, bootstrap dwóch prób.
  - miary: P(akceptacja), P(zaakceptowany naprawdę lepszy), P(fałszywa akceptacja), oczekiwany zysk = średnia po powtórzeniach [akceptacja] × prawdziwy zysk zwycięzcy, koszt = kroki (N + 1)·k epizodów.
- Zysk na 100k kroków = oczekiwany zysk / średni koszt × 100k. Dla procedury 1 to ekstrapolacja: zakłada, że kolejne iteracje dają podobny zysk.
- Procedura = punkt × k × N × parowanie × poziom σ (30 procedur).

## Opłacalność

- Przyrost TD3 na 100k kroków: nachylenie MNK `perf/eval_rl` względem `env_steps` w oknie [faza − 100k, faza + 100k] przyciętym do [0, 1M], per seed. Fazy: 300k, 600k, 1M.
- Tabela per env × faza: najlepszy zysk na 100k z procedur 1-2 vs przyrost TD3 na 100k.

## Reguła (zamknięta)

- **Przeszukiwanie lokalne się opłaca**, jeśli istnieje procedura, która w Walker2d lub Swimmer w ≥ 1 fazie ma zysk na 100k kroków > 0 i > przyrostu TD3 na 100k kroków dla obu seedów (2/2), a ta sama procedura ma w HalfCheetah odsetek fałszywych akceptacji ≤ 5% (średnia po 6 komórkach przy tym poziomie σ).
- Doprecyzowanie „> 0”: przy ujemnym nachyleniu TD3 procedura, która nic nie akceptuje, nie może przejść.
- Wielokrotność: 30 procedur × 2 envy × 3 fazy. Warunek 2/2 seedów i fałszywe akceptacje w HalfCheetah to jedyna ochrona, wynik „opłaca się” traktować jako wskazanie do sprawdzenia, nie dowód.

## Raport

- Tabele z punktów 1-3, wykres oczekiwany zysk vs koszt w krokach z linią przyrostu TD3 (per env × faza).
- `python -m analysis.local_search ../experiments/local-search-v1/experiment.toml` (z `src/`) → `local_search.{json,html}`.
