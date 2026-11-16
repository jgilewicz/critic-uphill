# critic-curvature-es-v1: krok ES (opcjonalnie z hesjanem krytyka) vs najlepsze zaburzenie, offline

- **Status:** prerejestracja 2026-11-16, przed liczeniem stanów, hesjanów i kroków.
- **Charakter:** offline, bez treningu, lokalnie na CPU (bez grantu).
- **Code:** branch `critic-curvature-es-v1` (z `td3-sa-pilot`).
- **Spec:** `experiment.toml` (`[states]`, `[steps]`, `[evaluation]`, `[compare]`, `[[hypotheses]]`). Checkpointy, σ, klucz zaburzeń i krzywe TD3: `critic-degeneracy-v1`. R_i: `local-search-v1`.
- **Bez decyzji o metodzie:** wynik wraca do uzgodnienia.

## Pytanie

- Czy krok ES zbudowany z 40 zaburzeń (gradient z prawdziwych zwrotów) daje więcej niż najlepsze pojedyncze zaburzenie, czy hesjan krytyka TD3 go poprawia i czy tani krok (k = 1, 3) bije TD3 na tym samym koszcie w plateau?

## Dane

- Komórka = env × seed × checkpoint (300k, 600k, 1M) × poziom σ (5%, 20%): 36 komórek, 24 w Walker2d + Swimmer.
- R_i: 81 głów × 30 epizodów z local-search-v1 (seedy 1 000 000-009 i 2 000 000-019). δθ_i odtworzone z klucza critic-degeneracy-v1 (te same głowy co w local-search-v1).
- Głowa aktora = Linear(256 → A) + tanh: P = 1542 (Walker2d, HalfCheetah), 514 (Swimmer). H_θ liczone jawnie (P × P).

## Stany

- 2000 stanów on-policy: 20 epizodów aktora (seedy 6 000 000-019), wszystkie stany żywe zebrane, 2000 losowanych bez zwracania.
- Odstępstwo od v1 (2000 stanów z ostatnich 100k przejść bufora): `buffer.npz` jest lokalnie tylko dla 300k. Dlatego odniesienie dist_H małych zaburzeń przeliczam na tych samych stanach co kroki, zamiast brać t2_TD3 z v1.

## Kierunki

- Współrzędne wybielone u: δθ = σ·S·ε per liść, S = std liścia aktora (jak `perturb`).
- g_u = Σ_i (R_i − R_aktor)·ε_i / (N·σ), N = 40 jednego poziomu σ. d_ES = S·g_u = Σ_i (R_i − R_aktor)·δθ_i / (N·σ²), czyli wzór ze specyfikacji.
- H_θ = średnia_s J_sᵀ H_a,s J_s (Gauss-Newton). H_a = hesjan min(Q1, Q2) krytyka TD3 po akcji w a = π(s). J_s = ∂a/∂θ głowy.
- d_N = S·(−H_u + λI)⁻¹·g_u, H_u = S·H_θ·S. λ = max(0, −μ_min) + 1e-3·μ_max, μ = wartości własne −H_u (macierz dodatnio określona, kondycja ≤ ok. 1000).
- Interpretacja: hesjan w tych samych współrzędnych co szum zaburzeń. W surowych θ d_N mieszałoby prekondycjonowanie S² z g z Newtonem.
- Źródła R_i:
  - oracle: średnia z 30 epizodów,
  - k = 1 i k = 3: k z 30 epizodów bez zwracania, te same k seedów dla aktora i 40 zaburzeń (parowanie), 10 niezależnych losowań na k.

## Kroki

- dist_H(θ') = średnia_s −½·δaᵀH_aδa, δa = π_θ'(s) − π_θ(s) na 2000 stanach (jak −t2_TD3 w v1).
- Odniesienie = średnia dist_H 40 zaburzeń σ 5% komórki. α z bisekcji na dokładnym dist_H(θ + α·d) = 0.5×, 1×, 2× odniesienia (tolerancja względna 1e-3).
- Koszt kroku k = rzeczywiste kroki jego 41·k epizodów (aktor + 40 zaburzeń). Oracle to nie procedura, koszt tylko raportowany.

## Ocena

- Jeden batch na checkpoint, 30 świeżych seedów (5 000 000-029), wspólnych dla wszystkich głów: aktor, najlepsze pojedyncze zaburzenie per σ (najwyższa średnia z 30 epizodów local-search-v1, ponownie ocenione), 252 kroki (2 σ × 2 kierunki × 3 krotności × (oracle + 2 k × 10 losowań)). Razem 255 głów.
- Zysk = średnia po epizodach R(głowa) − R(aktor), parowana.
- Przyrost TD3 na koszt c = nachylenie OLS `perf/eval_rl` w [t − 100k, t + 100k] przyciętym do [0, 1M] (jak local-search-v1) × c, ten sam run.

## Hipotezy (zamknięte)

- **H6:** oracle g, d_ES, najlepsza z 3 krotności: zysk > zysku najlepszego pojedynczego zaburzenia tego samego σ w ≥ 16 z 24 komórek Walker2d + Swimmer.
- **H7:** oracle g: zysk d_N > zysku d_ES przy ≥ 2 z 3 krotności dist_H w ≥ 16 z 24 komórek.
- **H8:** istnieje procedura (k, kierunek, krotność) i poziom σ z średnim zyskiem po 10 losowaniach > 0 i > przyrostu TD3 na jej średni koszt w Walker2d 600k s0 i w ≥ 4 z 6 checkpointów Swimmer (300k, 600k, 1M × seedy 0, 1) przy tym σ.
- **HC (kontrola):** mediana po 12 komórkach HalfCheetah najlepszego zysku kroku oracle (po kierunkach i krotnościach) ≤ 2% zwrotu aktora.
- Wielokrotność i stronniczość:
  - H6 wybiera najlepszą krotność na tych samych 30 epizodach, na których jest oceniana, a najlepsze pojedyncze zaburzenie było wybrane na innych. To faworyzuje krok ES.
  - H8 to maksimum po 24 procedurach × σ. Wynik dodatni to wskazanie, nie dowód.

## Raport

- Per komórka: zyski kroków (oracle, k = 1, 3) vs najlepsze zaburzenie, λ i kondycja, α, koszt, przyrost TD3, werdykty H6-H8 i HC.
- `python -m analysis.curvature_es ../experiments/critic-curvature-es-v1/experiment.toml` (z `src/`) → `curvature_es.{json,html}`.
