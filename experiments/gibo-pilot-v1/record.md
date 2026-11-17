# gibo-pilot-v1: czy decyzja GIBO na głowie aktora bije TD3, offline

- **Status:** prerejestracja 2026-11-17, przed bramką, dopasowaniem GP i rolloutami.
- **Charakter:** pilot offline, lokalnie na CPU (bez grantu). Wersja online (wykrywanie plateau, przekazanie pałeczki TD3) po wyniku pilota, w osobnej sesji.
- **Code:** branch `gibo-pilot-v1`.
- **Spec:** `experiment.toml`. Checkpointy, σ, klucz zaburzeń: `critic-degeneracy-v1`. R_i (g_ES oracle): `local-search-v1`. Stany, współrzędne u, dist_H: `critic-curvature-es-v1`.
- **Bez decyzji o metodzie:** wynik wraca do uzgodnienia.

## Pytanie

- Czy jedna decyzja GIBO (lokalne szukanie gradientu z GP w 16-wymiarowej podprzestrzeni, epizody parowane, weryfikacja przed podmianą) daje większy zysk niż TD3 przez ten sam budżet kroków?

## Dane

- Komórka = env × seed × checkpoint (300k, 600k, 1M): 18 komórek, po 1 decyzji na komórkę i ramię.
- Walker2d (6, hipotezy), HalfCheetah (6, kontrola), Swimmer (6, opisowo).

## Podprzestrzeń

- Współrzędne u jak w critic-curvature-es-v1: δθ = S·δu, S = std liścia aktora.
- **b:** top-16 wektorów własnych M = S·(średnia_s J_sᵀJ_s)·S na 2000 stanach on-policy (J = ∂akcja/∂wagi głowy). Kierunki, które najmocniej zmieniają akcje.
- **d (kontrola):** losowe bazy ortonormalne (QR z gaussa). 20 do bramki, losowanie 0 dla ramienia G-d.
- **Wybielenie:** dla każdego wektora v_k skala c_k z bisekcji, tak by krok x_k = 1 dawał dist_H = 1× odniesienia (średnia dist_H 40 zaburzeń σ 5%).

## Bramka G0 (bez rolloutów)

- Energia g_u (σ 5%, oracle z 30 epizodów) w podprzestrzeni: |Vᵀg_u|² / |g_u|². Dla losowej oczekiwane d/P ≈ 1%.
- Przechodzi: energia w b ≥ 3× średniej z 20 losowych w ≥ 4/6 Walker2d.
- Nie przechodzi → bez rolloutów, notatka analityczna.

## GIBO (ramię G-b, kontrola G-d)

- Centrum stałe w aktorze. 8 rund × 4 zapytania, 1 epizod na zapytanie.
- Obserwacja: y = R(wariant) − R(aktor) na tym samym seedzie resetu. Runda r: wszystkie zapytania wszystkich ramion i aktor na seedzie 9 000 000 + r.
- GP: średnia a priori 0, dokładny punkt (0, 0), kernel RBF ARD, długości skali w [0.5, 2] (1 = odniesienie σ 5%), hiperparametry z maksymalnej wiarygodności po każdej rundzie (runda 1 na wartościach początkowych).
- Akwizycja: zachłannie, każde zapytanie minimalizuje ślad kowariancji posterioru ∇f(0), w kuli |x| ≤ 1.

## ES-b (kontrola bez GP)

- Te same rundy i seedy, 2 pary antytetyczne na rundę w b, x = ±ε, ε ~ N(0, I/16).

## Krok i weryfikacja

- Kierunek: G = średnia posterioru ∇f(0), ES-b = estymator ES. Mapowany do θ, norma 1.
- Line search: 0.25 / 0.5 / 1× odniesienia dist_H (bisekcja), 2 parowane epizody na głowę, wybór najwyższej średniej y.
- Weryfikacja sekwencyjna na świeżych seedach: po 5 epizodach CI > 0 → akceptacja, CI < 0 → odrzucenie, inaczej do 20 i akceptacja tylko gdy dolna granica > 0.
- Budżet: ≤ 58 epizodów wariantów + ≤ 30 epizodów aktora na decyzję. Koszt = kroki wszystkich tych epizodów (każde ramię liczone w całości).

## Ocena

- Jeden batch na checkpoint, 30 świeżych seedów 9 300 000-029, wspólnych dla głów: aktor, 9 głów line search, krok ES oracle 1× i jego rzut na b 1×.
- Wszystkie batche ≥ 16 wierszy (dopełnione kopiami aktora): jedna ścieżka numeryczna XLA.
- Zysk decyzji = prawdziwy zysk wybranej głowy, jeśli zaakceptowana, inaczej 0.

## Hipotezy (zamknięte)

- **G0:** jak wyżej.
- **L:** zysk decyzji G-b > 0 i > przyrostu TD3 dla jej kosztu w ≥ 4/6 Walker2d. Przyrost TD3 = nachylenie OLS perf/eval_rl w ±100k × koszt.
- **HC:** mediana po 6 komórkach HalfCheetah zysku decyzji G-b / |R_aktor| ≥ −2%.
- G-d i ES-b: porównania opisowe (czy pomaga GP, czy podprzestrzeń). Swimmer opisowo.
- **Reguła stopu:** G0 nie przechodzi → stop przed rolloutami. L nie przechodzi → kierunek zamknięty.
- Uwaga: weryfikacja sekwencyjna (dwa spojrzenia) lekko podnosi α ponad 5%.

## Raport

- Per komórka: energia w b i d, akceptacje, zyski decyzji i kandydatów, koszt, przyrost TD3, cos kierunku z rzutem g_u, krok ES oracle jako referencja, werdykty G0, L, HC.
