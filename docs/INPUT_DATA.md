# INPUT_DATA.md

Dokument opisuje wszystkie wejściowe kolumny i cechy używane w pipeline'ie przetwarzania danych.

---

## 📥 Wejściowe kolumny ze StatsBomb Events

Pipeline przetwarza dane w formacie **SPADL** (Soccer Player Action Description Language), który jest standaryzowaną reprezentacją eventów StatsBomb.

### Kolumny bezpośrednio z SPADL Actions

Po konwersji StatsBomb Events → SPADL Actions, dostępne są następujące kolumny:

| Kolumna | Typ | Opis | Zakres/Format |
|---------|-----|------|---------------|
| `game_id` | int | Identyfikator meczu | Unikalny ID |
| `original_event_id` | str | ID oryginalnego eventu StatsBomb | UUID |
| `period_id` | int | Część meczu | 1-5 (1,2 = połowy, 3,4 = dogrywka, 5 = karne) |
| `time_seconds` | float | Czas od początku okresu (w sekundach) | 0.0 - ~3600.0 |
| `team_id` | int | ID zespołu wykonującego akcję | Unikalny ID |
| `player_id` | int | ID zawodnika wykonującego akcję | Unikalny ID |
| `start_x` | float | Pozycja X początku akcji | 0.0 - 105.0 (metry) |
| `start_y` | float | Pozycja Y początku akcji | 0.0 - 68.0 (metry) |
| `end_x` | float | Pozycja X końca akcji | 0.0 - 105.0 (metry) |
| `end_y` | float | Pozycja Y końca akcji | 0.0 - 68.0 (metry) |
| `type_id` | int | Typ akcji | 0-22 (23 kategorie) |
| `result_id` | int | Rezultat akcji | 0-5 (6 kategorii) |
| `bodypart_id` | int | Część ciała użyta do akcji | 0-5 (6 kategorii) |
| `possession` | int | Numer posiadania piłki | Sekwencyjny numer |
| `possession_team_id` | int | ID zespołu posiadającego piłkę | Unikalny ID |

### Kolumny wzbogacone (enriched)

Po ekstrakcji posiadań, dodawane są następujące kolumny:

| Kolumna | Typ | Opis | Źródło |
|---------|-----|------|--------|
| `xG` | float | Expected Goals (prawdopodobieństwo gola) | Obliczone przez socceraction |
| `xT` | float | Expected Threat (zagrożenie) | Obliczone przez model xT |
| `under_pressure` | bool | Czy akcja wykonana pod presją przeciwnika | StatsBomb Events |
| `counterpress` | bool | Czy była próba kontrapresingu | StatsBomb Events |

---

## 🔧 Feature Engineering - Cechy obliczane

W funkcji `_update_action()` z powyższych kolumn obliczane są dodatkowe cechy:

### Cechy geometryczne

```python
dx = end_x - start_x                    # Zmiana pozycji X
dy = end_y - start_y                    # Zmiana pozycji Y
distance = sqrt(dx² + dy²)              # Dystans akcji (metry)
angle = arctan2(dy, dx)                 # Kąt akcji (radiany)
```

### Cechy czasowe

```python
time_diff = time_seconds.diff()         # Różnica czasu między akcjami (sekundy)
```

### Cechy kontekstowe

```python
opposite_action = (team_id != possession_team_id)  # Czy akcja przeciwnika (bool)
```

---

## 📊 Normalizacja - 46 wymiarów

W funkcji `_normalize_features()` wszystkie cechy są normalizowane i przekształcane do **46 wymiarów**:

### 1. Spatial Features (4 wymiary)

**Źródło:** `start_x`, `start_y`, `end_x`, `end_y`

**Transformacja:** Normalizacja do [0, 1]

```python
start_x_norm = start_x / 105.0          # [0.0 - 1.0]
start_y_norm = start_y / 68.0           # [0.0 - 1.0]
end_x_norm = end_x / 105.0              # [0.0 - 1.0]
end_y_norm = end_y / 68.0               # [0.0 - 1.0]
```

**Znaczenie:**
- Znormalizowane współrzędne pozycji na boisku
- Niezależne od rzeczywistych wymiarów boiska

### 2. Geometric Features (3 wymiary)

**Źródło:** `distance`, `angle` (obliczone)

**Transformacja:** Normalizacja dystansu + trygonometria kąta

```python
distance_norm = distance / 105.0        # [0.0 - ~1.4] (przekątna boiska)
angle_sin = sin(angle)                  # [-1.0 - 1.0]
angle_cos = cos(angle)                  # [-1.0 - 1.0]
```

**Znaczenie:**
- `distance_norm`: jak daleko przemieszcza się piłka
- `sin/cos`: kierunek ruchu piłki (cykliczny, bez discontinuity)

### 3. Temporal Features (1 wymiar)

**Źródło:** `time_diff` (obliczone)

**Transformacja:** Capping + normalizacja

```python
time_diff_norm = min(time_diff / 10.0, 1.0)  # [0.0 - 1.0], cap at 10s
```

**Znaczenie:**
- Tempo gry (krótkie/długie przerwy między akcjami)
- Capping zapobiega outlierom (długie przerwy)

### 4. Contextual Features (3 wymiary)

**Źródło:** `under_pressure`, `counterpress`, `opposite_action`

**Transformacja:** Bool → Float

```python
under_pressure_float = float(under_pressure)      # {0.0, 1.0}
counterpress_float = float(counterpress)          # {0.0, 1.0}
opposite_action_float = float(opposite_action)    # {0.0, 1.0}
```

**Znaczenie:**
- Kontekst taktyczny i defensywny akcji
- Binary features (tak/nie)

### 5. Categorical Features (35 wymiarów)

#### 5.1 Type ID (23 wymiary)

**Źródło:** `type_id` (0-22)

**Transformacja:** One-hot encoding

```python
type_onehot = [0,0,...,1,...,0]  # 23 dimensions, one bit = 1
```

**Kategorie akcji (SPADL):**
```
0: pass
1: cross
2: throw_in
3: freekick_crossed
4: freekick_short
5: corner_crossed
6: corner_short
7: take_on
8: foul
9: tackle
10: interception
11: shot
12: shot_penalty
13: shot_freekick
14: keeper_save
15: keeper_claim
16: keeper_punch
17: keeper_pick_up
18: clearance
19: bad_touch
20: non_action
21: dribble
22: goalkick
```

#### 5.2 Result ID (6 wymiarów)

**Źródło:** `result_id` (0-5)

**Transformacja:** One-hot encoding

```python
result_onehot = [0,0,...,1,...,0]  # 6 dimensions
```

**Kategorie rezultatu:**
```
0: fail
1: success
2: offside
3: owngoal
4: yellow_card
5: red_card
```

#### 5.3 Bodypart ID (6 wymiarów)

**Źródło:** `bodypart_id` (0-5)

**Transformacja:** One-hot encoding

```python
bodypart_onehot = [0,0,...,1,...,0]  # 6 dimensions
```

**Kategorie części ciała:**
```
0: foot
1: foot_left
2: foot_right
3: head
4: other
5: head/other
```

---

## 📋 Podsumowanie przepływu danych

### Wejście: SPADL Actions DataFrame

```python
DataFrame({
    'start_x': 52.5,
    'start_y': 34.0,
    'end_x': 60.1,
    'end_y': 30.0,
    'time_seconds': 125.5,
    'type_id': 0,        # pass
    'result_id': 1,      # success
    'bodypart_id': 2,    # foot_right
    'under_pressure': True,
    'counterpress': False,
    'team_id': 214,
    'possession_team_id': 214,
    'xG': 0.0,
    'xT': 0.025
})
```

### Feature Engineering

```python
# Obliczane cechy:
dx = 60.1 - 52.5 = 7.6
dy = 30.0 - 34.0 = -4.0
distance = sqrt(7.6² + 4.0²) = 8.59
angle = arctan2(-4.0, 7.6) = -0.481
time_diff = 2.3  # różnica z poprzednią akcją
opposite_action = False
```

### Normalizacja → 46-wymiarowy wektor

```python
[
    # Spatial (4)
    0.500,  # start_x_norm = 52.5/105
    0.500,  # start_y_norm = 34.0/68
    0.572,  # end_x_norm = 60.1/105
    0.441,  # end_y_norm = 30.0/68
    
    # Geometric (3)
    0.082,  # distance_norm = 8.59/105
    -0.462, # sin(angle) = sin(-0.481)
    0.887,  # cos(angle) = cos(-0.481)
    
    # Temporal (1)
    0.230,  # time_diff_norm = 2.3/10
    
    # Contextual (3)
    1.0,    # under_pressure
    0.0,    # counterpress
    0.0,    # opposite_action
    
    # Type ID one-hot (23) - pass = 0
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    
    # Result ID one-hot (6) - success = 1
    0,1,0,0,0,0,
    
    # Bodypart ID one-hot (6) - foot_right = 2
    0,0,1,0,0,0
]
```

---

## 🎯 Konstrukcja etykiet (Funkcja celu)

**Szczegółowy opis konstrukcji etykiet znajduje się w dedykowanym dokumencie:**

📄 **[LABEL_GENERATION.md](./LABEL_GENERATION.md)**

**Krótkie podsumowanie:**
- **Funkcja:** `get_actions_value()` w `src/ml/preprocessing/action_valuation.py`
- **Formuła:** `label = max(sum(xG), xT_final)`
- **Zakres:** [0.0 - 1.0]
- **Interpretacja:** Wartość zagrożenia bramkowego sekwencji
- **Loss function:** Mean Squared Error (MSE)

---

## 🔢 Statystyki wejściowych danych

### Euro 2024 (51 meczów)

| Metryka | Wartość |
|---------|---------|
| Średnia liczba eventów na mecz | ~2000 |
| Średnia liczba akcji SPADL na mecz | ~1600 |
| Średnia liczba posiadań na mecz | ~200 |
| Średnia długość posiadania | 4-6 akcji |
| Min długość posiadania | 1 akcja |
| Max długość posiadania | 30+ akcji |
| **Total akcji (51 meczów)** | **~81,600** |
| **Total posiadań (51 meczów)** | **~10,200** |

### Rozkład typów akcji (type_id)

Najczęstsze typy akcji w Euro 2024:

```
pass (0):           ~45%
dribble (21):       ~15%
tackle (9):         ~10%
shot (11):          ~5%
cross (1):          ~4%
interception (10):  ~3%
clearance (18):     ~3%
inne:               ~15%
```

### Rozkład rezultatów (result_id)

```
success (1):        ~65%
fail (0):           ~30%
offside (2):        ~3%
inne:               ~2%
```

---

## 📦 Format danych w różnych etapach

### 1. SPADL Actions (po konwersji)
```python
DataFrame: (n_actions, 15 kolumn)
Kolumny: game_id, period_id, time_seconds, team_id, player_id,
         start_x, start_y, end_x, end_y, type_id, result_id, 
         bodypart_id, possession, possession_team_id, original_event_id
```

### 2. Enriched Actions (po wzbogaceniu)
```python
DataFrame: (n_actions, 19+ kolumn)
+ xG, xT, under_pressure, counterpress
```

### 3. Updated Actions (po feature engineering)
```python
DataFrame: (n_actions, 25+ kolumn)
+ dx, dy, distance, angle, time_diff, opposite_action
```

### 4. Normalized Features (po normalizacji)
```python
np.ndarray: (n_actions, 46)
dtype: float32
```

### 5. Sequences (po tworzeniu sekwencji)
```python
np.ndarray: (n_sequences, sequence_length=6, 46)
dtype: float32
```

### 6. Final Tensors
```python
X: (n_sequences, 6, 46)  # Input features
y: (n_sequences,)         # Labels (xT values)
```

---

## 🔍 Walidacja danych wejściowych

### Asercje w kodzie

```python
# Spatial features muszą być w [0, 1] po normalizacji
assert np.all(spatial >= 0) and np.all(spatial <= 1)

# Temporal features muszą być w [0, 1]
assert np.all(temporal >= 0) and np.all(temporal <= 1)

# Categorical features muszą być one-hot
assert np.all(np.sum(type_onehot, axis=1) == 1)
assert np.all(np.sum(result_onehot, axis=1) == 1)
assert np.all(np.sum(bodypart_onehot, axis=1) == 1)
```

### Handling missing values

```python
# SPADL gwarantuje brak NaN w podstawowych kolumnach
# Jeśli brakuje bodypart_id, wypełniane jest domyślną wartością (0)
# time_diff dla pierwszej akcji = 0.0
```

---

## 📚 Źródła danych

### StatsBomb Open Data
- **Format:** JSON z zagnieżdżoną strukturą
- **Współrzędne:** (0-120, 0-80) metry
- **Typy eventów:** ~40 różnych typów

### SPADL (Soccer Player Action Description Language)
- **Format:** Płaska struktura DataFrame
- **Współrzędne:** (0-105, 0-68) metry (standaryzowane)
- **Typy akcji:** 23 ujednolicone kategorie
- **Biblioteka:** `socceraction` (Python)

### xT Model (Expected Threat)
- **Model:** Grid-based (12x8 cells)
- **Wartości:** Prawdopodobieństwo zdobycia gola z danej pozycji
- **Źródło:** Trenowany na danych StatsBomb

---

**Koniec dokumentu**

*Dokument utworzony: 8 lutego 2026*
*Plik: docs/INPUT_DATA.md*
