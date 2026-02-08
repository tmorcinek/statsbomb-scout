# DATA_PROCESSING_SCHEMA.md

Dokument przedstawia szczegółowy schemat przetwarzania danych od surowych eventów StatsBomb do sekwencji treningowych dla modelu deep learning.

---

## 📊 Ogólny przegląd pipeline'u

```
StatsBomb Raw Events
         ↓
    SPADL Actions
         ↓
    Possessions
         ↓
  Feature Engineering
         ↓
   Normalization
         ↓
  Sequence Creation
         ↓
   Label Generation
         ↓
    Tensors (X, y)
         ↓
      Model
```

---

## 1️⃣ Etap 1: StatsBomb Raw Events → SPADL Actions

### 1.1 Dane wejściowe: StatsBomb Events

**Format:** JSON ze złożoną strukturą

```json
{
  "id": "12345-abcd-6789",
  "type": {"id": 30, "name": "Pass"},
  "timestamp": "00:00:15.234",
  "location": [45.2, 32.1],
  "pass": {
    "end_location": [52.3, 28.5],
    "recipient": {"id": 5532, "name": "Player X"},
    "body_part": {"id": 40, "name": "Right Foot"}
  },
  "under_pressure": true,
  "possession": 2,
  "possession_team": {"id": 214, "name": "Spain"}
}
```

**Charakterystyka:**
- ~1800-2500 eventów na mecz
- Różne typy: Pass, Shot, Dribble, Carry, Tackle, etc.
- Zagnieżdżona struktura (`pass`, `shot`, `duel`, etc.)
- Współrzędne: (0-120, 0-80) dla boiska 120x80m

### 1.2 Konwersja: `spadl.statsbomb.convert_to_actions()`

**Funkcja:** `src/ml/preprocessing/possessions_extraction.py::extract_actions_from_events()`

```python
def extract_actions_from_events(game, events_df):
    # 1. Konwersja do SPADL
    actions = spadl.statsbomb.convert_to_actions(
        events=events_df,
        home_team_id=game['home_team_id']
    )
    
    # 2. Wzbogacenie o dodatkowe dane
    actions = enrich_actions_with_event_data(actions, events_df)
    
    return actions
```

### 1.3 Dane wyjściowe: SPADL Actions

**Format:** DataFrame z uproszczoną strukturą

| Kolumna | Typ | Opis | Przykład |
|---------|-----|------|----------|
| `game_id` | int | ID meczu | 3943043 |
| `original_event_id` | str | ID oryginalnego eventu | "12345-abcd" |
| `period_id` | int | Część meczu (1-5) | 1 |
| `time_seconds` | float | Czas w sekundach | 125.5 |
| `team_id` | int | ID zespołu | 214 |
| `player_id` | int | ID zawodnika | 5532 |
| `start_x` | float | Pozycja X start (0-105) | 45.2 |
| `start_y` | float | Pozycja Y start (0-68) | 32.1 |
| `end_x` | float | Pozycja X end (0-105) | 52.3 |
| `end_y` | float | Pozycja Y end (0-68) | 28.5 |
| `type_id` | int | Typ akcji (0-22) | 0 (pass) |
| `result_id` | int | Rezultat (0-5) | 1 (success) |
| `bodypart_id` | int | Część ciała (0-5) | 1 (right foot) |

**Różnice vs StatsBomb:**
- Płaska struktura (bez zagnieżdżeń)
- Ujednolicone współrzędne (0-105, 0-68)
- Enumeracje zamiast stringów
- ~1500-2000 akcji na mecz (mniej niż eventów)

---

## 2️⃣ Etap 2: SPADL Actions → Possessions (Posiadania)

### 2.1 Grupowanie po possession_id

**Funkcja:** `src/ml/preprocessing/possessions_extraction.py::extract_possessions()`

```python
def extract_possessions(game, events_df):
    # 1. Konwersja do SPADL
    actions = extract_actions_from_events(game, events_df)
    
    # 2. Normalizacja kierunku (zawsze gra w prawo)
    actions = normalize_pitch(actions, game['home_team_id'])
    
    # 3. Dodanie xG, xT, pressure, etc.
    actions = enrich_actions(actions, events_df)
    
    # 4. Grupowanie po possession_id
    possessions = {}
    for poss_id, group in actions.groupby('possession'):
        possessions[poss_id] = group.reset_index(drop=True)
    
    return possessions
```

### 2.2 Normalizacja kierunku gry

**Funkcja:** `normalize_pitch()`

**Problem:** StatsBomb zapisuje eventy relatywnie do drużyn:
- Drużyna gospodarz: gra w lewo → prawo
- Drużyna gość: gra w prawo → lewo

**Rozwiązanie:** Rotacja współrzędnych tak, aby posiadająca drużyna **zawsze grała w prawo**.

```python
def normalize_pitch(actions, home_team_id):
    for idx, action in actions.iterrows():
        # Jeśli posiadanie zespołu gości
        if action['team_id'] != home_team_id:
            # Odbicie lustrzane
            actions.loc[idx, 'start_x'] = 105 - action['start_x']
            actions.loc[idx, 'start_y'] = 68 - action['start_y']
            actions.loc[idx, 'end_x'] = 105 - action['end_x']
            actions.loc[idx, 'end_y'] = 68 - action['end_y']
    
    return actions
```

**Wizualizacja:**

```
PRZED normalizacją:
┌──────────────────────────────────────┐
│ Team A (home) →                      │  Gra w prawo
│                      ← Team B (away) │  Gra w lewo
└──────────────────────────────────────┘

PO normalizacji:
┌──────────────────────────────────────┐
│ Team A →                             │  Gra w prawo
│ Team B →                             │  RÓWNIEŻ gra w prawo (po rotacji)
└──────────────────────────────────────┘
```

### 2.3 Wzbogacanie akcji

**Funkcja:** `enrich_actions()`

Dodaje dodatkowe kolumny:

```python
def enrich_actions(actions, events_df):
    # 1. xG (Expected Goals)
    actions['xG'] = calculate_xg_values(events_df)
    
    # 2. xT (Expected Threat)
    actions['xT'] = calculate_xt_values(actions, xt_model)
    
    # 3. Under pressure
    actions['under_pressure'] = extract_pressure_info(events_df)
    
    # 4. Counterpress
    actions['counterpress'] = extract_counterpress_info(events_df)
    
    # 5. Opposite action (akcja przeciwnika poprzedzająca)
    actions['opposite_action'] = determine_opposite_actions(actions)
    
    return actions
```

### 2.4 Dane wyjściowe: Dict[int, DataFrame]

**Struktura:**

```python
possessions = {
    1: DataFrame([  # Posiadanie #1 (4 akcje)
        {start_x: 52.5, start_y: 34, type_id: 0, ...},
        {start_x: 60.1, start_y: 30, type_id: 0, ...},
        {start_x: 72.3, start_y: 28, type_id: 2, ...},
        {start_x: 89.5, start_y: 34, type_id: 10, ...}
    ]),
    2: DataFrame([  # Posiadanie #2 (8 akcji)
        ...
    ]),
    ...
}
```

**Statystyki dla Euro 2024:**
- ~150-250 posiadań na mecz
- Średnio 4-6 akcji na posiadanie
- Min: 1 akcja (np. strzał z odbioru)
- Max: 20-30 akcji (długa kombinacja)

---

## 3️⃣ Etap 3: Feature Engineering

### 3.1 Ekstrakcja surowych cech

Pierwszym krokiem jest ekstrakcja surowych wartości z akcji SPADL:

- **Spatial**: Pozycje start/end (w oryginalnych współrzędnych 0-105, 0-68)
- **Geometric**: Obliczanie odległości i kątów
- **Temporal**: Różnice czasu między akcjami
- **Contextual**: Flagi boolowskie (under_pressure, counterpress, opposite_action)
- **Categorical**: Identyfikatory kategorii (type_id, result_id, bodypart_id)

---

## 4️⃣ Etap 4: Normalizacja

### 4.1 Funkcja: `SequencePreprocessor._normalize_features()`

**Kod:** `src/ml/preprocessing/sequence_preprocessor.py`

```python
def _normalize_features(self, actions_df):
    features_list = []
    
    for idx, action in actions_df.iterrows():
        # === 1. SPATIAL (4) ===
        start_x_norm = action['start_x'] / 105.0
        start_y_norm = action['start_y'] / 68.0
        end_x_norm = action['end_x'] / 105.0
        end_y_norm = action['end_y'] / 68.0
        
        # === 2. GEOMETRIC (3) ===
        dx = action['end_x'] - action['start_x']
        dy = action['end_y'] - action['start_y']
        distance = np.sqrt(dx**2 + dy**2) / 105.0  # Normalizacja
        angle = np.arctan2(dy, dx)
        angle_sin = np.sin(angle)
        angle_cos = np.cos(angle)
        
        # === 3. TEMPORAL (1) ===
        if idx == 0:
            time_diff = 0.0
        else:
            time_diff = action['time_seconds'] - actions_df.iloc[idx-1]['time_seconds']
            time_diff = min(time_diff, 10.0) / 10.0  # Cap na 10s
        
        # === 4. CONTEXTUAL (3) ===
        under_pressure = 1.0 if action['under_pressure'] else 0.0
        counterpress = 1.0 if action['counterpress'] else 0.0
        opposite_action = 1.0 if action['opposite_action'] else 0.0
        
        # === 5. CATEGORICAL (35) ===
        # One-hot encoding
        type_id_encoded = np.zeros(23)
        type_id_encoded[action['type_id']] = 1.0
        
        result_id_encoded = np.zeros(6)
        result_id_encoded[action['result_id']] = 1.0
        
        bodypart_id_encoded = np.zeros(6)
        bodypart_id_encoded[action['bodypart_id']] = 1.0
        
        # === KONKATENACJA (46) ===
        feature_vector = np.concatenate([
            [start_x_norm, start_y_norm, end_x_norm, end_y_norm],  # 4
            [distance, angle_sin, angle_cos],                      # 3
            [time_diff],                                           # 1
            [under_pressure, counterpress, opposite_action],       # 3
            type_id_encoded,                                       # 23
            result_id_encoded,                                     # 6
            bodypart_id_encoded                                    # 6
        ])
        
        features_list.append(feature_vector)
    
    return np.array(features_list)  # shape: (num_actions, 46)
```

### 4.2 Podział cech (46 wymiarów)

| Kategoria | Cechy | Liczba | Zakres |
|-----------|-------|--------|--------|
| **Spatial** | start_x, start_y, end_x, end_y | 4 | [0.0, 1.0] |
| **Geometric** | distance, sin(angle), cos(angle) | 3 | [0.0, 1.0], [-1, 1], [-1, 1] |
| **Temporal** | time_diff | 1 | [0.0, 1.0] |
| **Contextual** | under_pressure, counterpress, opposite_action | 3 | {0.0, 1.0} |
| **Categorical** | type_id (one-hot 23), result_id (6), bodypart_id (6) | 35 | {0.0, 1.0} |
| **TOTAL** | | **46** | |

### 4.3 Przykład wektora cech

**Akcja:** Pass (type_id=0), Success (result_id=1), Right Foot (bodypart_id=1)

```python
[
    0.5,  0.4,  0.6,  0.3,     # Spatial: start/end positions
    0.15, 0.8, -0.6,            # Geometric: distance, sin, cos
    0.2,                        # Temporal: 2s od poprzedniej akcji
    1.0, 0.0, 0.0,              # Contextual: under_pressure=True
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  # type_id: Pass
    0,1,0,0,0,0,                # result_id: Success
    0,1,0,0,0,0                 # bodypart_id: Right Foot
]
```

---

## 5️⃣ Etap 5: Tworzenie Sekwencji

### 5.1 Dwa tryby przetwarzania

#### **Tryb TRAINING**

**Cel:** Maksymalizacja liczby przykładów treningowych

**Metoda:** Sliding window

```python
possession = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]  # 10 akcji
sequence_length = 6

# Generuje 5 sekwencji:
sequences = [
    [A1, A2, A3, A4, A5, A6],      # Sekwencja 1
    [A2, A3, A4, A5, A6, A7],      # Sekwencja 2
    [A3, A4, A5, A6, A7, A8],      # Sekwencja 3
    [A4, A5, A6, A7, A8, A9],      # Sekwencja 4
    [A5, A6, A7, A8, A9, A10]      # Sekwencja 5
]
```

**Liczba sekwencji:** `max(1, len(possession) - sequence_length + 1)`

**Kod:**

```python
def _create_sequences_sliding_window(self, features, sequence_length):
    sequences = []
    num_actions = len(features)
    
    if num_actions < sequence_length:
        # Jeśli za krótkie, paddinguj
        sequences.append(self._pad_sequence(features, sequence_length))
    else:
        # Sliding window
        for i in range(num_actions - sequence_length + 1):
            sequences.append(features[i:i+sequence_length])
    
    return np.array(sequences)
```

#### **Tryb VALIDATION**

**Cel:** Jeden output na posiadanie (dla ewaluacji)

**Metoda:** Ostatnie N akcji lub padding

```python
possession = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]  # 10 akcji
sequence_length = 6

# Generuje 1 sekwencję (ostatnie 6):
sequence = [A5, A6, A7, A8, A9, A10]
```

**Padding dla krótkich posiadań:**

```python
possession = [A1, A2, A3, A4]  # 4 akcje (< 6)
sequence_length = 6

# Paddinguje zerami:
sequence = [
    [0, 0, ..., 0],  # Padding (46 zer)
    [0, 0, ..., 0],  # Padding (46 zer)
    A1,
    A2,
    A3,
    A4
]
```

**Kod:**

```python
def _create_sequences_last_n(self, features, sequence_length):
    num_actions = len(features)
    
    if num_actions < sequence_length:
        # Paddinguj
        sequence = self._pad_sequence(features, sequence_length)
    else:
        # Ostatnie N akcji
        sequence = features[-sequence_length:]
    
    return np.array([sequence])  # Zwróć array z jedną sekwencją
```

### 5.2 Handling paddingu w modelu

**Masking layer w Keras:**

```python
model = Sequential([
    Masking(mask_value=0.0, input_shape=(6, 46)),  # Ignoruje paddingowane kroki
    LSTM(128),
    Dense(1)
])
```

**Jak działa masking:**

```python
sequence = [
    [0, 0, ..., 0],  # Timestep 1: zamaskowany (ignorowany przez LSTM)
    [0, 0, ..., 0],  # Timestep 2: zamaskowany
    [0.5, 0.4, ...], # Timestep 3: przetwarzany
    [0.6, 0.3, ...], # Timestep 4: przetwarzany
    [0.7, 0.2, ...], # Timestep 5: przetwarzany
    [0.8, 0.1, ...]  # Timestep 6: przetwarzany
]
```

---

## 6️⃣ Etap 6: Label Generation (Generowanie etykiet)

### 6.1 Funkcja: `_create_label()`

**Kod:**

```python
def _create_label(self, actions_df):
    return get_actions_value(actions_df)
```

### 6.2 Funkcja: `get_actions_value()`

**Kod:** `src/ml/preprocessing/action_valuation.py`

```python
def get_actions_value(actions_df):
    # 1. Suma xG wszystkich strzałów w sekwencji
    total_xg = actions_df['xG'].sum()
    
    # 2. xT ostatniej akcji
    final_xt = actions_df['xT'].iloc[-1]
    
    # 3. Wartość = max(sum(xG), xT)
    value = max(total_xg, final_xt)
    
    return value
```

### 6.3 Logika wyceny

**Przypadek 1: Sekwencja ze strzałem**

```python
actions = [
    Pass (xG=0.0, xT=0.02),
    Pass (xG=0.0, xT=0.05),
    Dribble (xG=0.0, xT=0.08),
    Pass (xG=0.0, xT=0.15),
    Shot (xG=0.35, xT=0.20)
]

total_xg = 0.35
final_xt = 0.20
value = max(0.35, 0.20) = 0.35  # xG dominuje
```

**Przypadek 2: Sekwencja bez strzału**

```python
actions = [
    Pass (xG=0.0, xT=0.02),
    Pass (xG=0.0, xT=0.05),
    Pass (xG=0.0, xT=0.12),
    Pass (xG=0.0, xT=0.18),
    Pass (xG=0.0, xT=0.25)
]

total_xg = 0.0
final_xt = 0.25
value = max(0.0, 0.25) = 0.25  # xT dominuje
```

**Przypadek 3: Gol**

```python
actions = [
    Pass (xG=0.0, xT=0.05),
    Cross (xG=0.0, xT=0.10),
    Shot (xG=0.45, xT=0.30, result=GOAL)
]

total_xg = 0.45  # Nawet jeśli gol, używamy xG
final_xt = 0.30
value = max(0.45, 0.30) = 0.45
```

### 6.4 Rozkład wartości

**Dla Euro 2024:**

```
Percentyle wartości sekwencji:
P25:  0.005 (niskie zagrożenie)
P50:  0.015 (średnie zagrożenie)
P75:  0.045 (wysokie zagrożenie)
P90:  0.120 (bardzo wysokie)
P99:  0.350 (strzały z pola karnego)
Max:  0.950 (gole z kilku metrów)
```

---

## 7️⃣ Etap 7: Finalne Tensory

### 7.1 Zwracane struktury

**Funkcja:** `SequencePreprocessor.process_matches()`

```python
X, y, possessions, matches = preprocessor.process_matches(
    matches_data,
    mode=PreprocessingMode.TRAINING
)
```

**Wyjście:**

| Zmienna | Typ | Shape | Opis |
|---------|-----|-------|------|
| `X` | np.ndarray | (N, 6, 46) | Sekwencje cech |
| `y` | np.ndarray | (N,) | Etykiety (wartości) |
| `possessions` | pd.DataFrame | (M,) | Oryginalne posiadania |
| `matches` | np.ndarray | (N,) | Match IDs dla każdej sekwencji |

**Relacje:**
- `N` = liczba sekwencji (zależy od trybu)
- `M` = liczba posiadań
- W trybie TRAINING: `N >> M` (sliding windows)
- W trybie VALIDATION: `N == M` (jeden output na possession)

### 7.2 Przykładowe wartości dla Euro 2024

#### **Training set (37 meczów):**

```python
X_train.shape = (18_432, 6, 46)   # 18k sekwencji
y_train.shape = (18_432,)
possessions_train.shape = (8_956,)  # ~9k posiadań
matches_train.shape = (18_432,)

# Średnio: 18432/8956 = 2.06 sekwencji na posiadanie
```

#### **Validation set (9 meczów):**

```python
X_val.shape = (2_187, 6, 46)
y_val.shape = (2_187,)
possessions_val.shape = (2_187,)    # 1:1 (tryb VALIDATION)
matches_val.shape = (2_187,)
```

#### **Test set (5 meczów):**

```python
X_test.shape = (1_245, 6, 46)
y_test.shape = (1_245,)
possessions_test.shape = (1_245,)
matches_test.shape = (1_245,)
```

---

## 8️⃣ Przepływ danych - Podsumowanie

### 8.1 Diagram szczegółowy

```
┌─────────────────────────────────────────────────┐
│ StatsBomb JSON Events (~2000/mecz)              │
│ • Zagnieżdżona struktura                        │
│ • Różne typy eventów                            │
│ • Współrzędne (0-120, 0-80)                     │
└──────────────────┬──────────────────────────────┘
                   │
                   │ spadl.convert_to_actions()
                   ↓
┌─────────────────────────────────────────────────┐
│ SPADL Actions (~1500/mecz)                      │
│ • Płaska struktura DataFrame                    │
│ • Współrzędne (0-105, 0-68)                     │
│ • Enumeracje zamiast stringów                   │
└──────────────────┬──────────────────────────────┘
                   │
                   │ normalize_pitch()
                   │ + enrich_actions()
                   ↓
┌─────────────────────────────────────────────────┐
│ Normalized & Enriched Actions                   │
│ • Kierunek gry: zawsze w prawo                  │
│ • + xG, xT, pressure, counterpress              │
└──────────────────┬──────────────────────────────┘
                   │
                   │ groupby('possession')
                   ↓
┌─────────────────────────────────────────────────┐
│ Possessions (~200/mecz)                         │
│ Dict[int, DataFrame]                            │
│ • Średnio 4-6 akcji/posiadanie                  │
└──────────────────┬──────────────────────────────┘
                   │
                   │ FEATURE ENGINEERING
                   │ (Ekstrakcja surowych cech)
                   ↓
┌─────────────────────────────────────────────────┐
│ Raw Feature Vectors (per action)                │
│ • Spatial: start_x, start_y, end_x, end_y       │
│ • Geometric: dx, dy, distance, angle            │
│ • Temporal: time_seconds                        │
│ • Contextual: booleans                          │
│ • Categorical: type_id, result_id, bodypart_id  │
└──────────────────┬──────────────────────────────┘
                   │
                   │ NORMALIZACJA
                   │ (_normalize_features)
                   ↓
┌─────────────────────────────────────────────────┐
│ Normalized Feature Vectors (per action)         │
│ np.ndarray (num_actions, 46)                    │
│ • Spatial (4): skalowanie do [0,1]              │
│ • Geometric (3): skalowanie + sin/cos           │
│ • Temporal (1): skalowanie do [0,1]             │
│ • Contextual (3): konwersja do {0.0, 1.0}       │
│ • Categorical (35): one-hot encoding            │
└──────────────────┬──────────────────────────────┘
                   │
      ┌────────────┴────────────┐
      │                         │
      ↓ TRAINING                ↓ VALIDATION
┌─────────────┐          ┌─────────────┐
│ Sliding     │          │ Last N      │
│ Window      │          │ + Padding   │
└──────┬──────┘          └──────┬──────┘
       │                        │
       │ Sequences: N >> M      │ Sequences: N == M
       └────────────┬───────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Sequences (num_sequences, 6, 46)                │
│ + Padding (zeros) dla krótkich posiadań         │
└──────────────────┬──────────────────────────────┘
                   │
                   │ get_actions_value()
                   ↓
┌─────────────────────────────────────────────────┐
│ Labels (num_sequences,)                         │
│ value = max(sum(xG), xT_final)                  │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────┐
│ FINALNE TENSORY                                 │
│ X: (N, 6, 46) - sekwencje cech                  │
│ y: (N,) - etykiety wartości                     │
│ possessions: (M,) - oryginalne posiadania       │
│ matches: (N,) - match IDs                       │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────┐
│ MODEL (Attention LSTM)                          │
│ Input: (batch, 6, 46)                           │
│ Output: (batch, 1) - predicted value            │
└─────────────────────────────────────────────────┘
```

### 8.2 Przepływ liczbowy - przykład

**Jeden mecz Euro 2024:**

```
StatsBomb Events: 2156 eventów
         ↓
SPADL Actions: 1687 akcji
         ↓
Possessions: 187 posiadań
         ↓
Feature Engineering: 1687 surowych cech
  (spatial, geometric, temporal, contextual, categorical)
         ↓
Normalizacja: 1687 wektorów (46D)
  (skalowanie, one-hot encoding)
         ↓
Tworzenie Sekwencji (TRAINING mode):
  • Krótkie (< 6 akcji): 89 posiadań → 89 sekwencji (paddingowanych)
  • Średnie (6-10 akcji): 76 posiadań → 304 sekwencje (sliding window)
  • Długie (> 10 akcji): 22 posiadania → 176 sekwencji (sliding window)
  TOTAL: 569 sekwencji
         ↓
Labels: 569 wartości (range: 0.0-0.95)
         ↓
Tensory:
  X_match: (569, 6, 46)
  y_match: (569,)
```

**Całość (51 meczów):**

```
Training (37 meczów):   18,432 sekwencji
Validation (9 meczów):   2,187 sekwencji
Test (5 meczów):         1,245 sekwencji
───────────────────────────────────────
TOTAL:                  21,864 sekwencji
```

---

## 9️⃣ Optymalizacje i Best Practices

### 9.1 Memory Management

**Problem:** Wszystkie sekwencje w RAM

**Rozwiązanie:**
```python
# Generator dla dużych zbiorów
def sequence_generator(matches, batch_size=32):
    for batch_matches in chunks(matches, batch_size):
        X, y, _, _ = preprocessor.process_matches(batch_matches)
        yield X, y

model.fit(sequence_generator(train_matches), ...)
```

### 9.2 Caching

**Preprocessed data:**
```python
# Zapisz przetworzone dane
np.savez_compressed(
    'data/processed/train.npz',
    X=X_train, y=y_train,
    possessions=possessions_train,
    matches=matches_train
)

# Wczytaj
data = np.load('data/processed/train.npz', allow_pickle=True)
X_train = data['X']
y_train = data['y']
```

### 9.3 Parallel Processing

**Przetwarzanie wielu meczów:**
```python
from multiprocessing import Pool

def process_single_match(match_data):
    match, events = match_data
    return preprocessor.process_match(match_id, match, events)

with Pool(processes=4) as pool:
    results = pool.map(process_single_match, matches_data)
```

---

## 🔟 Walidacja jakości danych

### 10.1 Asercje w preprocessorze

```python
def _validate_features(self, features):
    assert features.shape[1] == 46, f"Expected 46 features, got {features.shape[1]}"
    assert np.all(features[:, :4] >= 0) and np.all(features[:, :4] <= 1), \
        "Spatial features must be in [0, 1]"
    assert np.all(features[:, 7] >= 0) and np.all(features[:, 7] <= 1), \
        "Time diff must be in [0, 1]"
```

### 10.2 Statystyki danych

```python
def print_dataset_stats(X, y, possessions):
    print(f"Sequences: {len(X)}")
    print(f"Possessions: {len(possessions)}")
    print(f"Ratio: {len(X) / len(possessions):.2f} sequences/possession")
    print(f"\nLabel statistics:")
    print(f"  Min:  {y.min():.4f}")
    print(f"  Mean: {y.mean():.4f}")
    print(f"  Max:  {y.max():.4f}")
    print(f"  P25:  {np.percentile(y, 25):.4f}")
    print(f"  P50:  {np.percentile(y, 50):.4f}")
    print(f"  P75:  {np.percentile(y, 75):.4f}")
    print(f"  P90:  {np.percentile(y, 90):.4f}")
```

---

**Koniec dokumentu**

*Dokument utworzony: 8 lutego 2026*
*Plik: docs/DATA_PROCESSING_SCHEMA.md*
