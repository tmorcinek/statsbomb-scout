# Pipeline Treningu Modelu

**Projekt:** StatsBomb Scout  
**Data:** 10 lutego 2026

---

## 📋 Przegląd

Pipeline treningu modelu składa się z **7 kroków** wykonywanych sekwencyjnie w Jupyter Notebook (`pipeline.ipynb`).

---

## 🔄 Schemat Pipeline

<div align="center">

```
   ┌─────────────────────┐
   │  1. LOAD DATA       │
   │  StatsBomb          │
   └──────────┬──────────┘
   ▼
   ┌─────────────────────┐
   │  2. SPLIT DATA      │
   │  Train/Val/Test     │
   │  70% / 20% / 10%    │
   └──────────┬──────────┘
   ▼
   ┌─────────────────────┐
   │  3. PREPROCESSING   │
   │  Sequences 6×46     │
   └──────────┬──────────┘
   ▼
   ┌─────────────────────┐
   │  4. BUILD MODEL     │
   │  LSTM / Transformer │
   │  / Attention LSTM   │
   └──────────┬──────────┘
   ▼
   ┌─────────────────────┐
   │  5. TRAIN MODEL     │
   └──────────┬──────────┘
   ▼
   ┌─────────────────────┐
   │  6. EVALUATE        │
   │  Val metrics        │
   │  Test metrics       │
   └──────────┬──────────┘
   ▼
   ┌─────────────────────┐
   │  7. SAVE RESULTS    │
   │  model / metrics /  │
   │  training history   │
   └─────────────────────┘
```

</div>

---

## 📝 Szczegółowy Opis Kroków

### **Krok 0: Setup - Importy i Konfiguracja**

```python
import config
from src.data.data_loader import load_statsbomb_socceraction_data
from src.ml.models.model_factory import create_model
from src.ml.train import ModelTrainer
```

**Cel:** Załadowanie bibliotek i konfiguracji projektu

**Konfiguracja z `config.py`:**
- `SEQUENCE_LENGTH = 6` - długość sekwencji
- `MINIMUM_SEQUENCE_LENGTH = 6` - minimalna długość
- `BATCH_SIZE = 32`
- `EPOCHS = 100` (może się zatrzymać wcześniej przez EarlyStopping)

---

### **Krok 1: Load and Split Data**

```python
data = load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282)
train_matches, val_matches, test_matches = split_matches(data)
```

**Wejście:**
- Dane StatsBomb (competition_id=55, season_id=282)
- Format: SPADL (Soccer Player Action Description Language)

**Wyjście:**
- `train_matches` - mecze treningowe
- `val_matches` - mecze walidacyjne
- `test_matches` - mecze testowe

**Podział:**
- Train: **70%** meczów
- Validation: **20%** meczów
- Test: **10%** meczów

---

### **Krok 2: Preprocessing Data**

```python
preprocessor = SequencePreprocessor(
    sequence_length=config.SEQUENCE_LENGTH,
    minimum_sequence_length=config.MINIMUM_SEQUENCE_LENGTH,
    xt_model=get_default_xt_model()
)

X_train, y_train, p_train, m_train = preprocessor.process_matches(train_matches)
X_val, y_val, p_val, m_val = preprocessor.process_matches(val_matches)
X_test, y_test, p_test, m_test = preprocessor.process_matches(test_matches)
```

**Operacje:**
1. Generowanie sekwencji akcji (długość 6)
2. Ekstrakcja cech (46 cech na akcję)
3. Obliczanie xThreat dla każdej akcji
4. Generowanie etykiet (wartość sekwencji)

**Wyjście:**
- `X_train`: Sekwencje akcji **shape: (N, 6, 46)**
- `y_train`: Wartości sekwencji **shape: (N,)**
- `p_train`: Player IDs
- `m_train`: Match IDs

**Typowe rozmiary:**
- Train set: ~50,000-100,000 sequences
- Val set: ~15,000-30,000 sequences
- Test set: ~15,000-30,000 sequences

> 📘 **Szczegółowy opis preprocessing pipeline:** Zobacz rozdział [Szczegółowy Pipeline Preprocessing](#szczegółowy-pipeline-preprocessing) poniżej

---

### **Krok 3: Build Model**

```python
MODEL_TYPE = 'bigru'  # Options: 'lstm', 'attention_lstm', 'transformer'
model = create_model(
    MODEL_TYPE,
    input_shape=(config.SEQUENCE_LENGTH, X_train.shape[2])
)

model.summary()
```

**Dostępne architektury:**

| Model Type        | Opis                                      |
|-------------------|-------------------------------------------|
| `lstm`            | Podstawowy LSTM (2 warstwy)               |
| `attention_lstm`  | LSTM z mechanizmem attention              |
| `transformer`     | Transformer z multi-head attention        |
| `bigru`           | Bidirectional GRU z attention             |

**Input Shape:** `(6, 46)`
- 6 timesteps (akcje w sekwencji)
- 46 cech na akcję

**Output:** Pojedyncza wartość [0, 1] - przewidywana wartość sekwencji

---

### **Krok 4: Train Model**

```python
trainer = ModelTrainer(model, f"models/{MODEL_TYPE}/")
trainer.train(
    X_train, y_train,
    X_val, y_val,
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS
)
```

**Parametry treningu:**
- **Batch size:** 32
- **Epochs:** 100 (max)
- **Optimizer:** Adam (learning_rate=0.001)
- **Loss function:** MSE (Mean Squared Error)
- **Metrics:** MAE (Mean Absolute Error)

**Callbacks:**

1. **EarlyStopping**
   - Monitoruje: `val_loss`
   - Patience: 20 epok
   - Zatrzymuje trening gdy brak poprawy

2. **ReduceLROnPlateau**
   - Monitoruje: `val_loss`
   - Factor: 0.5 (zmniejsza learning rate o połowę)
   - Patience: 5 epok

3. **ModelCheckpoint**
   - Monitoruje: `val_loss`
   - Zapisuje najlepszy model: `best_model.keras`

**Typowy czas treningu:**
- LSTM: ~30-40 minut
- Attention LSTM: ~45-60 minut
- Transformer: ~50-70 minut
- BiGRU: ~35-50 minut

---

### **Krok 5: Evaluate Model**

#### 5.1 Ewaluacja na zbiorze walidacyjnym

```python
val_metrics = trainer.evaluate(X_val, y_val)
print(val_metrics)
```

#### 5.2 Ewaluacja na zbiorze testowym

```python
test_metrics = trainer.evaluate(X_test, y_test)
print(test_metrics)
```

**Metryki:**
- `loss` (MSE) - Mean Squared Error
- `mae` - Mean Absolute Error
- `rmse` - Root Mean Squared Error (obliczane ręcznie)

**Przykładowe wyniki (attention_lstm_large):**
```json
{
  "val_loss": 0.000519,
  "val_mae": 0.000518,
  "test_loss": 0.000284,
  "test_mae": 0.000283,
  "test_rmse": 0.016853
}
```

---

### **Krok 6: Save Results**

```python
all_metrics = {
    'validation': val_metrics,
    'test': test_metrics
}

trainer.save_training_metrics(all_metrics)
trainer.plot_training_history(filename="training_history.png")
```

**Zapisywane pliki:**

```
models/{MODEL_TYPE}/
├── best_model.keras           # Najlepszy model (najniższa val_loss)
├── metrics.json               # Metryki ewaluacji
└── training_history.png       # Wykresy Loss i MAE
```

**Przykład `metrics.json`:**
```json
{
  "validation": {
    "loss": 0.000519,
    "mae": 0.000518
  },
  "test": {
    "loss": 0.000284,
    "mae": 0.000283,
    "rmse": 0.016853
  }
}
```

---

### **Krok 7: Analysis**

```python
val_matches_df = matches_info_df(val_matches)
print(f"Total matches: {len(val_matches_df)}")
print(f"Total events: {val_matches_df['events_count'].sum()}")
```

**Analiza:**
- Liczba meczów w każdym zbiorze
- Liczba zdarzeń (events)
- Średnia liczba zdarzeń na mecz

---

## 🔬 Szczegółowy Pipeline Preprocessing

Ten rozdział opisuje **szczegółowo** jak działa przetwarzanie danych od surowych eventów StatsBomb do sekwencji treningowych.

---

### 📊 Architektura Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE (SequencePreprocessor)            │
└─────────────────────────────────────────────────────────────────────────────┘

MATCH DATA (StatsBomb Events)
│
├─ Match: 3788741
│  Events: 2,459
│  Players: 22
│
▼
┌──────────────────────────────────────────────────────────────┐
│ KROK 1: Konwersja do SPADL Format                           │
│ (extract_actions_from_events)                               │
└──────────────────────────────────────────────────────────────┘
│  StatsBomb JSON → SPADL Actions
│  • Normalizacja typów akcji (23 typy)
│  • Normalizacja pozycji (x,y coordinates)
│  • Dodanie ID posiadania (possession)
│
▼
┌──────────────────────────────────────────────────────────────┐
│ KROK 2: Wzbogacenie Danych                                  │
│ (enrich_actions_with_event_data)                           │
└──────────────────────────────────────────────────────────────┘
│  Dodaje metadane z oryginalnych eventów:
│  • xG (Expected Goals) dla strzałów
│  • under_pressure, counterpress (bool)
│  • possession_team_id, possession_team_name
│  • player_name, team_name
│
▼
┌──────────────────────────────────────────────────────────────┐
│ KROK 3: Ekstrakcja Posiadań                                 │
│ (extract_possessions)                                        │
└──────────────────────────────────────────────────────────────┘
│  Grupowanie akcji według possession ID:
│  • possession_1: 8 akcji
│  • possession_2: 12 akcji
│  • possession_3: 3 akcji (odrzucone - za krótkie)
│  • ...
│  • possession_n: 15 akcji
│
│  Filtracja: len >= minimum_sequence_length (6)
│
▼
┌──────────────────────────────────────────────────────────────┐
│ KROK 4: Normalizacja Boiska                                 │
│ (normalize_pitch)                                            │
└──────────────────────────────────────────────────────────────┘
│  Zawsze attacking left → right:
│  • Jeśli team != home_team: obrót o 180°
│  • start_x' = 105 - start_x
│  • start_y' = 68 - start_y
│
▼
┌──────────────────────────────────────────────────────────────┐
│ KROK 5: Feature Engineering                                 │
│ (_update_action)                                            │
└──────────────────────────────────────────────────────────────┘
│  Dodaje computed features (7):
│  ✅ xT (Expected Threat) - z xT grid
│  ✅ dx, dy - przemieszczenie wektora
│  ✅ distance - odległość ruchu
│  ✅ angle - kąt ruchu
│  ✅ time_diff - różnica czasu
│  ✅ opposite_action - czy akcja przeciwnika
│
▼
┌──────────────────────────────────────────────────────────────┐
│ KROK 6: Normalizacja i One-Hot Encoding                     │
│ (_normalize_features)                                        │
└──────────────────────────────────────────────────────────────┘
│  46 cech na akcję:
│  ├─ Spatial (4): start_x, start_y, end_x, end_y → [0,1]
│  ├─ Geometric (3): distance, sin(angle), cos(angle)
│  ├─ Temporal (1): time_diff (capped at 10s)
│  ├─ Contextual (3): under_pressure, counterpress, opposite
│  └─ Categorical (35): one-hot
│     ├─ type_id (23): pass, dribble, shot, ...
│     ├─ result_id (6): success, fail, ...
│     └─ bodypart_id (6): foot, head, ...
│
│  Output: (n_actions, 46)
│
▼
┌──────────────────────────────────────────────────────────────┐
│ KROK 7: Tworzenie Sekwencji                                 │
│ (_create_sequences / _create_simple_sequence)               │
└──────────────────────────────────────────────────────────────┘
│
│  MODE: TRAINING
│  ├─ Długie posiadanie (n ≥ 6):
│  │   Sliding window → wiele sekwencji
│  │   [a1,a2,a3,a4,a5,a6] → seq1
│  │   [a2,a3,a4,a5,a6,a7] → seq2
│  │   [a3,a4,a5,a6,a7,a8] → seq3
│  │
│  └─ Krótkie posiadanie (n < 6):
│      Padding z zerami
│      [a1,a2,a3,a4] → [a1,a2,a3,a4,0,0]
│
│  MODE: VALIDATION
│  └─ Zawsze pojedyncza sekwencja:
│      - Jeśli n ≥ 6: ostatnie 6 akcji
│      - Jeśli n < 6: padding z zerami
│
│  Output: (n_sequences, 6, 46)
│
▼
┌──────────────────────────────────────────────────────────────┐
│ KROK 8: Generowanie Etykiet                                 │
│ (_create_label → get_actions_value)                         │
└──────────────────────────────────────────────────────────────┘
│  Wartość sekwencji = max(Σ xG, xT_końcowe)
│
│  Przykład:
│  • Sekwencja z golem: xG=0.8 → label=0.8
│  • Sekwencja bez strzału: xT=0.15 → label=0.15
│
│  Output: (n_sequences,)
│
▼
┌──────────────────────────────────────────────────────────────┐
│ OUTPUT: Training Data                                        │
└──────────────────────────────────────────────────────────────┘
   X: (N, 6, 46)  - Sequences
   y: (N,)        - Labels
   p: (N,)        - Possession IDs
   m: (N,)        - Match IDs
```

---

### 🔍 Szczegóły Krok po Kroku

#### **KROK 1: Konwersja do SPADL Format**

**Funkcja:** `extract_actions_from_events()`

**Input:** StatsBomb Events (JSON format)
```json
{
  "id": "abc123",
  "type": {"name": "Pass"},
  "location": [50.0, 40.0],
  "pass": {
    "end_location": [60.0, 45.0],
    "recipient": {"id": 5503, "name": "Lionel Messi"}
  }
}
```

**Output:** SPADL Actions (DataFrame)
```
   action_id  period_id  time_seconds  team_id  player_id  type_id  result_id  bodypart_id  start_x  start_y  end_x  end_y
0          0          1          0.00      217       5503        0          1            1     52.5     34.0   63.0   40.8
1          1          1          1.23      217       5503        2          1            1     63.0     40.8   70.2   38.5
...
```

**23 Typy Akcji (type_id):**
- 0: pass
- 1: cross
- 2: throw_in
- 3: dribble
- 4: tackle
- 5: interception
- 6: shot
- 7: shot_penalty
- 8: shot_freekick
- ... (23 total)

---

#### **KROK 2: Wzbogacenie Danych**

**Funkcja:** `enrich_actions_with_event_data()`

**Dodawane kolumny:**
- `xG` - Expected Goals (tylko dla strzałów)
- `under_pressure` - czy akcja pod presją
- `counterpress` - czy kontratak
- `possession` - ID posiadania piłki
- `possession_team_id` - drużyna posiadająca
- `player_name`, `team_name` - nazwy (metadane)

**Przykład xG:**
```python
# Dla strzałów z events['extra']['shot']['statsbomb_xg']
action_145: type=shot, xG=0.23
action_146: type=pass, xG=0.00
```

---

#### **KROK 3: Ekstrakcja Posiadań**

**Funkcja:** `extract_possessions()`

**Proces:**
1. Grupowanie akcji według `possession` (ID posiadania)
2. Filtrowanie posiadań: `len >= minimum_sequence_length`

**Przykład:**
```
Match 3788741: 2459 events
│
├─ possession_1: 8 akcji ✅ (≥6)
│   [pass, pass, dribble, pass, cross, shot, clearance, throw_in]
│
├─ possession_2: 12 akcji ✅ (≥6)
│   [pass, pass, pass, dribble, pass, pass, shot, ...]
│
├─ possession_3: 3 akcji ❌ (< 6, odrzucone)
│   [pass, interception, clearance]
│
├─ possession_4: 15 akcji ✅ (≥6)
│   [...15 actions...]
│
└─ ...
   Total: 89 possessions → 67 valid (≥6 actions)
```

**Statystyki:**
- Średnia długość posiadania: ~8-10 akcji
- ~25% posiadań odrzuconych (< 6 akcji)
- ~75% posiadań użytecznych

---

#### **KROK 4: Normalizacja Boiska**

**Funkcja:** `normalize_pitch()`

**Cel:** Zawsze atakujemy od lewej → prawej strony (0 → 105)

**Transformacja (jeśli team != home_team):**
```python
# Obrót o 180°
start_x' = 105 - start_x
start_y' = 68 - start_y
end_x' = 105 - end_x
end_y' = 68 - end_y
```

**Wizualizacja:**
```
PRZED NORMALIZACJĄ (away team):
┌─────────────────────────────────────┐
│      ⚽ ←─────────────── ⚽         │  Away attacks this way
│         (105,34)      (20,34)       │
└─────────────────────────────────────┘

PO NORMALIZACJI:
┌─────────────────────────────────────┐
│      ⚽ ───────────────→ ⚽         │  Always attack this way
│       (0,34)        (85,34)         │
└─────────────────────────────────────┘
```

---

#### **KROK 5: Feature Engineering**

**Funkcja:** `_update_action()`

**Dodawane cechy (7 nowych kolumn):**

| Feature          | Formula                        | Przykład         |
|------------------|--------------------------------|------------------|
| `xT`             | `xt_model.predict(end_x, end_y)` | 0.025           |
| `dx`             | `end_x - start_x`              | +12.5 (forward) |
| `dy`             | `end_y - start_y`              | -3.2 (left)     |
| `distance`       | `√(dx² + dy²)`                 | 12.9 m          |
| `angle`          | `atan2(dy, dx)`                | -0.25 rad       |
| `time_diff`      | `time[i] - time[i-1]`          | 1.2 s           |
| `opposite_action`| `team_id != possession_team`   | 0 (same team)   |

**xThreat (Expected Threat):**
- Grid 12×8 na boisku (105m × 68m)
- Każda komórka ma wartość xT [0, 1]
- Wyższa wartość = bliżej bramki przeciwnika

```
xT Grid Visualization:
      Low Threat              High Threat
┌──────────────────────────────────────────┐
│ 0.01  0.01  0.02  0.03  0.05  0.08  0.15 │
│ 0.01  0.02  0.03  0.05  0.08  0.12  0.20 │
│ 0.02  0.03  0.05  0.08  0.12  0.18  0.25 │
│ ...                                      │
└──────────────────────────────────────────┘
```

---

#### **KROK 6: Normalizacja i One-Hot Encoding**

**Funkcja:** `_normalize_features()`

**Output: 46 cech**

```python
Feature Vector (46 dims):
┌─────────────────────────────────────────┐
│ [0] start_x      : 0.52  (52.5/105)    │
│ [1] start_y      : 0.50  (34.0/68)     │
│ [2] end_x        : 0.60  (63.0/105)    │
│ [3] end_y        : 0.60  (40.8/68)     │
├─────────────────────────────────────────┤
│ [4] distance     : 0.12  (12.6/105)    │
│ [5] sin(angle)   : 0.45                │
│ [6] cos(angle)   : 0.89                │
├─────────────────────────────────────────┤
│ [7] time_diff    : 0.12  (1.2s/10s)    │
├─────────────────────────────────────────┤
│ [8] under_pressure : 0.0               │
│ [9] counterpress   : 0.0               │
│ [10] opposite_action: 0.0              │
├─────────────────────────────────────────┤
│ [11-33] type_id one-hot (23 dims)      │
│   [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,...]  │
│   → type_id=3 (dribble)                │
├─────────────────────────────────────────┤
│ [34-39] result_id one-hot (6 dims)     │
│   [0,1,0,0,0,0] → result_id=1 (success)│
├─────────────────────────────────────────┤
│ [40-45] bodypart_id one-hot (6 dims)   │
│   [0,1,0,0,0,0] → bodypart_id=1 (foot) │
└─────────────────────────────────────────┘
Total: 4+3+1+3+23+6+6 = 46 features
```

**Normalizacja zakresów:**
- Spatial: `[0, 105]` → `[0, 1]`, `[0, 68]` → `[0, 1]`
- Distance: `[0, 105]` → `[0, 1]`
- Time: cap at 10s, normalize to `[0, 1]`
- Angles: `sin/cos` → already `[-1, 1]`

---

#### **KROK 7: Tworzenie Sekwencji**

**2 Tryby:** Training vs Validation

##### **Mode: TRAINING**

**Długie posiadanie (n ≥ 6):** Sliding Window

```python
Possession: 10 akcji [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
Window size: 6

Sequences generated:
seq_0: [a0, a1, a2, a3, a4, a5]  → label_0
seq_1: [a1, a2, a3, a4, a5, a6]  → label_1
seq_2: [a2, a3, a4, a5, a6, a7]  → label_2
seq_3: [a3, a4, a5, a6, a7, a8]  → label_3
seq_4: [a4, a5, a6, a7, a8, a9]  → label_4

Total: 5 sequences from 1 possession
```

**Krótkie posiadanie (6 ≤ n < 6):** Padding

```python
Possession: 4 akcje [a0, a1, a2, a3]
Window size: 6

Sequence:
seq_0: [a0, a1, a2, a3, 0, 0]  → label_0
        ↑───actual──↑  ↑─pad─↑

Total: 1 sequence (padded)
```

##### **Mode: VALIDATION**

**Zawsze 1 sekwencja** na posiadanie:

```python
# Długie posiadanie (n ≥ 6)
Possession: 10 akcji
Sequence: ostatnie 6 akcji [a4, a5, a6, a7, a8, a9]

# Krótkie posiadanie (n < 6)
Possession: 4 akcje
Sequence: [a0, a1, a2, a3, 0, 0] (padded)
```

**Dlaczego różne tryby?**
- **Training:** Maksymalizacja danych (data augmentation)
- **Validation:** Realistyczna ewaluacja (1 predykcja = 1 posiadanie)

---

#### **KROK 8: Generowanie Etykiet**

**Funkcja:** `get_actions_value()`

**Formula:**
```python
label = max(Σ xG, xT_last)
```

**Przypadki:**

| Scenario                  | xG Sum | xT Last | Label | Reason                |
|---------------------------|--------|---------|-------|-----------------------|
| Goal scored               | 0.85   | 0.20    | 0.85  | max(0.85, 0.20)       |
| Shot missed               | 0.23   | 0.18    | 0.23  | max(0.23, 0.18)       |
| No shot, good position    | 0.00   | 0.35    | 0.35  | max(0.00, 0.35)       |
| No shot, bad position     | 0.00   | 0.05    | 0.05  | max(0.00, 0.05)       |
| Multiple shots            | 1.15   | 0.10    | 1.15  | sum of all xG         |

**Interpretacja:**
- Label ≈ **"prawdopodobieństwo strzelenia gola z tej sekwencji"**
- High label (>0.5): Sekwencja zakończona golem lub bardzo dobrą szansą
- Low label (<0.1): Słaba sekwencja bez zagrożenia

---

### 📊 Przykład Kompletny

**Input:** Match 3788741

```
Events: 2,459
Players: 22
Possessions (total): 89
```

**Po filtracji (≥6 akcji):**
```
Valid possessions: 67
Avg length: 9.2 actions
```

**Mode: TRAINING**

```python
possession_1: 8 actions
  → 3 sequences (sliding window)
  → 3 labels

possession_2: 12 actions
  → 7 sequences
  → 7 labels

possession_3: 6 actions (exactly)
  → 1 sequence (no sliding)
  → 1 label

possession_4: 4 actions (too short, but ≥ min_length)
  → 1 sequence (padded)
  → 1 label

... (67 possessions)

Total sequences: ~450-500 per match
```

**Mode: VALIDATION**

```python
possession_1: 8 actions
  → 1 sequence (last 6)
  → 1 label

possession_2: 12 actions
  → 1 sequence (last 6)
  → 1 label

possession_3: 4 actions
  → 1 sequence (padded)
  → 1 label

... (67 possessions)

Total sequences: 67 (= number of valid possessions)
```

**Output Shapes:**

```python
# Training mode
X_train: (87234, 6, 46)  # ~450 seq/match × ~200 matches
y_train: (87234,)
p_train: (87234,)
m_train: (87234,)

# Validation mode
X_val: (13456, 6, 46)   # ~67 seq/match × ~200 matches
y_val: (13456,)
p_val: (13456,)
m_val: (13456,)
```

---

### 🎯 Kluczowe Decyzje Projektowe

#### 1. **Dlaczego sequence_length = 6?**

**Analiza empiryczna:**
- Średnia długość posiadania: 8-10 akcji
- Posiadania < 6 akcji: ~25% (za krótkie, mało informacji)
- Posiadania ≥ 6 akcji: ~75% (wystarczająco długie)
- Sekwencje 6 akcji: wystarczające do przewidzenia wartości

**Trade-off:**
- Krótsze (4-5): za mało kontekstu
- Dłuższe (8-10): mniej przykładów treningowych

#### 2. **Dlaczego padding z zerami?**

**Alternatywy:**
- ❌ Masking (komplikuje architekturę)
- ❌ Replikacja ostatniej akcji (fałszywe informacje)
- ✅ **Padding z zerami** (prosty, działa z LSTM/Transformer)

Model "widzi" padding jako "brak akcji".

#### 3. **Dlaczego sliding window tylko w treningu?**

**Training:**
- Chcemy więcej danych (data augmentation)
- Jedna posiadanie → wiele sekwencji

**Validation:**
- Realistyczna ewaluacja
- Jedna posiadanie → jedna predykcja
- Unikamy "przecieku" danych (data leakage)

#### 4. **Dlaczego max(xG, xT)?**

**Obserwacje:**
- xG > 0 tylko dla strzałów
- xT > 0 dla wszystkich pozycji
- Niektóre sekwencje kończą się bez strzału, ale w dobrej pozycji

**Rozwiązanie:**
```python
label = max(sum(xG), xT_last)
```
- Jeśli był strzał: użyj xG (bardziej precyzyjne)
- Jeśli nie było strzału: użyj xT (pozycja końcowa)

---

### 🔧 Parametry Konfiguracji

| Parameter                 | Value | Impact                                    |
|---------------------------|-------|-------------------------------------------|
| `sequence_length`         | 6     | Długość sekwencji wejściowej              |
| `minimum_sequence_length` | 6     | Min. długość posiadania (filtracja)      |
| `field_length`            | 105   | Długość boiska (SPADL)                    |
| `field_width`             | 68    | Szerokość boiska (SPADL)                  |
| `max_time_diff`           | 10s   | Cap dla time_diff normalization           |
| `n_features`              | 46    | Liczba cech po feature engineering        |

---

### ⚠️ Edge Cases i Obsługa Błędów

#### 1. **Posiadanie z 0 akcjami**
```python
if len(actions) == 0:
    # Pomijane automatycznie przez groupby
    pass
```

#### 2. **Brakujące wartości w eventach**
```python
# xG dla non-shot actions
xG = events['extra'].get('shot', {}).get('statsbomb_xg', 0.0)
# Default: 0.0
```

#### 3. **NaN w współrzędnych**
```python
# One-hot encoding obsługuje NaN
def one_hot_numpy(ids, K):
    valid = (~np.isnan(a)) if a.dtype.kind == "f" else (a >= 0)
    # NaN → [0,0,0,...,0]
```

#### 4. **Bardzo długie posiadania (>50 akcji)**
```python
# Sliding window generuje wiele sekwencji
# Przykład: 50 akcji → 45 sekwencji
# To jest OK, więcej danych treningowych
```

---

### 📈 Statystyki Pipeline

**Typowy mecz (LaLiga):**
```
Input:
  Events: ~2,500
  Players: 22

After SPADL conversion:
  Actions: ~2,300 (usunięte non-action events)

After possession extraction:
  Possessions: ~90
  Valid (≥6 actions): ~67 (75%)
  Avg length: 9.2 actions

After sequence generation (training mode):
  Sequences: ~450-500
  Labels: ~450-500

After sequence generation (validation mode):
  Sequences: ~67 (one per possession)
  Labels: ~67
```

**Cały dataset (282 mecze):**
```
Training set (70%): ~197 matches
  → ~90,000 sequences

Validation set (20%): ~56 matches
  → ~3,700 possessions
  → ~3,700 sequences (validation mode)

Test set (10%): ~28 matches
  → ~1,900 possessions
  → ~1,900 sequences (validation mode)
```

---

## 🎯 Dane Wejściowe i Wyjściowe

### Struktura Danych

```
INPUT: StatsBomb Data
├── Matches (JSON)
├── Events (JSON)
└── Lineups (JSON)

     ↓ [data_loader.py]

SPADL Format
├── actions.parquet
├── matches.parquet
└── players.parquet

     ↓ [SequencePreprocessor]

Training Data
├── X_train: (N, 6, 46)  - Sekwencje akcji
├── y_train: (N,)        - Wartości sekwencji
├── p_train: (N,)        - Player IDs
└── m_train: (N,)        - Match IDs

     ↓ [ModelTrainer]

Trained Model
├── best_model.keras     - Wytrenowany model
├── metrics.json         - Metryki
└── training_history.png - Historia treningu
```

---

## 📊 Przykładowy Output Treningu

```
Train set: 87234 sequences with shape (87234, 6, 46)
Val set: 28901 sequences with shape (28901, 6, 46)
Test set: 29456 sequences with shape (29456, 6, 46)

Epoch 1/100
2726/2726 [==============================] - 45s 16ms/step
  loss: 0.0012 - mae: 0.0234 - val_loss: 0.0008 - val_mae: 0.0189

Epoch 2/100
2726/2726 [==============================] - 43s 16ms/step
  loss: 0.0007 - mae: 0.0156 - val_loss: 0.0006 - val_mae: 0.0134

...

Epoch 28/100
2726/2726 [==============================] - 44s 16ms/step
  loss: 0.0003 - mae: 0.0031 - val_loss: 0.0005 - val_mae: 0.0052

Early stopping - no improvement for 20 epochs
Best model saved with val_loss: 0.0005 at epoch 28

Validation metrics: {'loss': 0.000519, 'mae': 0.000518}
Test metrics: {'loss': 0.000284, 'mae': 0.000283, 'rmse': 0.016853}

Pipeline completed successfully!
```

---

## 🔧 Konfiguracja

### Parametry w `config.py`

| Parametr                    | Wartość | Opis                              |
|-----------------------------|---------|-----------------------------------|
| `SEQUENCE_LENGTH`           | 6       | Długość sekwencji akcji           |
| `MINIMUM_SEQUENCE_LENGTH`   | 6       | Minimalna długość sekwencji       |
| `BATCH_SIZE`                | 32      | Rozmiar batcha                    |
| `EPOCHS`                    | 100     | Maksymalna liczba epok            |
| `EARLY_STOPPING_PATIENCE`   | 20      | Patience dla EarlyStopping        |
| `REDUCE_LR_PATIENCE`        | 5       | Patience dla ReduceLROnPlateau    |

---

## 🚀 Uruchamianie Pipeline

### Metoda 1: Jupyter Notebook

```bash
jupyter notebook pipeline.ipynb
```

Wykonaj komórki sekwencyjnie (Shift+Enter)

### Metoda 2: Python Script

```bash
jupyter nbconvert --to script pipeline.ipynb
python pipeline.py
```

### Metoda 3: Batch Training (train_models.py)

```bash
python src/training/train_models.py
```

Trenuje wiele modeli z różnymi konfiguracjami

---

## 📈 Wizualizacja Rezultatów

### Training History Plot

Wykres `training_history.png` pokazuje:

```
┌─────────────────────────────────────────────────────────────┐
│  Loss (MSE)                  │  MAE                          │
│                              │                               │
│  1.0 ┐                       │  0.03 ┐                       │
│      │ train_loss            │       │ train_mae             │
│  0.8 ┤─────────              │  0.02 ┤─────────              │
│      │         ──────        │       │         ──────        │
│  0.6 ┤               ────    │  0.01 ┤               ────    │
│      │ val_loss          ─── │       │ val_mae           ─── │
│  0.4 ┤                       │  0.00 ┤                       │
│      │                       │       │                       │
│  0.2 ┘                       │       ┘                       │
│      0    10   20   30  epoki│       0    10   20   30  epoki│
└─────────────────────────────────────────────────────────────┘
```

**Interpretacja:**
- ✅ **Dobry fit:** train_loss i val_loss zbliżone, obie maleją
- ⚠️ **Overfitting:** val_loss rośnie, train_loss maleje
- ⚠️ **Underfitting:** obie metryki wysokie, nie maleją

---

## ✅ Checklist Treningu

- [ ] Załadowane dane StatsBomb
- [ ] Podział train/val/test wykonany
- [ ] Preprocessing zakończony pomyślnie
- [ ] Model zbudowany i skompilowany
- [ ] Trening uruchomiony
- [ ] EarlyStopping zadziałał (lub osiągnięto max epochs)
- [ ] Model zapisany (`best_model.keras`)
- [ ] Metryki zapisane (`metrics.json`)
- [ ] Wykresy wygenerowane (`training_history.png`)
- [ ] Test metrics < 0.001 (MAE) dla attention models ✅

---

## 📚 Powiązane Dokumenty

- **METRICS.md** - Opis metryk ewaluacji (MAE, RMSE, Loss)
- **BEST_MODELS_COMPARISON.md** - Porównanie najlepszych modeli
- **MODEL_CONFIGURATION_MAPPING.md** - Mapowanie konfiguracji
- **LABEL_GENERATION.md** - Generowanie etykiet xThreat
- **DATA_PROCESSING_SCHEMA.md** - Przetwarzanie danych

---

**Ostatnia aktualizacja:** 10 lutego 2026  
**Autor:** Tomasz Morcinek  
**Plik źródłowy:** `pipeline.ipynb`




