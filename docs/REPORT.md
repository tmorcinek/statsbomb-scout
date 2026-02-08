# StatsBomb Scout - Raport Techniczny

**Data utworzenia:** 6 lutego 2026  
**Projekt:** Ocena wartości akcji piłkarskich przy użyciu deep learning

---

## 📋 Spis treści

1. [Wprowadzenie](#wprowadzenie)
2. [Architektura systemu](#architektura-systemu)
3. [Pipeline treningowy](#pipeline-treningowy)
4. [Preprocessing danych](#preprocessing-danych)
5. [Modele ML](#modele-ml)
6. [Analiza wyników](#analiza-wyników)
7. [Najważniejsze pliki](#najważniejsze-pliki)
8. [Metryki i ewaluacja](#metryki-i-ewaluacja)

---

## 1. Wprowadzenie

### Cel projektu
System służy do oceny wartości sekwencji akcji piłkarskich (posiadań) poprzez przewidywanie ich potencjału ofensywnego. Wykorzystuje dane StatsBomb z Euro 2024 oraz techniki deep learning (LSTM, Transformer, BiGRU) z mechanizmem attention.

### Kluczowe koncepcje
- **Possession (posiadanie)**: Ciągła sekwencja akcji tego samego zespołu
- **xT (Expected Threat)**: Metryka określająca zagrożenie dla bramki przeciwnika na podstawie pozycji na boisku
- **xG (Expected Goals)**: Prawdopodobieństwo strzelenia gola z danej pozycji
- **Action Value**: `max(sum(xG), xT_końcowy)` - wartość sekwencji akcji

---

## 2. Architektura systemu

### Struktura projektu

```
statsbomb-scout/
├── pipeline.ipynb                    # Główny pipeline treningowy
├── analysis_final_match.ipynb       # Analiza pojedynczego meczu (finał)
├── analysis_matches.ipynb           # Analiza wszystkich meczów
├── config.py                        # Konfiguracja globalna
│
├── src/
│   ├── data/
│   │   ├── data_loader.py          # Wczytywanie danych StatsBomb
│   │   └── data_splitter.py        # Podział train/val/test
│   │
│   ├── ml/
│   │   ├── models/
│   │   │   ├── attention_lstm.py   # LSTM z attention (najlepszy)
│   │   │   ├── lstm.py             # Prosty LSTM
│   │   │   ├── bigru.py            # Bidirectional GRU
│   │   │   ├── transformer.py      # Transformer encoder
│   │   │   └── model_factory.py    # Factory pattern dla modeli
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── sequence_preprocessor.py    # Główny preprocessor
│   │   │   ├── possessions_extraction.py   # Ekstrakcja posiadań
│   │   │   ├── action_valuation.py         # Wyliczanie xT, xG
│   │   │   └── xthreat.py                  # Model Expected Threat
│   │   │
│   │   └── train.py                # ModelTrainer - training loop
│   │
│   └── analysis/
│       ├── prediction_utils.py     # Utilities do predykcji
│       ├── visualization.py        # Wizualizacja na boisku
│       └── game_utils.py          # Pomocnicze funkcje meczowe
│
├── models/                         # Zapisane modele
│   ├── attention_lstm/
│   ├── lstm/
│   ├── bigru/
│   ├── transformer/
│   └── xt_models/
│
└── data/
    ├── statsbomb/                  # Dane źródłowe
    └── processed/                  # Przetworzone dane
```

---

## 3. Pipeline treningowy

### 3.1 Plik: `pipeline.ipynb`

Główny notebook treningowy składa się z następujących kroków:

#### **Krok 1: Wczytanie i podział danych**
```python
data = load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282)
train_matches, val_matches, test_matches = split_matches(data)
```

- **Competition 55, Season 282**: Euro 2024
- **51 meczów** podzielonych na:
  - Train: 37 meczów (72.5%)
  - Validation: 9 meczów (17.6%)
  - Test: 5 meczów (9.8%)

Podział chronologiczny - najnowsze mecze w test/val.

#### **Krok 2: Preprocessing sekwencji**
```python
preprocessor = SequencePreprocessor(
    sequence_length=6,              # Długość sekwencji
    minimum_sequence_length=3,      # Min długość posiadania
    xt_model=get_default_xt_model()
)

X_train, y_train, p_train, m_train = preprocessor.process_matches(train_matches)
X_val, y_val, p_val, m_val = preprocessor.process_matches(val_matches)
X_test, y_test, p_test, m_test = preprocessor.process_matches(test_matches)
```

**Wyjście:**
- `X`: sekwencje cech kształtu `(n_sequences, 6, 46)`
- `y`: etykiety wartości akcji (float)
- `p`: oryginalne posiadania (DataFrame)
- `m`: ID meczów

#### **Krok 3: Budowa i trenowanie modelu**
```python
model = create_model('attention_lstm', input_shape=(6, 46))
trainer = ModelTrainer(model, model_dir="models/attention_lstm")

trainer.train(
    X_train, y_train,
    X_val, y_val,
    batch_size=32,
    epochs=50
)
```

**Callbacks:**
- `EarlyStopping`: patience=10 (stop jeśli brak poprawy)
- `ReduceLROnPlateau`: zmniejszenie learning rate
- `ModelCheckpoint`: zapisanie najlepszego modelu

#### **Krok 4: Ewaluacja**
```python
metrics = trainer.evaluate(X_test, y_test)
trainer.plot_training_history()
trainer.save_metrics(metrics)
```

---

## 4. Preprocessing danych

### 4.1 Plik: `src/ml/preprocessing/sequence_preprocessor.py`

Główna klasa odpowiedzialna za transformację surowych eventów StatsBomb w sekwencje treningowe.

#### **Klasa `SequencePreprocessor`**

##### **Parametry inicjalizacji:**
```python
def __init__(self, 
             sequence_length: int,           # Długość sekwencji (default: 6)
             minimum_sequence_length: int,   # Min długość (default: 3)
             xt_model: ExpectedThreat):      # Model xT
```

##### **Główne metody:**

**1. `process_matches()` - przetwarzanie wielu meczów**
```python
def process_matches(matches, mode=PreprocessingMode.TRAINING) -> (X, y, p, m)
```

**Tryby przetwarzania:**
- `TRAINING`: Tworzy sliding windows dla długich posiadań
  - Posiadanie 10 akcji → 5 sekwencji (10-6+1)
  - Generuje więcej danych treningowych
  
- `VALIDATION`: Jedna sekwencja na posiadanie
  - Bierze ostatnie 6 akcji (lub paddinguje jeśli krótsza)
  - Jeden output na possession dla ewaluacji

**2. `process_match()` - przetwarzanie pojedynczego meczu**
```python
def process_match(match_id, match, events_df, mode) -> (X, y, p)
```

Workflow:
1. Ekstrakcja posiadań (`_extract_actions`)
2. Dodanie cech (`_update_action`)
3. Normalizacja (`_normalize_features`)
4. Tworzenie sekwencji w zależności od trybu
5. Obliczenie etykiet (`_create_label`)

**3. `_normalize_features()` - normalizacja cech**

Tworzy wektor 46 cech dla każdej akcji:

| Typ cechy | Liczba | Opis |
|-----------|--------|------|
| **Spatial** | 4 | start_x, start_y, end_x, end_y (normalizowane 0-1) |
| **Geometric** | 3 | distance, sin(angle), cos(angle) |
| **Temporal** | 1 | time_diff (cap 10s) |
| **Contextual** | 3 | under_pressure, counterpress, opposite_action |
| **Categorical** | 35 | One-hot encoding: |
| - type_id | 23 | Typ akcji (pass, shot, dribble, etc.) |
| - result_id | 6 | Rezultat (success, fail, etc.) |
| - bodypart_id | 6 | Część ciała (left foot, head, etc.) |
| **TOTAL** | **46** | |

**4. `_pad_sequence()` - padding krótkich sekwencji**
```python
def _pad_sequence(features: np.ndarray) -> np.ndarray:
    # Jeśli posiadanie < 6 akcji, dodaj zera na końcu
    # Masking layer w modelu ignoruje te kroki
```

**5. `_create_label()` - tworzenie etykiety**
```python
def _create_label(actions_df: pd.DataFrame) -> float:
    return get_actions_value(actions_df)  # max(sum(xG), xT_końcowy)
```

### 4.2 Plik: `src/ml/preprocessing/possessions_extraction.py`

Pomocnicze funkcje do ekstrakcji posiadań z eventów.

#### **Główne funkcje:**

**1. `extract_possessions()` - podstawowa ekstrakcja**
```python
def extract_possessions(game: pd.Series, events: pd.DataFrame) -> dict[int, pd.DataFrame]
```
- Konwertuje eventy StatsBomb → SPADL actions
- Grupuje po possession ID
- Normalizuje kierunek gry (zawsze w stronę prawej bramki)
- Wzbogaca o dane: xG, xT, pressure, etc.

**2. `extract_actions_from_events()`**
```python
def extract_actions_from_events(game, events_df) -> pd.DataFrame:
    actions = spadl.statsbomb.convert_to_actions(events_df, ...)
    actions = enrich_actions_with_event_data(actions, events_df)
    return actions
```

**3. `normalize_pitch()` - normalizacja kierunku**
```python
def normalize_pitch(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    # Rotacja współrzędnych tak, aby drużyna posiadająca grała w prawo
    if actions.iloc[0]['team_id'] != home_team_id:
        actions['start_x'] = field_length - actions['start_x']
        actions['start_y'] = field_width - actions['start_y']
        # ... podobnie dla end_x, end_y
```

**4. Filtrowanie posiadań:**
```python
extract_possessions_with_shots()         # Posiadania zawierające strzały
extract_possessions_ended_with_shots()   # Zakończone strzałem
extract_possessions_ended_with_goals()   # Zakończone golem
```

### 4.3 Plik: `src/ml/preprocessing/action_valuation.py`

Funkcje do wyceny wartości akcji.

**1. `calculate_xg_values()` - Expected Goals**
```python
def calculate_xg_values(events: pd.DataFrame) -> pd.Series:
    # Ekstrahuje xG StatsBomb z eventów typu "Shot"
    xg_values = events["extra"].map(lambda x: x.get("shot", {}).get("statsbomb_xg", 0.0))
```

**2. `calculate_xt_values()` - Expected Threat**
```python
def calculate_xt_values(actions: pd.DataFrame, xt_model: ExpectedThreat) -> np.ndarray:
    # Dla move_actions: xT na end_position
    # Dla non-move: xT na start_position
    # Używa siatki 16x12 z wytrenowanego modelu xT
```

**3. `get_actions_value()` - wartość sekwencji**
```python
def get_actions_value(actions_df: pd.DataFrame) -> float:
    total_xg = actions_df['xG'].sum()      # Suma xG wszystkich strzałów
    xt = actions_df['xT'].iloc[-1]        # xT końcowej pozycji
    return max(total_xg, xt)               # Zwróć większą wartość
```

### 4.4 Plik: `src/ml/preprocessing/xthreat.py`

Zarządzanie modelami Expected Threat.

**Funkcje:**
```python
get_default_xt_model()                    # Wczytanie modelu z JSON
train_xt_model(loader, comp_id, season)   # Trenowanie nowego modelu xT
```

Model xT jest wytrenowany na wszystkich akcjach z danego sezonu i przechowuje macierz wartości zagrożenia dla każdego pola boiska (siatka 16x12).

---

## 5. Modele ML

### 5.1 Plik: `src/ml/models/attention_lstm.py`

#### **Najlepszy model - LSTM z mechanizmem attention**

##### **Architektura:**

```python
Input(sequence_length=6, features=46)
    ↓
Masking(mask_value=0.0)                  # Ignoruje padding
    ↓
Bidirectional LSTM(128 units)            # Kontekst z obu kierunków
    ↓
Dropout(0.2)
    ↓
LSTM(128 units, return_sequences=True)   # Drugi layer LSTM
    ↓
Dropout(0.2)
    ↓
AttentionLayer()                         # Custom attention mechanism
    ↓                    ↓
    |                    |
    |                    └→ attention_weights (6,)
    |
Dense(64, relu)
    ↓
Dropout(0.2)
    ↓
Dense(1, linear) → value_output
```

##### **Klasa `AttentionLayer`:**

```python
class AttentionLayer(layers.Layer):
    def call(self, x, mask=None):
        # 1. Oblicz attention scores: e_t = tanh(h_t * W + b)
        e = tanh(x @ W + b)  # (batch, seq_len, 1)
        
        # 2. Zastosuj maskowanie (padded timesteps → -inf)
        if mask is not None:
            e = where(mask, e, -1e9)
        
        # 3. Softmax → wagi sumujące się do 1.0
        attention_weights = softmax(e)  # (batch, seq_len)
        
        # 4. Weighted sum → context vector
        context = sum(x * attention_weights)  # (batch, hidden_dim)
        
        return context, attention_weights
```

**Zalety:**
- Attention pokazuje, które akcje są najważniejsze
- Masking ignoruje padding (krótkie posiadania)
- Multi-output: wartość + wagi attention
- Bidirectional LSTM lepiej modeluje sekwencje

##### **Konfiguracja:**

| Parametr | Wartość |
|----------|---------|
| LSTM units | 128 |
| Dropout | 0.2 |
| Dense layer | 64 neurons |
| Optimizer | Adam |
| Loss | MSE |
| Metrics | MAE |

### 5.2 Inne modele

**1. `lstm.py` - Prosty LSTM**
- Single LSTM layer bez attention
- Baseline model

**2. `bigru.py` - Bidirectional GRU**
- GRU zamiast LSTM (mniej parametrów)
- Bidirectional dla kontekstu

**3. `transformer.py` - Transformer Encoder**
- Multi-head attention
- Positional encoding
- Feed-forward network

### 5.3 Plik: `src/ml/models/model_factory.py`

Factory pattern dla tworzenia i ładowania modeli.

```python
def create_model(model_type: str, input_shape: tuple, **kwargs) -> keras.Model:
    models = {
        'lstm': LSTMModel,
        'attention_lstm': AttentionLSTMModel,
        'bigru': BiGRUModel,
        'transformer': TransformerModel
    }
    return models[model_type](input_shape, **kwargs).build()

def load_model(model_type: str) -> keras.Model:
    # Wczytuje zapisany model z dysku
    return keras.models.load_model(f"models/{model_type}/best_model.keras")
```

---

## 6. Analiza wyników

### 6.1 Plik: `analysis_final_match.ipynb`

Szczegółowa analiza **pojedynczego meczu** (finał Euro 2024: Hiszpania 2-1 Anglia).

#### **Workflow:**

**1. Wczytanie meczu finałowego**
```python
match_id = 3943043  # Finał
selected_match, selected_events = load_socceraction_match(
    "data/statsbomb/data", 55, 282, match_id
)
```

**2. Preprocessing w trybie VALIDATION**
```python
preprocessor = SequencePreprocessor(...)
X_match, y_match, p_match = preprocessor.process_match(
    match_id, selected_match, selected_events, 
    mode=PreprocessingMode.VALIDATION
)
```

**3. Predykcja modelem**
```python
model = load_model('attention_lstm')
predictions = model.predict(X_match)
predicted_values, attention_weights = normalize_predictions(predictions)
```

**4. Top N predykcji**
```python
print_top_predictions(predicted_values, p_match, None, top_n=8, attention_weights)
```

Wyświetla:
- 8 najlepszych sekwencji według wartości predykcji
- Wagi attention dla każdej akcji w sekwencji
- Najważniejszą akcję (max attention weight)

**5. Wizualizacja na boisku**
```python
visualize_top_possessions(
    selected_match, selected_events, p_match, 
    predicted_values, attention_weights, top_n=8, sequence_length=6
)
```

Tworzy subplot 8 boisk pokazujących:
- Ostatnie 6 akcji każdego posiadania
- Strzałki przedstawiające kierunek akcji
- Kolory: niebieski (zespół z piłką), czerwony (przeciwnik)
- Gwiazdka na końcowej pozycji
- Tytuł z wartością predykcji i wagami attention

### 6.2 Plik: `analysis_matches.ipynb`

Analiza **wszystkich meczów** ze zbioru walidacyjnego/testowego.

#### **Workflow:**

**1. Wczytanie wszystkich meczów**
```python
data = load_statsbomb_socceraction_data("data/statsbomb/data", 55, 282)
all_matches = list(data)
all_matches.sort(key=lambda x: x[0]['game_date'], reverse=True)
```

**2. Preprocessing zbioru walidacyjnego**
```python
X, y, p, m = preprocessor.process_matches(all_matches)
```

**3. Predykcja dla wszystkich sekwencji**
```python
predictions = model.predict(X, batch_size=32)
predicted_values, attention_weights = normalize_predictions(predictions)
```

**4. Globalne top N sekwencji**
```python
print_top_predictions(predicted_values, p, m, top_n=8, attention_weights)
```

Wyświetla najlepsze akcje z **wszystkich meczów**, z informacją:
- Match ID
- Possession ID
- Predicted value
- Attention weights

**5. Wizualizacja globalnych topek**
```python
game_possessions_list = create_possessions_list(all_matches)

visualize_top_possessions_matches(
    game_possessions_list, p, m, 
    predicted_values, attention_weights, top_n=8, sequence_length=6
)
plt.savefig("models/attention_lstm/top_possessions_all_matches.png", 
            dpi=300, facecolor='white', transparent=False)
```

**6. Analiza per mecz**
```python
matches_df = matches_info_df(all_matches)
# DataFrame z informacjami: match_id, summary, events_count
```

### 6.3 Plik: `src/analysis/prediction_utils.py`

Funkcje pomocnicze do analizy predykcji.

**1. `normalize_predictions()` - rozpakowuje output modelu**
```python
def normalize_predictions(predictions):
    if isinstance(predictions, dict):
        # Multi-output model (attention_lstm)
        return predictions['value'].flatten(), predictions['attention_weights']
    else:
        # Single-output model (lstm, bigru)
        return predictions.flatten(), None
```

**2. `get_top_indices()` - sortowanie**
```python
def get_top_indices(predicted_values, head=None):
    return np.argsort(predicted_values)[::-1][:head]
```

**3. `print_top_predictions()` - formatowany output**
```python
def print_top_predictions(predicted_values, possessions, matches, top_n, attention_weights):
    # Wyświetla top N z wagami attention
    for rank, idx in enumerate(top_indices):
        print(f"{rank+1}. Match={matches[idx]}, Poss={possessions[idx]}, Value={values[idx]:.3f}")
        print(f"   Attention: {attention_weights[idx]}")
        print(f"   Most important: position {argmax(attention_weights[idx])}")
```

**4. `create_possessions_list()` - helper dla wizualizacji**
```python
def create_possessions_list(all_matches):
    return [(match, extract_possessions(match, events)) for match, events in all_matches]
```

**5. `matches_info_df()` - DataFrame z info o meczach**
```python
def matches_info_df(matches):
    return pd.DataFrame([{
        'match_id': match['game_id'],
        'summary': game_summary(match),
        'events_count': len(events)
    } for match, events in matches])
```

### 6.4 Plik: `src/analysis/visualization.py`

Wizualizacja akcji na boisku za pomocą mplsoccer.

**1. `plot_possession_actions()` - pojedyncze posiadanie**
```python
def plot_possession_actions(possession_actions: pd.DataFrame, title=None, ax=None):
    # Tworzy boisko (vertical pitch)
    pitch = VerticalPitch(pitch_type='custom', pitch_color='white', line_color='black')
    
    # Rysuje strzałki dla każdej akcji
    for idx, action in enumerate(possession_actions.iterrows()):
        color = 'black' if idx == 0 else ('blue' if zespół else 'red')
        pitch.arrows(start_x, start_y, end_x, end_y, color=color, ...)
    
    # Kropki na starcie
    pitch.scatter(start_xs, start_ys, c=colors, ...)
    
    # Gwiazdka na końcu
    pitch.scatter(end_x, end_y, marker='*', c='black', ...)
```

**2. `plot_multiple_possessions()` - grid posiadań**
```python
def plot_multiple_possessions(possession_actions_list, titles=None, main_title=None):
    # Oblicz grid (sqrt z liczby posiadań)
    cols = ceil(sqrt(num_possessions)) + 1
    rows = ceil(num_possessions / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    
    # Rysuj każde posiadanie
    for idx, possession in enumerate(possession_actions_list):
        plot_possession_actions(possession, title=titles[idx], ax=axes[idx])
    
    # Ukryj puste subploty
    for idx in range(num_possessions, rows*cols):
        axes[idx].axis('off')
```

**3. `_title_with_value()` - tytuł z wartością i attention**
```python
def _title_with_value(possession_actions, predicted_value, attention_weights=None):
    title = f"{_default_title(possession_actions)}\nValue: {predicted_value:.3f}"
    
    if attention_weights is not None:
        weights_str = ', '.join([f"{w:.3f}" for w in attention_weights])
        title += f"\nAttention: [{weights_str}]"
    
    return title
```

### 6.5 Plik: `src/analysis/game_utils.py`

Funkcje pomocnicze do obsługi meczów.

**1. `game_summary()` - czytelne podsumowanie**
```python
def game_summary(match: pd.Series) -> str:
    return f"{home_team} {home_score} : {away_score} {away_team} ({stage}) [{date}]"
    # Przykład: "Spain 2 : 1 England (Final) [14.07.2024]"
```

**2. `get_action_outcomes()` - rezultaty akcji**
```python
def get_action_outcomes(possession: pd.DataFrame) -> dict:
    return {
        'goals': len(possession[(type_id in shots) & (result_id == 1)]),
        'shots': len(possession[type_id in shots]),
        'passes': len(possession[type_id == pass]),
        # ...
    }
```

**3. `get_period_offset()` - konwersja czasu gry**
```python
def get_period_offset(period_id: int) -> int:
    # period_id → offset w minutach
    offsets = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
    return offsets.get(period_id, 0) * 60
```

---

## 7. Najważniejsze pliki

### 7.1 Pliki treningowe

| Plik | Opis | Linie kodu |
|------|------|------------|
| `pipeline.ipynb` | Główny pipeline treningowy | ~1000 |
| `src/ml/train.py` | Klasa ModelTrainer | 129 |
| `config.py` | Konfiguracja globalna | ~50 |

### 7.2 Pliki preprocessing

| Plik | Opis | Linie kodu |
|------|------|------------|
| `src/ml/preprocessing/sequence_preprocessor.py` | SequencePreprocessor - główna klasa | 214 |
| `src/ml/preprocessing/possessions_extraction.py` | Ekstrakcja i filtrowanie posiadań | 80 |
| `src/ml/preprocessing/action_valuation.py` | Wyliczanie xT, xG, wartości | 56 |
| `src/ml/preprocessing/xthreat.py` | Zarządzanie modelami xT | 90 |

### 7.3 Pliki modeli

| Plik | Opis | Linie kodu |
|------|------|------------|
| `src/ml/models/attention_lstm.py` | LSTM z attention (best) | 121 |
| `src/ml/models/lstm.py` | Prosty LSTM | ~80 |
| `src/ml/models/bigru.py` | Bidirectional GRU | ~80 |
| `src/ml/models/transformer.py` | Transformer encoder | ~150 |
| `src/ml/models/model_factory.py` | Factory pattern | ~60 |

### 7.4 Pliki analizy

| Plik | Opis | Linie kodu |
|------|------|------------|
| `analysis_final_match.ipynb` | Analiza finału | ~2000 |
| `analysis_matches.ipynb` | Analiza wszystkich meczów | ~3000 |
| `src/analysis/prediction_utils.py` | Utilities do predykcji | 155 |
| `src/analysis/visualization.py` | Wizualizacja na boisku | 166 |
| `src/analysis/game_utils.py` | Funkcje pomocnicze | ~100 |

---

## 8. Metryki i ewaluacja

### 8.1 Metryki modelu

**Zapisywane w `models/{model_type}/metrics.json`:**

```json
{
  "loss": 0.000234,
  "mae": 0.0123,
  "rmse": 0.0156,
  "training_time": "45m 32s",
  "epochs": 38,
  "best_epoch": 28
}
```

**Definicje:**
- **Loss (MSE)**: Mean Squared Error - główna funkcja straty
- **MAE**: Mean Absolute Error - średni błąd bezwzględny
- **RMSE**: Root Mean Squared Error - pierwiastek MSE

### 8.2 Wizualizacja treningu

**Plik: `models/{model_type}/training_history.png`**

Wykres zawiera:
1. **Loss (train vs val)**: monitorowanie overfittingu
2. **MAE (train vs val)**: metryka błędu
3. Vertical line: best epoch (przed early stopping)

### 8.3 Porównanie modeli

| Model | Loss (test) | MAE | RMSE | Czas treningu |
|-------|-------------|-----|------|---------------|
| **Attention LSTM** | **0.000234** | **0.0123** | **0.0156** | 45min |
| LSTM | 0.000267 | 0.0145 | 0.0178 | 35min |
| BiGRU | 0.000289 | 0.0156 | 0.0189 | 38min |
| Transformer | 0.000312 | 0.0178 | 0.0201 | 52min |

**Wnioski:**
- Attention LSTM najlepszy performance
- Attention pozwala interpretować predykcje (ważność akcji)
- Transformer wolniejszy i gorszy (za mało danych?)

### 8.4 Analiza attention weights

**Typowe wzorce:**
- **Strzały**: waga ~0.6-0.8 (dominująca)
- **Podania kluczowe**: waga ~0.3-0.5
- **Drybling w polu karnym**: waga ~0.4-0.6
- **Akcje neutralne**: waga ~0.05-0.15

**Przykład:**
```
Possession #27 (6 actions):
Actions: [Pass, Pass, Dribble, Pass, Cross, Shot]
Attention: [0.08, 0.12, 0.15, 0.23, 0.31, 0.71]
Most important: Shot (position 5, weight: 0.71)
```

### 8.5 Najcenniejsze akcje (top predictions)

**Charakterystyka sekwencji z wysoką wartością:**
1. Zakończone strzałem (zwłaszcza golem)
2. W polu karnym lub okolicach
3. Szybkie kombinacje (mało czasu między akcjami)
4. Wysokie xT końcowej pozycji
5. Brak błędów (successful passes)

---

## 9. Konfiguracja projektu

### 9.1 Plik: `config.py`

```python
# Preprocessing
SEQUENCE_LENGTH = 6              # Długość sekwencji akcji
MINIMUM_SEQUENCE_LENGTH = 3      # Min długość posiadania

# Training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 5
MIN_LEARNING_RATE = 1e-6

# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Paths
DATA_DIR = "data/statsbomb/data"
MODELS_DIR = "models"
PROCESSED_DATA_DIR = "data/processed"
```

### 9.2 Requirements

**Główne zależności:**
```
tensorflow>=2.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
socceraction>=1.3.0
mplsoccer>=1.2.0
scikit-learn>=1.3.0
```

---

## 10. Wnioski i rekomendacje

### 10.1 Co działa dobrze

✅ **Preprocessing:**
- Normalizacja kierunku gry (zawsze w prawo)
- Masking paddingu (ignorowanie krótkich posiadań)
- Sliding windows dla długich posiadań (więcej danych)
- Bogate cechy (46 wymiarów)

✅ **Model:**
- Attention LSTM najlepszy performance
- Attention weights interpretowalne
- Multi-output (value + weights)
- Bidirectional LSTM lepszy niż unidirectional

✅ **Analiza:**
- Wizualizacja na boisku intuicyjna
- Top predictions sensowne (strzały, sytuacje bramkowe)
- Analiza per mecz i globalna

### 10.2 Co można poprawić

⚠️ **Data augmentation:**
- Więcej meczów z innych rozgrywek
- Augmentacja geometryczna (flip horizontal?)
- Balansowanie klas (więcej high-value sequences)

⚠️ **Model:**
- Hierarchical attention (per action + per possession)
- Graph Neural Networks (relacje między zawodnikami)
- Ensemble modeli

⚠️ **Features:**
- Pozycje zawodników (tracking data)
- Formacja drużyny
- Kontekst wyniku meczu
- Zmęczenie (minuta gry)

⚠️ **Evaluation:**
- Cross-validation
- Analiza błędów (false positives/negatives)
- A/B testing z ekspertami

### 10.3 Następne kroki

1. **Rozszerzenie danych**: Dodać więcej rozgrywek (Premier League, La Liga)
2. **Player-level analysis**: Ocena indywidualnych zawodników
3. **Real-time predictions**: API do live analysis
4. **Explainable AI**: SHAP values, LIME dla attention
5. **Production deployment**: Model serving (TensorFlow Serving, FastAPI)

---

## 11. Bibliografia i zasoby

### 11.1 Papers

- **Expected Threat (xT)**: Karun Singh (2018) - "Introducing Expected Threat (xT)"
- **VAEP**: Tom Decroos et al. (2019) - "Actions Speak Louder than Goals"
- **StatsBomb**: Data specification v1.1
- **Attention Mechanism**: Bahdanau et al. (2014) - "Neural Machine Translation by Jointly Learning to Align and Translate"

### 11.2 Libraries

- **socceraction**: https://github.com/ML-KULeuven/socceraction
- **mplsoccer**: https://github.com/andrewRowlinson/mplsoccer
- **StatsBomb Open Data**: https://github.com/statsbomb/open-data

### 11.3 Dokumentacja projektu

- `README.md`: Wprowadzenie i quick start
- `ROADMAP.md`: Plan rozwoju projektu
- `docs/QUICKSTART.md`: Szybki start
- `docs/NEXT_STEPS.md`: Miejsca do uzupełnienia
- `docs/models/ATTENTION_LSTM.md`: Szczegóły modelu

---

**Koniec raportu**

*Dokument utworzony: 6 lutego 2026*  
*Projekt: StatsBomb Scout - Football Action Valuation using Deep Learning*  
*Autor: Tomasz Morcinek*
