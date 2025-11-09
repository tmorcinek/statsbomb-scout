# Attention LSTM Model

## Opis

Model **Attention-based LSTM** to rozszerzenie standardowego modelu LSTM, które dodatkowo zwraca **wagi attention** pokazujące względną ważność każdej akcji w sekwencji.

## Architektura

```
Input (sequence_length, n_features)
    ↓
Bidirectional LSTM (128 units)
    ↓
Dropout (0.2)
    ↓
LSTM (128 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
Attention Layer
    ├─→ Context Vector → Dense(64) → Dense(1) → VALUE
    └─→ ATTENTION WEIGHTS (suma = 1.0)
```

## Mechanizm Attention

Warstwa attention oblicza wagi dla każdej akcji w sekwencji:

1. **Scoring**: Dla każdego hidden state oblicza score: `e_t = tanh(h_t * W + b)`
2. **Normalizacja**: Stosuje softmax, aby wagi sumowały się do 1.0: `α_t = softmax(e_t)`
3. **Context**: Oblicza ważoną sumę: `context = Σ(α_t * h_t)`

**Kluczowe**: Wagi są **uczone automatycznie** podczas treningu - model sam dowiaduje się, które akcje są ważniejsze dla predykcji wartości.

## Wyjścia Modelu

Model zwraca **dwa wyjścia**:

1. **value** (shape: `(batch_size, 1)`): Predykowana wartość sekwencji
2. **attention_weights** (shape: `(batch_size, sequence_length)`): Wagi attention dla każdej akcji
   - Wagi sumują się do 1.0
   - Wyższa waga = większy wkład akcji w końcową wartość

## Użycie

### 1. Trening

W pliku `config.py` ustaw:
```python
MODEL_TYPE = 'attention_lstm'
```

Następnie uruchom:
```bash
python main.py
```

Model będzie zapisany w: `models/attention_lstm/best_model.h5`

**Uwaga**: Kod modelu znajduje się w: `src/models/attention_lstm.py`

### 2. Predykcja z wagami attention

```python
from tensorflow import keras
import numpy as np

# Wczytaj wytrenowany model
model = keras.models.load_model('models/attention_lstm/best_model.h5')

# Przygotuj dane (z preprocessing)
X = preprocessor.process_matches(matches)

# Predykcja
predictions = model.predict(X)

# Wyciągnij wartości i wagi
values = predictions['value']  # (n_samples, 1)
weights = predictions['attention_weights']  # (n_samples, sequence_length)

# Analiza pierwszej sekwencji
print(f"Predicted value: {values[0][0]:.4f}")
print(f"Attention weights: {weights[0]}")
print(f"Sum of weights: {weights[0].sum():.4f}")  # = 1.0

# Znajdź najważniejsze akcje
top_3_indices = np.argsort(weights[0])[-3:][::-1]
print(f"Most important actions: {top_3_indices}")
```

### 3. Przykład użycia

Zobacz plik `src/models/example_attention_usage.py` dla kompletnego przykładu.

```bash
python -m src.models.example_attention_usage
```

## Interpretacja Wag Attention

Wagi attention pokazują, które akcje w sekwencji były najważniejsze dla osiągnięcia końcowej wartości:

**Przykład**: Sekwencja 10 akcji prowadząca do gola
```
Action 1: 0.04  ████           (rozegranie z tyłu)
Action 2: 0.05  █████          (podanie boczne)
Action 3: 0.08  ████████       (podanie w przód)
Action 4: 0.12  ████████████   (drybling)
Action 5: 0.06  ██████         (krótkie podanie)
Action 6: 0.22  ██████████████████████  (kluczowe podanie)
Action 7: 0.18  ██████████████████      (przyjęcie piłki)
Action 8: 0.15  ███████████████         (drybling w polu karnym)
Action 9: 0.08  ████████       (ustawienie)
Action 10: 0.02 ██             (strzał - niska waga bo xG=1.0)
```

**Interpretacja**:
- Action 6 (waga 0.22): Kluczowe podanie - największy wkład
- Action 7-8 (wagi 0.18, 0.15): Przygotowanie do strzału
- Action 1-2 (wagi 0.04, 0.05): Rozegranie - mały wkład
- Action 10 (waga 0.02): Strzał ma niską wagę, bo sam w sobie ma xG=1.0

## Zalety Modelu

1. **Interpretowalność**: Możesz zobaczyć, które akcje model uznał za ważne
2. **Analityka**: Identyfikacja kluczowych momentów w akcjach ofensywnych
3. **Scouting**: Wykrywanie graczy, którzy wykonują kluczowe akcje (wysokie wagi)
4. **Porównywalność**: Możesz porównać wagi między różnymi sekwencjami

## Parametry

W pliku `config.py`:

```python
LSTM_UNITS = 128        # Liczba jednostek LSTM
LSTM_DROPOUT = 0.2      # Dropout rate
SEQUENCE_LENGTH = 10    # Długość sekwencji (liczba akcji)
```

## Różnice vs Standardowy LSTM

| Feature | LSTM | Attention LSTM |
|---------|------|----------------|
| Wyjście | Tylko wartość | Wartość + wagi |
| Interpretowalność | Niska | Wysoka |
| Liczba parametrów | Mniejsza | Większa |
| Czas treningu | Krótszy | Dłuższy |
| Use case | Predykcja | Predykcja + Analiza |

## Porady

1. **Dla analizy**: Używaj `return_attention=True` (domyślnie)
2. **Dla produkcji**: Jeśli nie potrzebujesz wag, ustaw `return_attention=False` (szybsze)
3. **Wizualizacja**: Narysuj heatmapę wag attention dla całego meczu
4. **Walidacja**: Sprawdź czy wagi mają sens (np. podanie przed golem ma wysoką wagę)

## Dalszy Rozwój

Możliwe rozszerzenia:
- **Multi-head attention**: Wiele "perspektyw" ważności akcji
- **Self-attention**: Każda akcja "patrzy" na inne akcje
- **Hierarchical attention**: Attention na poziomie posiadań i akcji

