# Mechanizm Attention w Modelach Sekwencyjnych

## Wprowadzenie

Mechanizm **attention** pozwala modelowi dynamicznie skupić się na najważniejszych akcjach w sekwencji podczas predykcji wartości posiadania piłki. Zamiast traktować wszystkie akcje równo, model **uczy się** które akcje są kluczowe dla wyniku.

---

## Modele z Attention w Projekcie

### 1. Attention LSTM
Model LSTM rozszerzony o warstwę attention z **tanh scoring**.

### 2. Transformer (Tanh Attention)
Model transformer z custom attention layer używającym **tanh scoring**.

### 3. Transformer (True Attention)  
Model transformer z **prawdziwą multi-head attention** z Keras (`MultiHeadAttention`).

---

## Jak Działa Attention?

### Podstawowa Idea

```
Sekwencja akcji:     [a1, a2, a3, a4, a5]
                       ↓   ↓   ↓   ↓   ↓
Hidden states:       [h1, h2, h3, h4, h5]
                       ↓   ↓   ↓   ↓   ↓
Attention weights:   [0.1, 0.3, 0.4, 0.15, 0.05]  <- suma = 1.0
                       ↓   ↓   ↓   ↓   ↓
Context vector:      weighted sum of hidden states
                       ↓
Output:              predicted value
```

**Interpretacja:** Akcja `a3` ma największą wagę (0.4), więc model uważa ją za najważniejszą dla predykcji.

---

## Attention LSTM - Szczegóły Implementacji

### Architektura

```
Input: (batch_size, sequence_length, n_features)
    ↓
Masking Layer (mask_value=0.0)
    ↓
Bidirectional LSTM
    ↓
Dropout
    ↓
LSTM (return_sequences=True) → hidden_states: (batch, seq_len, units)
    ↓
Dropout
    ↓
Attention Layer
    ├─→ Scoring: e_t = tanh(h_t · W + b)
    ├─→ Softmax: α_t = exp(e_t) / Σ exp(e_i)
    ├─→ Masking: α_t = 0 for padded positions
    └─→ Context: c = Σ(α_t · h_t)
    ↓
Dense Layers → value prediction
```

### Scoring Function (Tanh)

```python
# Dla każdego timestep t:
score_t = tanh(h_t @ W + b)  # W: (lstm_units, 1), b: (1,)

# Softmax z maskowaniem:
attention_weights = softmax(scores, mask=mask)
# Padded positions mają wagę = 0.0

# Context vector:
context = sum(attention_weights[t] * hidden_states[t] for t in range(seq_len))
```

### Kluczowe Właściwości

✅ **Learnable weights:** Macierz `W` i bias `b` są uczone podczas treningu  
✅ **Masking support:** Padded timesteps automatycznie dostają wagę 0.0  
✅ **Interpretability:** Wagi attention pokazują które akcje są ważne  
✅ **Suma = 1.0:** Wagi po softmax sumują się do 1.0 (dystrybucja prawdopodobieństwa)

---

## Transformer - Szczegóły Implementacji

### Architektura

```
Input: (batch_size, sequence_length, n_features)
    ↓
Masking Layer (mask_value=0.0)
    ↓
Positional Encoding (learned)
    ↓
Transformer Block × N:
    ├─→ Multi-Head Attention
    │   ├─ Query, Key, Value projections
    │   ├─ Scaled dot-product attention
    │   └─ Masking support
    ├─→ Add & LayerNorm (residual connection)
    ├─→ Feed-Forward Network (Dense + ReLU + Dense)
    └─→ Add & LayerNorm
    ↓
Global Average Pooling (masked)
    ↓
Dense Layers → value prediction
```

### True Multi-Head Attention

```python
# Keras MultiHeadAttention
attention_output = MultiHeadAttention(
    num_heads=4,
    key_dim=d_model // num_heads,  # per-head dimension
    dropout=0.1
)(query, value, attention_mask=mask)

# Wewnętrznie oblicza:
# 1. Q = query @ W_q,  K = key @ W_k,  V = value @ W_v
# 2. scores = (Q @ K^T) / sqrt(d_k)
# 3. scores = mask_fill(scores, mask, -inf)  # dla paddingu
# 4. attention = softmax(scores, dim=-1)
# 5. output = attention @ V
```

### Różnice: Tanh vs True Attention

| Właściwość | Tanh Attention | True Multi-Head Attention |
|------------|----------------|---------------------------|
| **Scoring** | `tanh(h·W+b)` | `(Q·K^T) / sqrt(d_k)` |
| **Liczba głów** | 1 (single-head) | Wiele (4-8) |
| **Projekcje** | Jedna macierz W | Osobne Q, K, V dla każdej głowy |
| **Skalowalność** | Mniej parametrów | Więcej parametrów |
| **Różnorodność** | Jedna perspektywa | Wiele perspektyw (heads) |
| **Standard** | Custom | Standardowa implementacja |

---

## Masking i Padding

### Problem

Sekwencje mają różną długość:
```
Sekwencja 1: [a1, a2, a3, a4, a5, a6, a7, a8]  ← 8 akcji
Sekwencja 2: [b1, b2, b3, b4, b5, 0, 0, 0]     ← 5 akcji + 3 padding
```

Bez maskingu, model traktowałby padding (0.0) jako prawdziwe akcje!

### Rozwiązanie

```python
# 1. Masking Layer
mask = Masking(mask_value=0.0)(inputs)
# Tworzy maskę: [True, True, True, True, True, False, False, False]

# 2. Propagacja maski przez warstwy
# LSTM, Attention, itp. automatycznie używają tej maski

# 3. Attention z maską
scores = calculate_scores(hidden_states)  # (batch, seq_len)
scores = mask_fill(scores, mask, -inf)    # Padding dostaje -inf
attention_weights = softmax(scores)        # Padding → 0.0 po softmax
```

### Weryfikacja Maskingu

Test dla sekwencji z 5 prawdziwymi akcjami + 3 padding:

```python
Input shape: (1, 8, 18)
# Pierwsze 5 timesteps: prawdziwe dane
# Ostatnie 3 timesteps: padding (0.0)

Attention weights:
[0.237, 0.250, 0.258, 0.255, 0.000, 0.000, 0.000, 0.000]
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^
 Prawdziwe akcje (suma=1.0)    Padding (zawsze 0.0)
```

✅ **Poprawne zachowanie:** Padding jest całkowicie ignorowany!

---

## Parametry Modeli

### Attention LSTM

```python
ModelConfig(
    name='attention_lstm_baseline',
    model_type='attention_lstm',
    model_params={
        'lstm_units': 64,      # Liczba jednostek LSTM
        'dropout': 0.2,        # Dropout rate
        'dense_units': 64,     # Jednostki w Dense layer
    },
    training_params={
        'batch_size': 32,
        'epochs': 100
    }
)
```

### Transformer (Tanh Attention)

```python
ModelConfig(
    name='transformer_baseline',
    model_type='transformer',
    model_params={
        'num_heads': 4,           # Liczba głów attention
        'd_model': 128,           # Wymiar embeddings
        'ff_dim': 512,            # Wymiar feed-forward
        'num_blocks': 2,          # Liczba bloków transformer
        'dropout': 0.1,
        'true_attention': False   # Użyj tanh attention
    },
    training_params={
        'batch_size': 32,
        'epochs': 100
    }
)
```

### Transformer (True Attention)

```python
ModelConfig(
    name='transformer_true_attention',
    model_type='transformer',
    model_params={
        'num_heads': 4,
        'd_model': 128,
        'ff_dim': 512,
        'num_blocks': 2,
        'dropout': 0.1,
        'true_attention': True    # Użyj prawdziwej multi-head attention
    },
    training_params={
        'batch_size': 32,
        'epochs': 100
    }
)
```

---

## Interpretacja Wag Attention

### Przykład - Analiza Najlepszych Posiadań

Po wytrenowaniu modelu możemy wyekstrahować wagi attention dla najlepszych sekwencji:

```python
# Predykcja z attention weights
predictions = model.predict(X_test)
values = predictions['value']              # (n_samples, 1)
attention_weights = predictions['attention_weights']  # (n_samples, seq_len)

# Znajdź top 10 najwyżej ocenionych sekwencji
top_indices = np.argsort(values.flatten())[-10:]

# Dla każdej sekwencji
for idx in top_indices:
    sequence = p[idx]  # DataFrame z akcjami
    weights = attention_weights[idx]  # Wagi dla każdej akcji
    
    # Znajdź najważniejsze akcje
    important_actions = np.argsort(weights)[-3:]  # Top 3
    
    print(f"Sekwencja {idx}: wartość = {values[idx]:.4f}")
    for action_idx in important_actions:
        action = sequence.iloc[action_idx]
        weight = weights[action_idx]
        print(f"  Akcja {action_idx}: waga={weight:.3f}, typ={action['type_name']}")
```

### Wizualizacja

Funkcja `visualize_top_sequences()` rysuje sekwencje z kolorowaniem według wag attention:
- 🔴 **Czerwony** = wysokie wagi (ważne akcje)
- 🔵 **Niebieski** = niskie wagi (mniej ważne akcje)

To pozwala **wizualnie** zobaczyć które akcje model uważa za kluczowe!

---

## Porównanie Modeli

### Complexity

| Model | Parametry | Czas treningu | Interpretability |
|-------|-----------|---------------|------------------|
| LSTM baseline | ~50K | Najszybszy | ❌ Brak attention |
| Attention LSTM | ~60K | Szybki | ✅ Single-head |
| Transformer (tanh) | ~100K | Średni | ✅ Multi-head (custom) |
| Transformer (true) | ~120K | Wolniejszy | ✅✅ Multi-head (standard) |

### Kiedy Użyć Którego?

- **LSTM baseline:** Quick baseline, brak potrzeby interpretacji
- **Attention LSTM:** Dobry balans performance/interpretability, prostszy niż transformer
- **Transformer (tanh):** Eksperymentalna wersja, mniej parametrów niż true attention
- **Transformer (true):** State-of-the-art, najlepsza wydajność, standardowa implementacja

---

## Testy Maskingu

Projekt zawiera comprehensive testy maskingu dla wszystkich modeli:

```bash
# Testy dla Attention LSTM
pytest tests/test_masking.py::TestAttentionLSTMModelMasking -v

# Testy dla Transformer
pytest tests/test_masking.py::TestTransformerTrueAttentionModelMasking -v
```

### Co Jest Testowane?

1. ✅ Padding jest maskowany (wagi = 0.0)
2. ✅ Różne długości sekwencji dają różne wyniki
3. ✅ Invariance: padding na końcu nie zmienia predykcji
4. ✅ Wagi attention sumują się do 1.0

---

## Referencje

### Kod

- `src/ml/models/attention_lstm.py` - Implementacja Attention LSTM
- `src/ml/models/transformer.py` - Implementacja Transformer
- `src/ml/models/model_factory.py` - Factory do tworzenia modeli
- `tests/test_masking.py` - Testy maskingu

### Trenowanie

```bash
# Wytrenuj wszystkie modele (LSTM, Attention LSTM, Transformer)
python train_models.py

# Modele zapisane w:
# - models/lstm_baseline_*/
# - models/attention_lstm_baseline_*/
# - models/transformer_baseline_*/
# - models/transformer_true_attention_*/
```

### Analiza

```bash
# Notebooki z analizą
jupyter notebook analysis_matches.ipynb      # Analiza wielu meczów
jupyter notebook analysis_final_match.ipynb  # Analiza jednego meczu
```

---

## Podsumowanie

✅ **Attention LSTM** - Prosty i efektywny mechanizm attention  
✅ **Transformer** - State-of-the-art architektura z multi-head attention  
✅ **Masking** - Prawidłowa obsługa paddingu w obu modelach  
✅ **Interpretability** - Wagi attention pokazują ważność akcji  
✅ **Testowanie** - Comprehensive testy zapewniają poprawność  

Mechanizm attention sprawia, że modele są nie tylko **dokładniejsze**, ale też **interpretowalene** - możemy zobaczyć **dlaczego** model przypisał daną wartość sekwencji!
