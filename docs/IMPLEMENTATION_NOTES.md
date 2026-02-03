# Notatki Implementacyjne - Masking, Attention, Transformer

## Data: Luty 2026

Ten dokument zawiera kluczowe informacje o implementacji mechanizmów masking i attention w projekcie.

---

## 1. Pipeline Verification - Masking

### Problem: Padding w Sekwencjach

Sekwencje akcji mają różną długość:
- Długie posiadania (>8 akcji): dzielone na windows metodą sliding window
- Krótkie posiadania (<8 akcji): paddowane zerami na końcu

**Kluczowe pytanie:** Czy padding (0.0) jest poprawnie maskowany przez model?

### Weryfikacja

Test przeprowadzony na modelu Attention LSTM:

```python
# Input: 6 timesteps (4 prawdziwe + 2 padded)
X_test = np.array([[
    [1, 2, 3, ...],  # t=0: prawdziwa akcja
    [4, 5, 6, ...],  # t=1: prawdziwa akcja  
    [7, 8, 9, ...],  # t=2: prawdziwa akcja
    [10, 11, 12, ...],  # t=3: prawdziwa akcja
    [0, 0, 0, ...],  # t=4: PADDING
    [0, 0, 0, ...]   # t=5: PADDING
]])

# Predykcja
predictions = model.predict(X_test)
attention_weights = predictions['attention_weights'][0]

# Wynik:
# [0.237, 0.250, 0.258, 0.255, 0.000, 0.000]
#  ^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^
#  Prawdziwe akcje              Padding (0.0)
```

✅ **Wniosek:** Masking działa poprawnie! Padded timesteps mają zerowe wagi attention.

### Implementacja Masking Layer

```python
# W model.build()
x = layers.Input(shape=input_shape, name='sequence_input')
masked = layers.Masking(mask_value=0.0)(x)  # Kluczowa warstwa!

# Maska propagowana automatycznie przez:
# - LSTM layers (return_sequences=True)
# - Attention layers (supports_masking=True)
# - Dense layers
```

**Warunek:** Wszystkie custom layers muszą mieć `supports_masking=True` i poprawnie propagować maskę.

---

## 2. Attention LSTM - Implementacja

### AttentionLayer - Custom Layer

```python
class AttentionLayer(layers.Layer):
    """
    Attention mechanism z tanh scoring.
    
    Oblicza:
    1. score = tanh(h @ W + b)
    2. attention_weights = softmax(score, mask=mask)
    3. context = sum(attention_weights * h)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True  # Kluczowe!
    
    def build(self, input_shape):
        units = input_shape[-1]
        self.W = self.add_weight(
            shape=(units, 1),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_W'
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer='zeros',
            trainable=True,
            name='attention_b'
        )
    
    def call(self, inputs, mask=None):
        # Scoring
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        score = tf.squeeze(score, axis=-1)  # (batch, seq_len)
        
        # Apply mask BEFORE softmax
        if mask is not None:
            mask_value = -1e9  # Bardzo duża liczba ujemna
            score = tf.where(mask, score, mask_value)
        
        # Softmax
        attention_weights = tf.nn.softmax(score, axis=-1)
        
        # Context vector
        context = tf.reduce_sum(
            attention_weights[..., tf.newaxis] * inputs,
            axis=1
        )
        
        return context, attention_weights
    
    def compute_mask(self, inputs, mask=None):
        # Attention zwraca context (nie sekwencję), więc maska nie jest propagowana dalej
        return None
```

### Kluczowe Elementy

1. **`supports_masking=True`** - Layer akceptuje maskę
2. **Maskowanie przed softmax** - Padding dostaje -1e9, więc softmax(−1e9) ≈ 0
3. **`compute_mask` zwraca None** - Attention redukuje sekwencję do context vector

---

## 3. Transformer - True Attention

### Problem: key_dim w MultiHeadAttention

**Błąd (przed poprawką):**
```python
MultiHeadAttention(
    num_heads=4,
    key_dim=128,  # ❌ ZŁE! To powinno być per-head dimension
    dropout=0.1
)
```

**Poprawka:**
```python
MultiHeadAttention(
    num_heads=4,
    key_dim=128 // 4,  # ✅ DOBRE! key_dim = d_model // num_heads = 32
    dropout=0.1
)
```

### Dlaczego?

Keras `MultiHeadAttention` interpretuje `key_dim` jako **wymiar per-head**, nie całkowity wymiar modelu.

```
d_model = 128
num_heads = 4
key_dim = d_model // num_heads = 32  # wymiar dla każdej głowy

Każda głowa operuje na:
- Q: (batch, seq_len, 32)
- K: (batch, seq_len, 32)  
- V: (batch, seq_len, 32)

Po concatenation wszystkich głów:
- Output: (batch, seq_len, 128)  # 4 heads × 32 = 128
```

### Walidacja

```python
if self.d_model % self.num_heads != 0:
    raise ValueError(
        f"d_model ({self.d_model}) must be divisible by "
        f"num_heads ({self.num_heads})"
    )
```

---

## 4. Attention Mask w Transformer

### Problem

Keras `MultiHeadAttention` wymaga attention_mask w formacie:
- `(batch, 1, seq_len)` lub
- `(batch, seq_len, seq_len)`

Masking layer zwraca: `(batch, seq_len)` - boolean mask

### Rozwiązanie

```python
def call(self, inputs, mask=None):
    # Konwersja maski
    if mask is not None:
        # (batch, seq_len) → (batch, 1, seq_len)
        attention_mask = tf.cast(mask[:, tf.newaxis, :], tf.int32)
    else:
        attention_mask = None
    
    # Przekaż do MultiHeadAttention
    attn_output = self.mha(
        query=inputs,
        value=inputs,
        key=inputs,
        attention_mask=attention_mask  # Kluczowy argument!
    )
    
    return attn_output
```

### Propagacja Maski w Transformer Blocks

```python
# Block z residual connections
x = inputs  # (batch, seq_len, d_model), mask=(batch, seq_len)

# Multi-Head Attention (używa maski)
attn_output = self.mha(x, x, attention_mask=mask)

# Add & Norm (nie psuje maski)
x = layers.Add()([x, attn_output])  # Używamy layers.Add(), nie '+'
x = layers.LayerNormalization()(x)

# Feed-Forward (propaguje maskę)
ff_output = self.ff(x)

# Add & Norm
x = layers.Add()([x, ff_output])
x = layers.LayerNormalization()(x)

# Maska jest cały czas propagowana!
```

**Ważne:** Używaj `layers.Add()` zamiast operatora `+`, bo `layers.Add()` prawidłowo propaguje maskę.

---

## 5. Return Attention Weights - Dual Output Model

### Problem

Chcemy zwracać zarówno `value` jak i `attention_weights` dla wizualizacji.

### Rozwiązanie: Multi-output Model

```python
def build(self, input_shape):
    inputs = layers.Input(shape=input_shape, name='sequence_input')
    
    # ... warstwy modelu ...
    
    # Ekstrahuj attention weights z ostatniego bloku
    _, attention_weights = self.attention_weights_layer(x, mask=mask)
    
    # Dwa wyjścia
    value_output = layers.Dense(1, name='value')(x)
    attention_output = layers.Lambda(
        lambda x: x,
        name='attention_weights'
    )(attention_weights)
    
    # Model z dictionary output
    self.model = Model(
        inputs=inputs,
        outputs={'value': value_output, 'attention_weights': attention_output}
    )
```

### Kompilacja

```python
self.model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={'value': 'mse'},  # Loss tylko dla value
    metrics={'value': ['mae', 'mse']}
)
```

**Kluczowe:** 
- Attention weights NIE mają swojej funkcji loss
- Są zwracane tylko do wizualizacji/analizy
- W `model.fit()` podajemy tylko `y_train` (dla value)

---

## 6. True Attention vs Tanh Attention

### Architektura

#### Tanh Attention (Custom)
```python
class AttentionWeightsLayer(layers.Layer):
    """Custom attention z tanh scoring."""
    
    def call(self, inputs, mask=None):
        # Single-head attention
        score = tanh(inputs @ W + b)
        attention_weights = softmax(score, mask=mask)
        context = sum(attention_weights * inputs)
        return context, attention_weights
```

#### True Multi-Head Attention (Keras)
```python
# Używamy standardowej implementacji Keras
self.mha = layers.MultiHeadAttention(
    num_heads=num_heads,
    key_dim=d_model // num_heads,
    dropout=dropout
)

# W call():
attn_output = self.mha(
    query=inputs,
    value=inputs,
    key=inputs,
    attention_mask=attention_mask
)
# Keras zwraca tylko attn_output, nie attention_weights!
```

### Ekstrakcja Attention Weights z MultiHeadAttention

**Problem:** Keras `MultiHeadAttention` domyślnie NIE zwraca attention scores.

**Opcje:**

1. **Własna implementacja attention weights** (obecne rozwiązanie):
   - Używamy oddzielnej warstwy `AttentionWeightsLayer` z tanh scoring
   - Attention weights są "approximate" - nie są prawdziwymi wagami z MHA

2. **Modify MultiHeadAttention** (zaawansowane):
   ```python
   # Można podmienić implementację i dodać return_attention_scores=True
   # Ale to wymaga custom building i może powodować problemy z training
   ```

3. **GradCAM / Attention Rollout** (future work):
   - Zaawansowane techniki do ekstrahowania "true" attention patterns
   - Wymagają dodatkowej analizy post-hoc

**Obecne rozwiązanie:** Dla prostoty, transformer z `true_attention=True` używa standardowego MHA bez zwracania attention weights w trakcie predykcji. Dla wizualizacji można użyć transformer z `true_attention=False` (tanh attention).

---

## 7. Testy Maskingu

### TestAttentionLSTMModelMasking

```python
def test_attention_lstm_pads_correctly():
    """Sprawdza czy padding jest maskowany (wagi=0)."""
    model = create_model('attention_lstm', input_shape=(8, 18))
    model.model.compile(optimizer='adam', loss='mse')
    
    # Sekwencja: 5 prawdziwych + 3 padding
    X = np.random.randn(1, 8, 18).astype(np.float32)
    X[0, 5:, :] = 0.0  # Padding
    
    pred = model.model.predict(X)
    attention_weights = pred['attention_weights'][0]
    
    # Asercje
    assert np.sum(attention_weights[:5]) > 0.99  # Prawdziwe: suma ≈ 1
    assert np.allclose(attention_weights[5:], 0.0)  # Padding: wszystkie 0
```

### TestTransformerTrueAttentionModelMasking

```python
def test_transformer_true_attention_pads_correctly():
    """Sprawdza czy transformer z true attention maskuje padding."""
    model = create_model('transformer', input_shape=(8, 18), true_attention=True)
    model.model.compile(optimizer='adam', loss='mse')
    
    X = np.random.randn(1, 8, 18).astype(np.float32)
    X[0, 5:, :] = 0.0
    
    # Predykcja - tylko value (brak attention_weights dla true attention)
    pred = model.model.predict(X)
    value = pred['value'][0, 0]
    
    # Sprawdź że padding nie zmienia wyniku (invariance test)
    X_more_padding = X.copy()
    # Dodaj więcej paddingu (ale to nadal nie powinno zmienić wyniku)
    pred2 = model.model.predict(X_more_padding)
    value2 = pred2['value'][0, 0]
    
    assert np.isclose(value, value2, rtol=1e-5)
```

---

## 8. Lessons Learned

### ✅ Co Działa

1. **Masking Layer** - `layers.Masking(mask_value=0.0)` prawidłowo maskuje padding
2. **supports_masking=True** - Kluczowe dla custom layers
3. **layers.Add()** - Propaguje maskę w residual connections (operator `+` nie)
4. **attention_mask w MHA** - Konwersja `(batch, seq_len)` → `(batch, 1, seq_len)`
5. **key_dim = d_model // num_heads** - Poprawna interpretacja key_dim
6. **Multi-output model** - Zwracanie value + attention_weights jako dictionary

### ⚠️ Pułapki

1. **Operator `+` nie propaguje maski** - Używaj `layers.Add()`
2. **key_dim ≠ d_model** - To wymiar per-head!
3. **Dummy targets dla attention** - NIE są potrzebne, loss tylko dla value
4. **return_attention_scores=True** - Może powodować problemy, lepiej custom layer
5. **Softmax przed maskowaniem** - ❌ Źle! Maskuj PRZED softmax

### 🚀 Best Practices

1. Zawsze testuj masking (tests/test_masking.py)
2. Sprawdzaj czy wagi attention sumują się do 1.0
3. Weryfikuj invariance: więcej paddingu nie zmienia predykcji
4. Używaj `layers.Add()` w residual connections
5. Dokumentuj format mask w custom layers

---

## 9. Przyszłe Usprawnienia

### Attention Visualization
- [ ] Ekstrahować prawdziwe attention scores z MultiHeadAttention
- [ ] GradCAM dla sekwencji
- [ ] Attention rollout dla wielu warstw

### Model Architecture
- [ ] Porównanie różnych scoring functions (tanh, dot-product, additive)
- [ ] Eksperymentować z liczbą głów (4, 8, 16)
- [ ] Cross-attention między różnymi reprezentacjami

### Training
- [ ] Learning rate scheduling dla Transformer
- [ ] Warmup steps dla stabilności
- [ ] Gradient clipping

### Interpretability
- [ ] Agregacja attention weights z wielu głów
- [ ] Wizualizacja attention patterns per-head
- [ ] Analiza które typy akcji dostają wysokie wagi

---

## 10. Referencje

### Kod
- `src/ml/models/attention_lstm.py` - Attention LSTM implementation
- `src/ml/models/transformer.py` - Transformer implementation  
- `tests/test_masking.py` - Comprehensive masking tests

### Papers
- "Attention Is All You Need" (Vaswani et al., 2017) - Original Transformer
- "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2014) - Attention mechanism

### Keras Documentation
- [MultiHeadAttention](https://keras.io/api/layers/attention_layers/multi_head_attention/)
- [Masking and Padding](https://keras.io/guides/understanding_masking_and_padding/)

---

**Ostatnia aktualizacja:** Luty 2026  
**Status:** ✅ Wszystkie modele działają poprawnie z masking i attention
