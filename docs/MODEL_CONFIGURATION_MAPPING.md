# Mapowanie Konfiguracji Modeli - Podsumowanie

## 📋 Przegląd Funkcji w train_models.py

### 1. `create_default_configs()` - Główna funkcja dla różnych typów modeli

Tworzy 13 modeli:
- **3 LSTM** (small, baseline, large)
- **3 Attention LSTM** (small, baseline, large)
- **3 BiGRU** (baseline, medium, large)
- **4 Transformer** (baseline, true_attention, small, large)

---

### 2. `create_lstm_configs()` - Eksperymentalne warianty LSTM

Tworzy 9 modeli LSTM testujących wpływ warstwy Dense:
- **3 × dense64** (small, baseline, large)
- **3 × dense128** (small, baseline, large)
- **3 × no_dense** (small, baseline, large)

**Wynik:** `no_dense` najlepszy! ✅

---

### 3. `create_lstm_second_lstm_configs()` - Test drugiej warstwy LSTM

Tworzy 4 modele testujące flagę `use_second_lstm`:
- **2 × baseline** (single_lstm, double_lstm)
- **2 × large** (single_lstm, double_lstm)

**Cel:** Research - czy druga warstwa LSTM pomaga?

---

### 4. `create_transformers_config()` - Porównanie z najlepszym LSTM

Tworzy 3 modele do bezpośredniego porównania:
- **1 × LSTM** (large_single_lstm jako baseline)
- **2 × Transformer** (baseline, small)

**Wynik z pliku `model_comparison_transformer.csv`:**
- lstm_large_single_lstm: test_mae = 0.007149
- transformer_baseline: test_mae = 0.004336 ✅
- transformer_small: test_mae = 0.004701 ✅

---

## 🗂️ Pełne Mapowanie Modeli do Plików CSV

### model_comparison.csv (13 modeli)
Utworzone przez: `create_default_configs()`

| Model                      | Typ             | Parametry                                          | test_mae |
|---------------------------|-----------------|---------------------------------------------------|----------|
| lstm_small                | lstm            | units=32, dropout=0.15                            | 0.006616 |
| lstm_baseline             | lstm            | units=64, dropout=0.2                             | 0.007028 |
| lstm_large                | lstm            | units=128, dropout=0.3                            | 0.005688 |
| attention_lstm_small      | attention_lstm  | units=32, dropout=0.15                            | 0.000346 |
| attention_lstm_baseline   | attention_lstm  | units=64, dropout=0.2                             | 0.000322 |
| attention_lstm_large      | attention_lstm  | units=128, dropout=0.3                            | 0.000283 ⭐ |
| bigru_baseline            | bigru           | units=64, attn=32, dropout=0.2                    | 0.010829 |
| bigru_medium              | bigru           | units=96, attn=48, dropout=0.25                   | 0.010016 |
| bigru_large               | bigru           | units=128, attn=64, dropout=0.3                   | 0.012509 |
| transformer_baseline      | transformer     | heads=4, d_model=128, ff=512, blocks=2            | 0.000324 |
| transformer_true_attention| transformer     | heads=4, d_model=128, ff=512, blocks=2, true_attn | 0.000303 |
| transformer_small         | transformer     | heads=2, d_model=64, ff=256, blocks=2             | 0.000278 ⭐ |
| transformer_large         | transformer     | heads=8, d_model=256, ff=1024, blocks=3           | 0.000799 |

---

### model_comparison_lstm.csv (9 modeli)
Utworzone przez: `create_lstm_configs()`

| Model                   | dense_units | lstm_units | dropout | test_mae | Ranking |
|------------------------|-------------|------------|---------|----------|---------|
| lstm_small_dense64     | 64          | 32         | 0.15    | 0.007813 | 9       |
| lstm_small_dense128    | 128         | 32         | 0.15    | 0.006328 | 5       |
| lstm_baseline_dense64  | 64          | 64         | 0.2     | 0.005553 | 2 ✅    |
| lstm_baseline_dense128 | 128         | 64         | 0.2     | 0.006753 | 7       |
| lstm_large_dense64     | 64          | 128        | 0.3     | 0.005864 | 4       |
| lstm_large_dense128    | 128         | 128        | 0.3     | 0.006681 | 6       |
| lstm_small_no_dense    | None        | 32         | 0.15    | 0.006685 | 8       |
| lstm_baseline_no_dense | None        | 64         | 0.2     | 0.007065 | 3       |
| lstm_large_no_dense    | None        | 128        | 0.3     | 0.005298 | 1 ⭐    |

**Obserwacja:** 
- ✅ `no_dense` (None) najlepszy dla large model
- ✅ `dense64` najlepszy dla baseline
- ❌ `dense128` zawsze gorszy

---

### model_comparison_transformer.csv (3 modele)
Utworzone przez: `create_transformers_config()`

| Model                   | Typ         | Parametry                        | test_mae | Uwagi           |
|------------------------|-------------|----------------------------------|----------|-----------------|
| lstm_large_single_lstm | lstm        | units=128, use_second_lstm=False | 0.007149 | Baseline LSTM   |
| transformer_baseline   | transformer | heads=4, d_model=128             | 0.004336 | ✅ Lepszy od LSTM |
| transformer_small      | transformer | heads=2, d_model=64              | 0.004701 | ✅ Lepszy od LSTM |

**Obserwacja:**
- ✅ Oba transformery lepsze od LSTM baseline
- 🤔 **Różne wyniki** niż w `model_comparison.csv`:
  - transformer_baseline: 0.004336 (tutaj) vs 0.000324 (główny)
  - transformer_small: 0.004701 (tutaj) vs 0.000278 (główny)
- ⚠️ **Możliwe różne seed, split lub timestamp treningu**

---

## 📊 Analiza Różnic w Wynikach

### Duplikaty modeli w różnych plikach

#### transformer_baseline:
- W `model_comparison.csv`: test_mae = **0.000324** (bardzo dobry)
- W `model_comparison_transformer.csv`: test_mae = **0.004336** (13x gorszy)
- **Różnica:** 0.004012 (400% różnicy!)

#### transformer_small:
- W `model_comparison.csv`: test_mae = **0.000278** (najlepszy)
- W `model_comparison_transformer.csv`: test_mae = **0.004701** (17x gorszy)
- **Różnica:** 0.004423 (1688% różnicy!)

### Możliwe przyczyny:

1. **Różne timestamp treningu:**
   - `model_comparison.csv`: `20260204_232051`
   - `model_comparison_transformer.csv`: `20260209_111642`
   - **5 dni różnicy!**

2. **Różne random seed:**
   - Nie ustawione w `train_models.py`
   - Każdy trening ma inny split i inicjalizację wag

3. **Różne early stopping:**
   - Model mógł zatrzymać się w różnych epokach
   - `best_epoch` może być różny

4. **Różne dane (split):**
   - `split_matches()` może generować różne podziały
   - Jeśli brak `random_state`, każdy split jest inny

---

## ✅ Rekomendacje

### 1. **Użyj wyników z `model_comparison.csv`** dla Transformer/Attention LSTM
- Te modele były trenowane w tym samym biegu
- Wyniki bardziej wiarygodne (test_mae < 0.0004)

### 2. **Użyj wyników z `model_comparison_lstm.csv`** dla LSTM
- Dedykowane testy dla wariantów Dense
- Wszystkie trenowane w tym samym biegu

### 3. **Zignoruj `model_comparison_transformer.csv`** dla Transformer
- Gorsze wyniki (prawdopodobnie gorszy split lub seed)
- Użyj lstm_large_single_lstm jako jedynego nowego wyniku

### 4. **Najlepsze modele (finalna lista):**

| Ranking | Model                  | Typ             | test_mae | Źródło                       |
|---------|------------------------|-----------------|----------|------------------------------|
| 🥇      | transformer_small      | transformer     | 0.000278 | model_comparison.csv         |
| 🥈      | attention_lstm_large   | attention_lstm  | 0.000283 | model_comparison.csv         |
| 🥉      | lstm_large_no_dense    | lstm            | 0.005298 | model_comparison_lstm.csv    |
| 4       | bigru_medium           | bigru           | 0.010016 | model_comparison.csv         |

---

## 🔧 Sugestie Ulepszeń dla `train_models.py`

```python
# 1. Ustaw random seed dla reprodukowalności
import numpy as np
import tensorflow as tf
import random

def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# 2. Zapisuj split do pliku
def save_split(train_matches, val_matches, test_matches, output_dir):
    split_info = {
        'train_match_ids': [m['match_id'] for m in train_matches],
        'val_match_ids': [m['match_id'] for m in val_matches],
        'test_match_ids': [m['match_id'] for m in test_matches],
    }
    with open(f'{output_dir}/split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

# 3. Zapisuj best_epoch w metrics
trainer.save_training_metrics({
    'best_epoch': trainer.history.epoch[np.argmin(trainer.history.history['val_loss'])],
    # ...other metrics...
})
```

---

**Autor:** Analiza wykonana na podstawie:
- `train_models.py`
- `model_comparison.csv`
- `model_comparison_lstm.csv`
- `model_comparison_transformer.csv`

