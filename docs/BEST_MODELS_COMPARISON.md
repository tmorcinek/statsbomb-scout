# Porównanie Najlepszych Modeli - Analiza Wyników

**Data:** 9 lutego 2026  
**Projekt:** StatsBomb Scout

---

## 📊 Podsumowanie Wykonawcze

Przetestowano **23 modele** w czterech kategoriach:
- **13 modeli LSTM** (różne konfiguracje)
- **3 modele Attention LSTM**
- **4 modele Transformer**
- **3 modele BiGRU**

---

## 🏆 Najlepsze Modele z Każdej Kategorii

### Tabela Porównawcza (sortowana według test_mae)

| Model                  | Typ             | test_mae | test_rmse | val_mae  | test_loss | val_loss | Gap (val-test mae) |
|------------------------|-----------------|----------|-----------|----------|-----------|----------|-------------------|
| **transformer_small** ⭐ | transformer     | **0.000278** | 0.016695  | 0.000486 | 0.000279  | 0.000486 | 0.000208          |
| **attention_lstm_large** | attention_lstm  | **0.000283** | 0.016853  | 0.000518 | 0.000284  | 0.000519 | 0.000235          |
| **lstm_large_no_dense** | lstm            | **0.005298** | 0.018332  | 0.006732 | 0.000336  | 0.000651 | 0.001433          |
| **bigru_medium**       | bigru           | **0.010016** | 0.027715  | 0.011768 | 0.000806  | 0.001175 | 0.001751          |

---

## 🎯 Wnioski

### 1. **Transformer Small** - ZWYCIĘZCA 🥇

**Konfiguracja:**
- `num_heads`: 2
- `d_model`: 64
- `ff_dim`: 256
- `num_blocks`: 2
- `dropout`: 0.1
- `epochs`: 75

**Wyniki:**
- ✅ **Najniższy test_mae**: 0.000278 (~0.028%)
- ✅ **Najniższy test_rmse**: 0.016695
- ✅ **Mała luka generalizacji**: 0.000208
- ✅ **Dobra generalizacja**: val_mae tylko nieznacznie wyższe od test_mae

**Interpretacja:**
- Średni błąd predykcji: **0.028 punktu procentowego** ⭐
- Model bardzo dobrze generalizuje na nowe dane
- Mniejsza architektura (small) okazała się lepsza niż większe warianty (baseline, large)

---

### 2. **Attention LSTM Large** - 2. miejsce 🥈

**Konfiguracja:**
- `lstm_units`: 128
- `dropout`: 0.3
- `epochs`: 100

**Wyniki:**
- ✅ **test_mae**: 0.000283 (~0.028%)
- ✅ **test_rmse**: 0.016853
- ✅ **Mała luka generalizacji**: 0.000235
- ✅ **Zwraca attention_weights** - interpretowalność!

**Interpretacja:**
- Prawie identyczny wynik jak Transformer Small
- Średni błąd: **0.028 punktu procentowego**
- Dodatkowa zaleta: **mechanizm attention pozwala interpretować decyzje**
- Świetna równowaga między wydajnością a interpretowalnością

---

### 3. **LSTM Large No Dense** - 3. miejsce 🥉

**Konfiguracja:**
- `lstm_units`: 128
- `dropout`: 0.3
- `dense_units`: None (brak dodatkowej warstwy Dense)
- `epochs`: 100

**Wyniki:**
- ⚠️ **test_mae**: 0.005298 (~0.53%)
- ✅ **test_rmse**: 0.018332
- ⚠️ **Większa luka generalizacji**: 0.001433

**Interpretacja:**
- Średni błąd: **0.53 punktu procentowego**
- **~19x gorszy** niż najlepsze modele (Transformer/Attention LSTM)
- Prostszy model LSTM bez attention nie radzi sobie tak dobrze
- Ciekawe: wariant **bez Dense layer** lepszy niż z Dense (64 lub 128 jednostek)

---

### 4. **BiGRU Medium** - 4. miejsce

**Konfiguracja:**
- `gru_units`: 96
- `attn_hidden`: 48
- `dropout`: 0.25
- `recurrent_dropout`: 0.12
- `l2_reg`: 0.012
- `epochs`: 125

**Wyniki:**
- ❌ **test_mae**: 0.010016 (~1.00%)
- ❌ **test_rmse**: 0.027715
- ❌ **Największa luka generalizacji**: 0.001751

**Interpretacja:**
- Średni błąd: **1.0 punktu procentowego**
- Najgorszy wynik spośród najlepszych z każdej kategorii
- Model ma problemy z generalizacją
- BiGRU nie radzi sobie dobrze z tym zadaniem

---

## 📈 Szczegółowa Analiza

### Ranking LSTM (Top 5)

| Pozycja | Model                 | test_mae | test_rmse | val_mae  | Opis                        |
|---------|----------------------|----------|-----------|----------|-----------------------------|
| 1       | lstm_large_no_dense  | 0.005298 | 0.018332  | 0.006732 | ✅ Bez Dense layer          |
| 2       | lstm_baseline_dense64| 0.005553 | 0.017851  | 0.006998 | 64 jednostki Dense          |
| 3       | lstm_large           | 0.005688 | 0.018028  | 0.007108 | Model z model_comparison.csv|
| 4       | lstm_large_dense64   | 0.005864 | 0.019036  | 0.007255 | 64 jednostki Dense          |
| 5       | lstm_small_dense128  | 0.006328 | 0.019095  | 0.007654 | Mały model, 128 Dense       |

**Wnioski dla LSTM:**
- ✅ **Większe modele (large) lepsze** od small/baseline
- ✅ **Brak Dense layer najlepszy** (no_dense wygrywa)
- ⚠️ Dodanie Dense layer (64 lub 128) **pogarsza** wyniki
- 📊 Wszystkie LSTM modele mają test_mae > 0.005 (0.5%)

---

### Ranking Transformer (wszystkie 4 modele)

| Pozycja | Model                      | test_mae | test_rmse | val_mae  | Opis                         |
|---------|---------------------------|----------|-----------|----------|------------------------------|
| 1       | transformer_small ⭐       | 0.000278 | 0.016695  | 0.000486 | ✅ 2 heads, 64 d_model       |
| 2       | transformer_true_attention | 0.000303 | 0.017377  | 0.000579 | True attention mechanism     |
| 3       | transformer_baseline       | 0.000324 | 0.017972  | 0.000501 | 4 heads, 128 d_model         |
| 4       | transformer_large          | 0.000799 | 0.028221  | 0.001205 | ❌ 8 heads, 256 d_model (overfitting) |

**Wnioski dla Transformer:**
- 🎯 **Mniejszy model (small) najlepszy!**
- ⚠️ **Duży model (large) znacznie gorszy** - overfitting (test_mae 3x gorsze)
- ✅ Wszystkie modele (poza large) mają **doskonałe wyniki** < 0.0004
- 📊 true_attention vs tanh attention - podobne wyniki

---

### Ranking Attention LSTM (wszystkie 3 modele)

| Pozycja | Model                      | test_mae | test_rmse | val_mae  | Opis                    |
|---------|---------------------------|----------|-----------|----------|-------------------------|
| 1       | attention_lstm_large ⭐    | 0.000283 | 0.016853  | 0.000518 | ✅ 128 jednostek        |
| 2       | attention_lstm_baseline    | 0.000322 | 0.017955  | 0.000532 | 64 jednostki            |
| 3       | attention_lstm_small       | 0.000346 | 0.018614  | 0.000572 | 32 jednostki            |

**Wnioski dla Attention LSTM:**
- ✅ **Większy model (large) najlepszy** - odwrotnie niż w Transformer
- ✅ **Wszystkie trzy modele mają doskonałe wyniki** < 0.0004
- 📊 Attention mechanizm znacząco poprawia wyniki vs zwykły LSTM
- 🎯 **Improvement: ~18x lepszy test_mae** (0.000283 vs 0.005298)

---

## 🔍 Analiza Mapowania Konfiguracji

### LSTM Models - `create_lstm_configs()`

**Warianty testowane:**
```
Rozmiar:  small (32) | baseline (64) | large (128)
Dense:    64         | 128           | None
Dropout:  0.15       | 0.2           | 0.3
```

**Najlepsze kombinacje:**
1. `large + no_dense` ✅
2. `baseline + dense64` ✅
3. `large` (z default dense) ✅

**Odrzucone kombinacje:**
- ❌ `dense128` - gorsze wyniki niż `dense64` lub `no_dense`
- ❌ `small` - za mały model

---

### Attention LSTM Models - `create_default_configs()`

**Warianty testowane:**
```
Rozmiar:  small (32) | baseline (64) | large (128)
Dropout:  0.15       | 0.2           | 0.3
Epochs:   50         | 75            | 100
```

**Wnioski:**
- ✅ **Large (128 jednostek) najlepszy**
- ✅ Wyższy dropout (0.3) pomaga
- ✅ Więcej epok (100) potrzebnych dla convergence

---

### Transformer Models - `create_transformers_config()` + `create_default_configs()`

**Warianty testowane:**
```
Model:    small      | baseline     | large        | true_attention
Heads:    2          | 4            | 8            | 4
d_model:  64         | 128          | 256          | 128
ff_dim:   256        | 512          | 1024         | 512
Blocks:   2          | 2            | 3            | 2
Epochs:   75         | 100          | 150          | 100
```

**Wnioski:**
- ✅ **Small (2 heads, 64 d_model) najlepszy**
- ❌ **Large (8 heads, 256 d_model) najgorszy** - overfitting
- 📊 Mniejsza architektura lepiej dopasowana do rozmiaru danych

---

### BiGRU Models - `create_default_configs()`

**Warianty testowane:**
```
Model:      baseline | medium      | large
GRU units:  64       | 96          | 128
Attn:       32       | 48          | 64
Dropout:    0.2      | 0.25        | 0.3
L2 reg:     0.01     | 0.012       | 0.015
```

**Wnioski:**
- ❌ **Wszystkie warianty słabe** (test_mae > 0.01)
- ⚠️ BiGRU nie nadaje się do tego zadania

---

## 💡 Rekomendacje

### 1. **Model Produkcyjny - Transformer Small** ⭐

**Zalecenia:**
```python
ModelConfig(
    name="transformer_small",
    model_type="transformer",
    model_params={
        'num_heads': 2,
        'd_model': 64,
        'ff_dim': 256,
        'num_blocks': 2,
        'dropout': 0.1,
    },
    training_params={'batch_size': 32, 'epochs': 75}
)
```

**Dlaczego:**
- ✅ **Najlepszy test_mae**: 0.000278
- ✅ Dobra generalizacja
- ✅ Szybsze trenowanie (75 epok vs 100)
- ✅ Mniejsza architektura = mniej overfitting

---

### 2. **Model Research/Interpretacja - Attention LSTM Large** 🔬

**Zalecenia:**
```python
ModelConfig(
    name="attention_lstm_large",
    model_type="attention_lstm",
    model_params={
        'lstm_units': 128,
        'dropout': 0.3
    },
    training_params={'batch_size': 32, 'epochs': 100}
)
```

**Dlaczego:**
- ✅ Prawie identyczny wynik: test_mae = 0.000283
- ✅ **Zwraca attention_weights** - interpretowalność
- ✅ Można analizować, które akcje są najważniejsze
- ✅ Lepsze dla celów badawczych

---

### 3. **Baseline Prosty - LSTM Large No Dense**

**Zalecenia:**
```python
ModelConfig(
    name="lstm_large_no_dense",
    model_type="lstm",
    model_params={
        'lstm_units': 128,
        'dropout': 0.3,
        'dense_units': None
    },
    training_params={'batch_size': 32, 'epochs': 100}
)
```

**Dlaczego:**
- ✅ Najprostszy model (brak attention, brak dense)
- ⚠️ Gorszy wynik: test_mae = 0.005298 (~19x gorszy)
- ✅ Dobry jako baseline do porównań
- ✅ Szybsze wnioskowanie

---

## 📊 Rozmiar Danych vs Architektura

**Dane wejściowe:**
- Sekwencja: **długość 6**
- Wektor: **46 cech** na każdy timestep
- Input shape: **(6, 46)**

**Obserwacje:**
1. **Małe dane → Mniejsze modele lepsze**
   - Transformer Small (2 heads, 64 d_model) > Large (8 heads, 256 d_model)
   
2. **Attention mechanizm kluczowy**
   - Attention LSTM: test_mae = 0.000283
   - Plain LSTM: test_mae = 0.005298
   - **Improvement: 18.7x** ⭐

3. **Dense layer niepotrzebny dla LSTM**
   - LSTM no_dense: 0.005298 ✅
   - LSTM dense64: 0.005553
   - LSTM dense128: 0.006681

---

## 🎓 Wnioski Końcowe

### Co się udało ✅

1. **Transformer Small i Attention LSTM Large** - doskonałe wyniki (test_mae < 0.0003)
2. **Mechanizm attention jest kluczowy** - 18x poprawa vs plain LSTM
3. **Mniejsze modele lepsze** - small/baseline > large (dla Transformer)
4. **Dobra generalizacja** - małe luki val_mae vs test_mae

### Co nie zadziałało ❌

1. **BiGRU** - słabe wyniki we wszystkich wariantach
2. **Duże modele** - overfitting (Transformer Large)
3. **Dense layers w LSTM** - pogarsza wyniki

### Następne kroki 🚀

1. ✅ **Deploy Transformer Small jako model produkcyjny**
2. ✅ **Użyj Attention LSTM Large do analizy i interpretacji**
3. 🔬 **Eksperymentuj z:**
   - Różne wartości dropout (0.05, 0.15, 0.2)
   - Różne learning rates
   - Ensemble Transformer Small + Attention LSTM Large
4. 📊 **Monitoruj overfitting** - EarlyStopping i regularization

---

**Plik wygenerowany:** `analyze_best_models.py`  
**Dane źródłowe:**
- `models/model_comparison.csv`
- `models/model_comparison_lstm.csv`
- `models/model_comparison_transformer.csv`

**Tabela zapisana:** `models/best_models_comparison.csv`

