# Metryki Ewaluacji Modeli

## 📋 Wprowadzenie

W projekcie **StatsBomb Scout** wykorzystujemy zestaw metryk do oceny jakości modeli sekwencyjnych przewidujących wartość sekwencji akcji piłkarskich. Wszystkie metryki są obliczane na zbiorze testowym i walidacyjnym, co pozwala na obiektywne porównanie różnych architektur modeli (LSTM, Transformer, BiGRU, Attention LSTM).

---

## 🎯 Główne Metryki

### 1. Loss (MSE - Mean Squared Error)

**Definicja:**
```
MSE = (1/N) * Σᵢ₌₁ᴺ (ŷᵢ - yᵢ)²
```

gdzie:
- `N` - liczba próbek
- `ŷᵢ` - wartość przewidziana przez model
- `yᵢ` - wartość rzeczywista (ground truth)

**Charakterystyka:**
- Główna funkcja straty (loss function) używana podczas treningu
- Silnie karze duże błędy (kwadrat różnicy)
- Wartości w zakresie [0, ∞)
- Im niższa, tym lepiej

**Interpretacja:**
- MSE = 0.0002 oznacza średni kwadrat błędu na poziomie 0.02%
- Wartość trudniejsza do bezpośredniej interpretacji (jednostki do kwadratu)

**Zastosowanie w projekcie:**
- Funkcja straty dla wszystkich modeli
- Metryka zapisywana w `metrics.json`
- Wyświetlana jako "Loss" na wykresach historii treningu

---

### 2. MAE (Mean Absolute Error)

**Definicja:**
```
MAE = (1/N) * Σᵢ₌₁ᴺ |ŷᵢ - yᵢ|
```

**Charakterystyka:**
- **Główna metryka porównawcza** między modelami
- W tych samych jednostkach co wartości docelowe [0, 1]
- Mniej wrażliwa na wartości odstające (outliers) niż MSE
- Łatwiejsza w interpretacji niż MSE

**Interpretacja:**
- MAE = 0.005 oznacza średni błąd bezwzględny ~0.5 punktu procentowego
- Wartość bezpośrednio pokazuje typowy błąd predykcji w skali [0, 1]

**Dlaczego MAE jest główną metryką?**
- Intuicyjna interpretacja (średni błąd bezwzględny)
- W jednostkach wartości docelowych
- Sprawiedliwa dla wszystkich zakresów wartości
- Odporna na pojedyncze duże błędy

**Zastosowanie w projekcie:**
- Metryka treningowa (monitorowana podczas treningu)
- **Kryterium rankingu modeli** w tabelach porównawczych
- Obliczana dla zbiorów: treningowego, walidacyjnego i testowego
- Zapisywana jako `test_mae` i `val_mae` w wynikach

---

### 3. RMSE (Root Mean Squared Error)

**Definicja:**
```
RMSE = √MSE = √[(1/N) * Σᵢ₌₁ᴺ (ŷᵢ - yᵢ)²]
```

**Charakterystyka:**
- Pierwiastek z MSE (loss)
- W tych samych jednostkach co wartości docelowe [0, 1]
- Bardziej interpretowalna niż MSE
- Bardziej wrażliwa na duże błędy niż MAE

**Interpretacja:**
- RMSE = 0.018 oznacza typową wielkość błędu ~1.8 punktu procentowego
- Pokazuje średnią odległość predykcji od rzeczywistości (z większą wagą dla dużych błędów)

**Różnica MAE vs RMSE:**
- RMSE ≥ MAE (zawsze)
- Im większa różnica RMSE - MAE, tym więcej wartości odstających
- RMSE mocniej karze duże błędy

**Zastosowanie w projekcie:**
- Metryka dodatkowa do oceny modeli
- Zapisywana jako `test_rmse` w wynikach
- Pomaga zidentyfikować modele z problemem wartości odstających

---

## 📊 Metryki w Praktyce

### Zapisywanie Metryk

Wszystkie metryki są zapisywane w pliku `metrics.json` w katalogu każdego modelu:

```json
{
  "test_loss": 0.000234,
  "test_mae": 0.0053,
  "test_rmse": 0.0183,
  "val_mae": 0.0067,
  "training_time": "45m 32s",
  "epochs": 38,
  "best_epoch": 28
}
```

### Monitorowanie Podczas Treningu

Podczas treningu monitorowane są:
- **Training Loss** - MSE na zbiorze treningowym
- **Validation Loss** - MSE na zbiorze walidacyjnym
- **Training MAE** - MAE na zbiorze treningowym
- **Validation MAE** - MAE na zbiorze walidacyjnym

### Callbacks

**EarlyStopping:**
- Monitoruje: `val_loss` (validation MSE)
- Zatrzymuje trening gdy brak poprawy przez 10 epok

**ModelCheckpoint:**
- Monitoruje: `val_loss`
- Zapisuje model z najniższą wartością `val_loss`

**ReduceLROnPlateau:**
- Monitoruje: `val_loss`
- Zmniejsza learning rate gdy brak poprawy

---

## 📈 Porównywanie Modeli

### Tabela Porównawcza

Modele są porównywane przede wszystkim według **test_mae** (główne kryterium):

```
                  name model_type  test_mae  test_rmse  val_mae
   lstm_large_no_dense       lstm  0.005298   0.018332 0.006732
 lstm_baseline_dense64       lstm  0.005553   0.017851 0.006998
    lstm_large_dense64       lstm  0.005864   0.019036 0.007255
```

### Interpretacja Wyników

**Dobry model:**
- test_mae < 0.006 (błąd < 0.6 punktu procentowego)
- test_rmse < 0.020 (typowy błąd < 2 punkty procentowe)
- Mała różnica między val_mae a test_mae (dobra generalizacja)

**Overfitting:**
- val_mae znacznie wyższe niż training mae
- test_mae wyższe niż val_mae

**Underfitting:**
- Wysokie wartości wszystkich metryk
- Brak poprawy w kolejnych epokach

---

## 🔍 Dodatkowe Metryki (Opcjonalne)

### R² (Coefficient of Determination)

**Definicja:**
```
R² = 1 - (SS_res / SS_tot)

gdzie:
SS_res = Σᵢ (yᵢ - ŷᵢ)²   # Suma kwadratów reszt
SS_tot = Σᵢ (yᵢ - ȳ)²    # Całkowita suma kwadratów
```

**Interpretacja:**
- R² = 1.0: Perfekcyjne dopasowanie
- R² = 0.7: Model wyjaśnia 70% wariancji
- R² = 0.0: Model nie lepszy niż średnia
- R² < 0.0: Model gorszy niż średnia (!)

**Zastosowanie:**
- Pokazuje procent wyjaśnionej wariancji
- Przydatne do porównania z modelami baseline

---

### Spearman's Rank Correlation

**Definicja:**
```python
from scipy.stats import spearmanr
rho, p_value = spearmanr(y_true, y_pred)
```

**Interpretacja:**
- ρ (rho) ∈ [-1, 1]
- ρ = 1.0: Perfekcyjna korelacja dodatnia
- ρ = 0.0: Brak korelacji
- ρ = -1.0: Perfekcyjna korelacja ujemna

**Zastosowanie:**
- Ocena jakości rankingowania sekwencji
- Odporna na wartości odstające
- Nie zakłada liniowej relacji

---

## 🎓 Wnioski

### Hierarchia Ważności Metryk

1. **test_mae** - główna metryka porównawcza
2. **val_mae** - sprawdzenie generalizacji
3. **test_rmse** - dodatkowa perspektywa (wrażliwość na outliers)
4. **test_loss** - funkcja optymalizacyjna

### Przykłady z Projektu

**Najlepszy Model (lstm_large_no_dense):**
- test_mae: **0.005298** ✅
- test_rmse: **0.018332** ✅
- val_mae: 0.006732
- Interpretacja: średni błąd ~0.53%, bardzo dobry wynik

**Model Średni (lstm_small_dense64):**
- test_mae: **0.007813** ⚠️
- test_rmse: **0.020785** ⚠️
- val_mae: 0.008942
- Interpretacja: średni błąd ~0.78%, akceptowalny ale gorszy

---

## 📝 Implementacja

### Obliczanie Metryk w Kodzie

```python
def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate model on test set."""
    # Get model metrics (loss, mae)
    results = self.model.evaluate(X_test, y_test_prepared, verbose=0)
    
    metrics = {
        'loss': results[0],
        'mae': results[1] if len(results) > 1 else np.nan
    }
    
    # Calculate RMSE manually
    y_pred = self.model.predict(X_test)
    metrics['rmse'] = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    return metrics
```

### Konfiguracja Kompilacji

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',           # MSE jako loss
    metrics=['mae']       # MAE jako metryka treningowa
)
```

---

## 📚 Referencje

- **MSE**: [Wikipedia - Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error)
- **MAE**: [Wikipedia - Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error)
- **RMSE**: [Wikipedia - Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
- **R²**: [Wikipedia - Coefficient of Determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)

---

**Autor:** Tomasz Morcinek  
**Data:** 9 lutego 2026  
**Projekt:** StatsBomb Scout

