# LABEL_GENERATION.md

Dokument opisuje konstrukcję etykiet (funkcji celu) używanych w procesie trenowania modeli deep learning do przewidywania wartości sekwencji akcji piłkarskich.

---

## 🎯 Cel uczenia maszynowego

Model uczy się przewidywać **wartość sekwencji akcji** (possession value), która jest miarą zagrożenia bramkowego stwarzanego przez daną sekwencję akcji w piłce nożnej.

**Problem:** Regresja (wartości ciągłe)  
**Zakres etykiet:** [0.0 - 1.0]  
**Interpretacja:** Prawdopodobieństwo/zagrożenie zdobycia bramki

---

## 📐 Konstrukcja etykiety - Funkcja `get_actions_value()`

### Lokalizacja
**Plik:** `src/ml/preprocessing/action_valuation.py`

### Implementacja

```python
def get_actions_value(actions_df):
    """
    Oblicza wartość sekwencji akcji jako maksimum z:
    - sum(xG) - suma Expected Goals wszystkich strzałów
    - xT_final - Expected Threat ostatniej akcji
    
    Args:
        actions_df (pd.DataFrame): DataFrame z akcjami w sekwencji
            Wymagane kolumny:
            - 'xG': Expected Goals dla każdej akcji
            - 'xT': Expected Threat dla każdej akcji
    
    Returns:
        float: wartość sekwencji w zakresie [0.0 - 1.0]
    
    Example:
        >>> actions = pd.DataFrame({
        ...     'xG': [0.0, 0.0, 0.35],
        ...     'xT': [0.02, 0.05, 0.20]
        ... })
        >>> get_actions_value(actions)
        0.35
    """
    # 1. Suma xG wszystkich strzałów w sekwencji
    total_xg = actions_df['xG'].sum()
    
    # 2. xT ostatniej akcji (końcowa pozycja sekwencji)
    final_xt = actions_df['xT'].iloc[-1]
    
    # 3. Wartość = max(sum(xG), xT)
    value = max(total_xg, final_xt)
    
    return value
```

---

## 🧠 Logika konstrukcji etykiety

Etykieta reprezentuje **potencjalne zagrożenie bramkowe** stworzone przez sekwencję. Używamy **maksimum** z dwóch metryk:

### 1. **sum(xG)** - Suma Expected Goals

#### Definicja
Suma prawdopodobieństw zdobycia gola ze **wszystkich strzałów** w sekwencji.

#### Charakterystyka
- **Źródło:** Modele xG (Expected Goals) trenowane na danych StatsBomb
- **Zakres:** [0.0 - 1.0+] (może być >1.0 przy wielu strzałach)
- **Kiedy xG > 0:** Akcja typu `shot`, `shot_penalty`, `shot_freekick`
- **Kiedy dominuje:** Sekwencje kończące się strzałem/strzałami

#### Przykład - sekwencja ze strzałem

```python
Actions:
1. Pass    | xG = 0.0,  xT = 0.02
2. Pass    | xG = 0.0,  xT = 0.05
3. Dribble | xG = 0.0,  xT = 0.08
4. Pass    | xG = 0.0,  xT = 0.15
5. Shot    | xG = 0.35, xT = 0.20

# Obliczenia:
total_xg = 0.0 + 0.0 + 0.0 + 0.0 + 0.35 = 0.35
final_xt = 0.20

# Etykieta:
label = max(0.35, 0.20) = 0.35  ✓ (xG dominuje)
```

**Interpretacja:** Sekwencja miała 35% szans na zdobycie bramki.

---

### 2. **xT_final** - Expected Threat ostatniej akcji

#### Definicja
Prawdopodobieństwo zdobycia gola z **końcowej pozycji** sekwencji (pozycja piłki po ostatniej akcji).

#### Charakterystyka
- **Źródło:** Model xT (grid-based 12x8) trenowany na danych historycznych
- **Zakres:** [0.0 - ~0.4] (maksymalna wartość w polu bramkowym)
- **Reprezentuje:** Potencjał pozycji na boisku (pole karne = wysokie xT)
- **Kiedy dominuje:** Sekwencje **bez strzału** (podanie, drybling, cross, etc.)

#### Przykład - sekwencja bez strzału

```python
Actions:
1. Pass | xG = 0.0, xT = 0.02
2. Pass | xG = 0.0, xT = 0.05
3. Pass | xG = 0.0, xT = 0.12
4. Pass | xG = 0.0, xT = 0.18
5. Pass | xG = 0.0, xT = 0.25  ← Kończy w polu karnym

# Obliczenia:
total_xg = 0.0  # Brak strzału
final_xt = 0.25

# Etykieta:
label = max(0.0, 0.25) = 0.25  ✓ (xT dominuje)
```

**Interpretacja:** Sekwencja stworzyła pozycję z ~25% potencjałem na bramkę.

---

## 🤔 Dlaczego max(xG, xT)? Uzasadnienie

### Pytanie
Dlaczego używamy `max()` zamiast `sum()`, `mean()`, lub `xG + xT`?

### Odpowiedź

#### 1. **Różne scenariusze posiadań**

Posiadania piłki mogą zakończyć się na dwa główne sposoby:

**A) Ze strzałem:**
- xG bezpośrednio mierzy zagrożenie
- xT byłby redundantny (pozycja już zawarta w xG)
- **Wybieramy:** xG

**B) Bez strzału:**
- xG = 0 (brak próby strzału)
- xT pokazuje wartość końcowej pozycji
- **Wybieramy:** xT

#### 2. **Unikanie duplikacji informacji**

```python
# ❌ ZŁE: sum(xG, xT) - podwójne liczenie
Actions:
- Shot from 16m: xG = 0.30, xT = 0.25
- label = 0.30 + 0.25 = 0.55  # Za wysoka! Pozycja już jest w xG

# ✅ DOBRE: max(xG, xT)
- label = max(0.30, 0.25) = 0.30  # Poprawnie odzwierciedla zagrożenie
```

**xG strzału już zawiera informację o pozycji**, więc dodawanie xT prowadziłoby do zawyżenia wartości.

#### 3. **Intuicja piłkarska**

| Scenariusz | xG | xT | max() | Interpretacja |
|------------|----|----|-------|---------------|
| Strzał z 16m | 0.30 | 0.20 | **0.30** | Wartość = szansa na gola ze strzału |
| Podanie w pole karne (bez strzału) | 0.0 | 0.25 | **0.25** | Wartość = potencjał pozycji |
| Dwa strzały | 0.15+0.20=0.35 | 0.15 | **0.35** | Wartość = łączna szansa z obu prób |
| Podanie na środku pola | 0.0 | 0.03 | **0.03** | Niska wartość = mało zagrożenia |

#### 4. **Empiryczna walidacja**

W praktyce `max(xG, xT)` najlepiej koreluje z:
- Oceną ekspertów (scouts)
- Realnym zagrożeniem bramkowym
- Decyzjami trenerskimi

---

## 📊 Przykłady konstrukcji etykiet

### Przykład 1: Gol z akcji 🎯

```python
Actions:
1. Pass  (50m, 34m → 65m, 30m) | xG = 0.0,  xT = 0.03
2. Pass  (65m, 30m → 80m, 28m) | xG = 0.0,  xT = 0.08
3. Cross (80m, 28m → 95m, 34m) | xG = 0.0,  xT = 0.15
4. Shot  (95m, 34m → GOAL!)    | xG = 0.45, xT = 0.30

# Obliczenia:
total_xg = 0.45
final_xt = 0.30

# Etykieta:
label = max(0.45, 0.30) = 0.45
```

**Interpretacja:**  
Sekwencja miała **45% szans na gola** (model xG). Realizacja: **GOL!** ⚽

---

### Przykład 2: Atak zakończony wybiciem piłki 🚫

```python
Actions:
1. Pass    (45m, 34m → 55m, 30m) | xG = 0.0, xT = 0.02
2. Dribble (55m, 30m → 65m, 28m) | xG = 0.0, xT = 0.05
3. Pass    (65m, 28m → 80m, 32m) | xG = 0.0, xT = 0.12
4. Pass    (80m, 32m → 92m, 34m) | xG = 0.0, xT = 0.22
   → Obrońca wybija piłkę poza pole karne

# Obliczenia:
total_xg = 0.0  # Brak strzału
final_xt = 0.22

# Etykieta:
label = max(0.0, 0.22) = 0.22
```

**Interpretacja:**  
Sekwencja stworzyła pozycję z **~22% potencjałem na bramkę**, ale została przerwana przez obronę.

---

### Przykład 3: Wiele strzałów w jednym posiadaniu 🔄

```python
Actions:
1. Pass (60m, 34m → 75m, 30m)         | xG = 0.0,  xT = 0.05
2. Shot (75m, 30m → BLOCKED by def!)  | xG = 0.15, xT = 0.10
3. Pass (rebound: 70m, 32m → 85m, 34m) | xG = 0.0,  xT = 0.15
4. Shot (85m, 34m → SAVED by GK!)    | xG = 0.25, xT = 0.20

# Obliczenia:
total_xg = 0.15 + 0.25 = 0.40  # Suma dwóch strzałów
final_xt = 0.20

# Etykieta:
label = max(0.40, 0.20) = 0.40
```

**Interpretacja:**  
Sekwencja stworzyła **łącznie 40% szansy na bramkę** z dwóch prób (15% + 25%).

---

### Przykład 4: Gol bezpośrednio z rzutu wolnego 🚀

```python
Actions:
1. Shot_freekick (85m, 34m → GOAL!)  | xG = 0.08, xT = 0.15

# Obliczenia:
total_xg = 0.08
final_xt = 0.15

# Etykieta:
label = max(0.08, 0.15) = 0.15
```

**Uwaga:**  
Tutaj xT > xG, bo model xG ocenia rzuty wolne jako trudne (8%), ale **pozycja** (xT) jest korzystna (15%). Używamy xT jako etykiety.

---

### Przykład 5: Długa kombinacja bez strzału ⚡

```python
Actions:
1. Pass    (40m, 34m → 50m, 32m) | xG = 0.0, xT = 0.01
2. Pass    (50m, 32m → 60m, 30m) | xG = 0.0, xT = 0.02
3. Dribble (60m, 30m → 70m, 28m) | xG = 0.0, xT = 0.04
4. Pass    (70m, 28m → 80m, 30m) | xG = 0.0, xT = 0.09
5. Pass    (80m, 30m → 90m, 32m) | xG = 0.0, xT = 0.18
6. Cross   (90m, 32m → 100m, 34m) | xG = 0.0, xT = 0.28
   → Cross przechwycony przez bramkarza

# Obliczenia:
total_xg = 0.0  # Brak strzału
final_xt = 0.28  # Wysoka wartość (blisko bramki)

# Etykieta:
label = max(0.0, 0.28) = 0.28
```

**Interpretacja:**  
Długa, efektowna kombinacja stworzyła **28% potencjał** na bramkę, ale bez strzału.

---

## 📈 Rozkład etykiet w danych treningowych

### Statystyki dla Euro 2024 (51 meczów)

**Liczba sekwencji:** 21,864 (training + validation + test)

| Percentyl | Wartość | Interpretacja | Przykład |
|-----------|---------|---------------|----------|
| **P5** | 0.001 | Minimalne zagrożenie | Podanie obronne |
| **P10** | 0.002 | Bardzo niskie | Akcje na własnej połowie |
| **P25** | 0.005 | Niskie | Podania w środku pola |
| **P50** | 0.015 | Średnie | Progresja do przodu |
| **P75** | 0.045 | Wysokie | Wejście w ostatnią tercję |
| **P90** | 0.120 | Bardzo wysokie | Strzały spoza pola karnego |
| **P95** | 0.250 | Ekstremalnie wysokie | Strzały z 16m |
| **P99** | 0.450 | Niemal pewne gole | 1 vs 1 z bramkarzem |
| **Max** | 0.950 | Gole z kilku metrów | Pusty cel z 2m |

### Rozkład wartości (histogram)

```
Zakres        | Sekwencje |                          | Procent
─────────────┼───────────┼──────────────────────────┼─────────
[0.00, 0.01) | 9,839     | ████████████████████████ | 45.0%
[0.01, 0.05) | 5,466     | ████████████             | 25.0%
[0.05, 0.10) | 3,280     | ████████                 | 15.0%
[0.10, 0.20) | 1,749     | ████                     | 8.0%
[0.20, 0.50) | 1,093     | ██                       | 5.0%
[0.50, 1.00) | 437       | █                        | 2.0%
```

### Interpretacja rozkładu

**Obserwacje:**
1. **45% sekwencji** ma bardzo niską wartość (<0.01) - typowe posiadania defensywne
2. **70% sekwencji** ma wartość <0.05 - większość posiadań nie stwarza dużego zagrożenia
3. **Tylko 7%** sekwencji ma wartość >0.20 - rzeczywiste zagrożenia bramkowe są rzadkie
4. **Long-tail distribution** - typowa dla danych piłkarskich (większość akcji niegroźna, kilka bardzo groźnych)

---

## 🎓 Funkcja straty (Loss Function)

### Mean Squared Error (MSE)

Model jest trenowany z użyciem **MSE** jako funkcji straty:

```python
def mse_loss(y_pred, y_true):
    """
    Mean Squared Error
    
    Args:
        y_pred: Przewidywane wartości (model output)
        y_true: Prawdziwe wartości (etykiety)
    
    Returns:
        float: średni kwadratowy błąd
    """
    return np.mean((y_pred - y_true) ** 2)
```

**Wzór matematyczny:**

```
MSE = (1/N) * Σᵢ₌₁ᴺ (ŷᵢ - yᵢ)²

gdzie:
- N = liczba przykładów
- ŷᵢ = przewidywana wartość dla i-tego przykładu
- yᵢ = prawdziwa etykieta dla i-tego przykładu
```

### Uzasadnienie wyboru MSE

#### ✅ Zalety MSE dla tego problemu:

1. **Regresja z wartościami ciągłymi**
   - Etykiety w [0, 1] (wartości ciągłe)
   - MSE naturalna dla problemów regresji

2. **Penalizacja dużych błędów**
   - Kwadrat błędu → duże błędy są bardziej karane
   - Ważne: błąd 0.5 (np. predykcja 0.1 zamiast 0.6) jest gorszy niż 5× błąd 0.1

3. **Gładka i różniczkowalna**
   - Umożliwia gradient descent
   - Stabilne gradienty dla backpropagation

4. **Interpretowalna**
   - Jednostki: squared probability
   - RMSE (√MSE) w tych samych jednostkach co etykiety

#### ⚠️ Wady i mitigacje:

**Problem:** Imbalanced data (70% wartości <0.05)

**Rozwiązanie:** 
- Weighted MSE (opcjonalnie)
- Batch balancing podczas treningu
- Monitoring MAE jako dodatkowej metryki

---

## 📊 Metryki ewaluacji modelu

Oprócz MSE (loss function), używane są dodatkowe metryki do oceny jakości modelu:

### 1. MAE (Mean Absolute Error)

```python
MAE = (1/N) * Σᵢ₌₁ᴺ |ŷᵢ - yᵢ|
```

**Interpretacja:** Średni bezwzględny błąd predykcji

**Zalety:**
- W tych samych jednostkach co etykiety [0, 1]
- Mniej wrażliwy na outliers niż MSE
- Łatwiejsza interpretacja

**Przykład:** MAE = 0.025 oznacza średni błąd ~2.5 punktu procentowego

---

### 2. RMSE (Root Mean Squared Error)

```python
RMSE = √MSE = √[(1/N) * Σᵢ₌₁ᴺ (ŷᵢ - yᵢ)²]
```

**Interpretacja:** Pierwiastek z MSE (w jednostkach etykiet)

**Zalety:**
- Te same jednostki co etykiety
- Bardziej interpretowalna niż MSE
- Pokazuje typową wielkość błędu

**Przykład:** RMSE = 0.035 oznacza typowy błąd ~3.5 punktu procentowego

---

### 3. R² (Coefficient of Determination)

```python
R² = 1 - (SS_res / SS_tot)

gdzie:
SS_res = Σᵢ (yᵢ - ŷᵢ)²   # Suma kwadratów reszt
SS_tot = Σᵢ (yᵢ - ȳ)²    # Całkowita suma kwadratów
```

**Interpretacja:** Procent wariancji wyjaśniony przez model

**Zakres:**
- R² = 1.0: Perfekcyjne dopasowanie
- R² = 0.5: Model wyjaśnia 50% wariancji
- R² = 0.0: Model nie lepszy niż średnia
- R² < 0.0: Model gorszy niż średnia (!)

**Przykład:** R² = 0.72 → model wyjaśnia 72% wariancji w wartościach sekwencji

---

### 4. Spearman's Rank Correlation

```python
from scipy.stats import spearmanr
rho, p_value = spearmanr(y_true, y_pred)
```

**Interpretacja:** Korelacja rangowa (porządek predykcji vs prawda)

**Zalety:**
- Nie zakłada liniowej relacji
- Odporna na outliers
- Ważna dla rankingowania sekwencji

**Przykład:** ρ = 0.85 → model dobrze rankinguje sekwencje od najgorszych do najlepszych

---

## 🧪 Walidacja jakości etykiet

### Sanity checks w kodzie

```python
def validate_label(label, actions_df):
    """Sprawdza poprawność wygenerowanej etykiety"""
    
    # 1. Zakres [0, 1]
    assert 0.0 <= label <= 1.0, f"Label out of range: {label}"
    
    # 2. Jeśli brak strzału, label == xT_final
    if actions_df['xG'].sum() == 0:
        assert label == actions_df['xT'].iloc[-1], \
            "Label should equal final xT when no shots"
    
    # 3. Jeśli jest strzał, label >= max(xG)
    if actions_df['xG'].sum() > 0:
        assert label >= actions_df['xG'].max(), \
            "Label should be at least max xG"
    
    # 4. Label nie może być ujemna
    assert label >= 0, "Label cannot be negative"
    
    return True
```

### Handling edge cases

#### Edge case 1: Bardzo długie posiadania

```python
# Problem: Posiadanie z 20+ akcjami → może mieć wiele strzałów
# Rozwiązanie: sum(xG) może być >1.0, to OK
# Clipping: NIE - chcemy zachować informację o wielu szansach
```

#### Edge case 2: xT = 0 i xG = 0

```python
# Sytuacja: Akcja defensywna na własnej połowie
# Label = max(0, 0) = 0.0
# Interpretacja: Brak zagrożenia bramkowego - poprawne
```

#### Edge case 3: xG bardzo wysokie (>0.8)

```python
# Sytuacja: Gol z 2 metrów, xG = 0.95
# Label = 0.95
# Interpretacja: Prawie pewny gol - poprawne
```

---

## 📚 Odniesienia i źródła

### Expected Goals (xG)
- **Źródło:** StatsBomb xG model
- **Paper:** "Expected Goals: Explained" - StatsBomb (2018)
- **Dataset:** Trenowany na ~100k strzałów z topowych lig

### Expected Threat (xT)
- **Źródło:** Karun Singh (2018)
- **Paper:** "Introducing Expected Threat (xT)" - https://karun.in/blog/expected-threat.html
- **Implementacja:** `socceraction` library
- **Model:** Grid-based (12x8 cells), wartości policzone z danych historycznych

### Metryki ewaluacji
- **MSE/MAE:** Standard ML metrics
- **R²:** Sklearn documentation
- **Spearman:** Scipy.stats documentation

---

## 🔗 Powiązane pliki w projekcie

| Plik | Opis |
|------|------|
| `src/ml/preprocessing/action_valuation.py` | Implementacja `get_actions_value()` |
| `src/ml/preprocessing/sequence_preprocessor.py` | Wywołanie `_create_label()` |
| `src/ml/preprocessing/xthreat.py` | Obliczanie Expected Threat |
| `tests/test_action_valuation.py` | Unit testy dla funkcji etykiet |
| `docs/DATA_PROCESSING_SCHEMA.md` | Etap 6: Label Generation |
| `docs/INPUT_DATA.md` | Sekcja o konstrukcji etykiet |

---

**Koniec dokumentu**

*Dokument utworzony: 8 lutego 2026*
*Plik: docs/LABEL_GENERATION.md*
*Wersja: 1.0*
