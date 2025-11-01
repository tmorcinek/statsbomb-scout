# Quick Start Guide - StatsBomb Scout

## 🚀 Szybki Start (5 minut)

### 1. Aktywacja środowiska

```bash
# Aktywuj środowisko wirtualne
source .venv/bin/activate  # macOS/Linux
# lub
.venv\Scripts\activate  # Windows

# Sprawdź czy wszystkie pakiety są zainstalowane
pip install -r requirements.txt
```

### 2. Uruchom główny pipeline

```bash
python main.py
```

To uruchomi pełny pipeline treningu na danych StatsBomb.

---

## 📁 Struktura Projektu

```
statsbomb-scout/
├── config.py              # Wszystkie parametry w jednym miejscu
├── main.py                # Główny pipeline treningu
├── requirements.txt       # Zależności Python
│
├── src/                   # Moduły projektu
│   ├── data_loader.py     # Wczytywanie danych StatsBomb
│   ├── preprocessing.py   # Przetwarzanie na sekwencje
│   ├── model.py           # Architektury LSTM/Transformer
│   ├── train.py           # Trenowanie i ewaluacja
│   └── xthreat.py         # Expected Threat (xT) model
│
├── data/                  # Dane
│   ├── raw/               # Surowe dane
│   ├── processed/         # Przetworzone dane
│   └── statsbomb/         # Dane StatsBomb Open Data
│
├── models/                # Zapisane modele
│   ├── best_model.h5      # Najlepszy model (auto-zapisywany)
│   └── xt_models/         # Modele xT (cachowane)
│
├── tests/                 # Testy jednostkowe
└── docs/                  # Dokumentacja
```

---

## 🎯 Twój Pierwszy Model (Krok po kroku)

### Krok 1: Sprawdź dane StatsBomb

Projekt zawiera już dane StatsBomb Open Data w folderze `data/statsbomb/`:
- **Konkurencja 55** (UEFA Euro)
- **Sezon 282** (UEFA Euro 2020)
- Około 51 meczów z pełnymi danymi event-level

Możesz też pobrać dodatkowe dane:

```python
from statsbombpy import sb

# Zobacz dostępne konkurencje
competitions = sb.competitions()
print(competitions[['competition_name', 'season_name']])

# Pobierz mecze z innej ligi
matches = sb.matches(competition_id=11, season_id=90)  # La Liga 2020/21
```

### Krok 2: Dostosuj konfigurację

Otwórz `config.py` i dostosuj parametry:

```python
MODEL_TYPE = 'lstm'        # lub 'transformer'
SEQUENCE_LENGTH = 10       # długość sekwencji akcji
BATCH_SIZE = 32
EPOCHS = 50
```

### Krok 3: Uruchom główny pipeline

```bash
python main.py
```

Pipeline automatycznie:
- Załaduje dane StatsBomb
- Podzieli na train/val/test
- Przeprocesuje sekwencje
- Wytrenuje model
- Zapisze najlepszy model jako `models/best_model.h5`

---

## 🔧 Podstawowa Konfiguracja

### Modyfikacja hiperparametrów

W pliku `config.py`:

```python
# Zmień długość sekwencji
SEQUENCE_LENGTH = 15  # więcej kontekstu

# Zmień architekturę LSTM
LSTM_UNITS = 256      # większa sieć
LSTM_DROPOUT = 0.3    # więcej regularyzacji

# Lub użyj Transformera
MODEL_TYPE = 'transformer'
TRANSFORMER_HEADS = 8
TRANSFORMER_BLOCKS = 3
```

### Dostosowanie trenowania

```python
# W config.py
BATCH_SIZE = 64       # większe batche = szybsze trenowanie
EPOCHS = 100          # więcej epok
LEARNING_RATE = 0.0005  # mniejszy learning rate
```

---

## 📊 Podstawowa Analiza

### Po wytrenowaniu modelu:

```python
from src.train import ModelTrainer
import tensorflow as tf

# Wczytaj zapisany model
model = tf.keras.models.load_model('models/best_model.h5')

# Predykcje na nowych danych
predictions = model.predict(X_test)

# Analiza wyników
import pandas as pd
results = pd.DataFrame({
    'true_xg': y_test,
    'pred_xg': predictions.flatten()
})

# Statystyki
print(results.describe())

# Top 10 najlepszych sekwencji
print(results.nlargest(10, 'pred_xg'))
```

---

## 🐛 Rozwiązywanie Problemów

### Problem: "No module named 'statsbombpy'"
```bash
pip install statsbombpy
```

### Problem: "TensorFlow not found"
```bash
pip install tensorflow
```

### Problem: Model nie uczy się (loss nie spada)
- Sprawdź czy dane są znormalizowane
- Zmniejsz learning rate w `config.py`
- Sprawdź czy labels mają sensowne wartości

### Problem: Model overfittuje (val_loss rośnie)
- Zwiększ dropout rate
- Użyj więcej danych
- Zmniejsz rozmiar sieci

### Problem: Out of memory podczas trenowania
- Zmniejsz `BATCH_SIZE`
- Zmniejsz `SEQUENCE_LENGTH`
- Zmniejsz rozmiar sieci (LSTM_UNITS)

---

## 📚 Przydatne Komendy

```bash
# Sprawdź wersję pakietów
pip list | grep tensorflow
pip list | grep pandas

# Zaktualizuj pakiety
pip install --upgrade tensorflow

# Zobacz strukturę plików
ls -R src/

# Sprawdź zapisane modele
ls -lh models/

# Uwaga: best_model.h5 to najlepszy model z treningu
# (automatycznie zapisywany przez ModelCheckpoint callback)

# Jeśli chcesz usunąć wszystkie wytrenowane modele:
rm models/*.h5

# Eksportuj environment
pip freeze > requirements.txt

# Zobacz logi TensorBoard (jeśli skonfigurujesz)
tensorboard --logdir=logs/
```

---

## 💡 Szybkie Eksperymenty

### Eksperyment 1: Porównaj LSTM vs Transformer

```bash
# W config.py ustaw MODEL_TYPE = 'lstm'
python main.py

# Zapisz wyniki, potem zmień na:
# MODEL_TYPE = 'transformer'
python main.py

# Porównaj metryki w models/metrics.json
```

### Eksperyment 2: Wpływ długości sekwencji

```python
# Testuj różne wartości
for seq_len in [5, 10, 15, 20]:
    # W config.py: SEQUENCE_LENGTH = seq_len
    # Uruchom trenowanie i zapisz wyniki
```

### Eksperyment 3: Feature importance

```python
# Trenuj model z różnymi zestawami cech
# Sprawdź które cechy są najważniejsze
```

---

## 🎓 Następne Kroki

1. **Przeczytaj:** `NEXT_STEPS.md` - szczegółowy przewodnik implementacji
2. **Eksperymentuj:** Modyfikuj parametry w `config.py`
3. **Dostosuj:** Zmodyfikuj preprocessing pod swoje potrzeby
4. **Skaluj:** Użyj wszystkich dostępnych meczów/lig
5. **Analizuj:** Stwórz rankingi zawodników na podstawie predykcji modelu

---

## 📞 Pomoc

- **Dokumentacja StatsBomb:** https://github.com/statsbomb/statsbombpy
- **TensorFlow Tutorials:** https://www.tensorflow.org/tutorials
- **Issues:** Sprawdź `NEXT_STEPS.md` dla typowych problemów

---

**Powodzenia! ⚽🚀**

