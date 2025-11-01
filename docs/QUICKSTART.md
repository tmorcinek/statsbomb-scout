# Quick Start Guide - StatsBomb Scout

## 🚀 Szybki Start (5 minut)

### 1. Instalacja środowiska

```bash
# Aktywuj środowisko wirtualne
source .venv/bin/activate  # macOS/Linux
# lub
.venv\Scripts\activate  # Windows

# Zainstaluj dodatkowe pakiety (jeśli jeszcze nie zainstalowane)
pip install jupyter seaborn
```

### 2. Uruchom przykład z dummy danymi

```bash
python example.py
```

To pokaże Ci jak działa cały pipeline z syntetycznymi danymi.

### 3. Otwórz Jupyter Notebook

```bash
jupyter notebook analysis.ipynb
```

Notebook pozwala na interaktywną eksplorację i eksperymenty.

---

## 📁 Struktura Projektu

```
statsbomb-scout/
├── config.py              # Wszystkie parametry w jednym miejscu
├── main.py                # Główny pipeline
├── example.py             # Demonstracja z dummy danymi
├── analysis.ipynb         # Jupyter notebook do eksperymentów
│
├── src/                   # Moduły projektu
│   ├── data_loader.py     # Wczytywanie danych StatsBomb
│   ├── preprocessing.py   # Przetwarzanie na sekwencje
│   ├── model.py           # Architektury LSTM/Transformer
│   └── train.py           # Trenowanie i ewaluacja
│
├── data/
│   ├── raw/              # Tutaj umieść pliki JSON/CSV
│   └── processed/        # Przetworzone dane
│
└── models/               # Zapisane modele
```

---

## 🎯 Twój Pierwszy Model (Krok po kroku)

### Krok 1: Pobierz dane StatsBomb

```python
from statsbombpy import sb

# Zobacz dostępne konkurencje
competitions = sb.competitions()
print(competitions)

# Pobierz mecze (przykład: La Liga 2020/21)
matches = sb.matches(competition_id=11, season_id=90)

# Pobierz wydarzenia z pierwszego meczu
events = sb.events(match_id=matches.iloc[0]['match_id'])

# Zapisz do pliku
events.to_csv('data/raw/events.csv', index=False)
```

### Krok 2: Dostosuj konfigurację

Otwórz `config.py` i dostosuj parametry:

```python
MODEL_TYPE = 'lstm'        # lub 'transformer'
SEQUENCE_LENGTH = 10       # długość sekwencji akcji
BATCH_SIZE = 32
EPOCHS = 50
```

### Krok 3: Uzupełnij preprocessing

Otwórz `src/preprocessing.py` i uzupełnij funkcje oznaczone `# TODO:`

**Najważniejsze:**
- `extract_possessions()` - wyciągnij fazy posiadania
- `create_features()` - stwórz cechy dla każdej akcji
- `create_labels()` - przypisz wartość xG do sekwencji

### Krok 4: Uruchom główny pipeline

```python
python main.py
```

Odkomentuj odpowiednie sekcje w `main.py` po przygotowaniu danych.

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
2. **Eksperymentuj:** Użyj `analysis.ipynb` do testowania
3. **Dostosuj:** Zmodyfikuj preprocessing pod swoje potrzeby
4. **Skaluj:** Użyj wszystkich dostępnych meczów
5. **Analizuj:** Stwórz rankingi zawodników

---

## 📞 Pomoc

- **Dokumentacja StatsBomb:** https://github.com/statsbomb/statsbombpy
- **TensorFlow Tutorials:** https://www.tensorflow.org/tutorials
- **Issues:** Sprawdź `NEXT_STEPS.md` dla typowych problemów

---

**Powodzenia! ⚽🚀**

