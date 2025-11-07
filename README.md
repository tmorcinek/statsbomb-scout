# StatsBomb Scout - Football Player Evaluation Model

Model oceny zawodników piłkarskich na podstawie sekwencji akcji z danych StatsBomb.

## 📋 Opis Projektu

Projekt implementuje pipeline machine learning do oceny zawodników piłkarskich poprzez analizę sekwencji akcji (podań, prowadzeń, odbiorów, strzałów). Model przewiduje wartość sekwencji akcji na podstawie expected goals (xG).

### Główne Funkcjonalności:
- Wczytywanie danych zdarzeń meczowych ze StatsBomb (format SPADL)
- Ekstrakcja faz posiadania piłki
- Generowanie sekwencji akcji o ustalonej długości
- Ekstrakcja cech dla każdej akcji (współrzędne, typ, długość/kąt podania, czas, presja)
- Wycena akcji za pomocą xG i xThreat
- Podział meczów na zbiory treningowy/walidacyjny/testowy
- Trenowanie modeli sekwencyjnych (LSTM lub Transformer)
- Ewaluacja i analiza wyników

## 🏗️ Struktura Projektu

```
statsbomb-scout/
├── data/
│   ├── raw/              # Surowe dane JSON/CSV
│   └── processed/        # Przetworzone dane
├── docs/                 # Dokumentacja
│   ├── QUICKSTART.md
│   └── NEXT_STEPS.md
├── models/               # Zapisane modele i wykresy (patrz models/README.md)
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Wczytywanie danych StatsBomb
│   ├── preprocessing.py  # Przetwarzanie na sekwencje
│   ├── data_splitter.py  # Podział meczów na train/val/test
│   ├── xthreat.py        # Modele xThreat
│   ├── action_valuation.py  # Wycena akcji
│   ├── model.py          # Architektura LSTM/Transformer
│   └── train.py          # Trenowanie i ewaluacja
├── tests/                # Testy jednostkowe
│   ├── test_preprocessing.py
│   ├── test_xthreat.py
│   └── test_action_valuation.py
├── statsbombpy/          # Lokalna kopia biblioteki statsbombpy
├── config.py             # Konfiguracja hiperparametrów
├── main.py               # Główny skrypt pipeline
├── requirements.txt      # Zależności
└── README.md
```

## 🚀 Instalacja

### 1. Klonowanie repozytorium
```bash
cd statsbomb-scout
```

### 2. Utworzenie środowiska wirtualnego
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# lub
.venv\Scripts\activate  # Windows
```

### 3. Instalacja zależności
```bash
pip install -r requirements.txt
```

## 📊 Użycie

### 1. Przygotowanie danych
Dane StatsBomb są już dostępne w folderze `data/statsbomb/data/`:
- Zawiera pliki JSON z danymi: `competitions.json`, `matches/`, `events/`, `lineups/`
- Pipeline automatycznie wczytuje dane przy użyciu `load_statsbomb_socceraction_data()`

### 2. Konfiguracja
Edytuj `config.py` aby dostosować parametry:
```python
# Model type: 'lstm' or 'transformer'
MODEL_TYPE = 'lstm'

# Data splits
SEQUENCE_LENGTH = 10      # Długość sekwencji akcji
VALIDATION_SPLIT = 0.2    # 20% danych na walidację
TEST_SPLIT = 0.1          # 10% danych na test

# Training parameters
BATCH_SIZE = 32           # Rozmiar batcha
EPOCHS = 50               # Liczba epok trenowania
LEARNING_RATE = 0.001     # Learning rate

# LSTM parameters
LSTM_UNITS = 128          # Liczba jednostek LSTM
LSTM_DROPOUT = 0.2        # Dropout rate

# Transformer parameters
TRANSFORMER_HEADS = 4     # Liczba głów attention
TRANSFORMER_DIM = 128     # Wymiar modelu
TRANSFORMER_FF_DIM = 512  # Wymiar feedforward
TRANSFORMER_BLOCKS = 2    # Liczba bloków transformera
```

### 3. Uruchomienie pipeline
```bash
python main.py
```

### 4. Zapisywane modele
Podczas treningu model automatycznie zapisuje wyniki do folderu `models/`:
- **models/lstm/** lub **models/transformer/** - folder w zależności od typu modelu
  - **best_model.h5** - najlepszy model (najniższa `val_loss`) zapisywany przez `ModelCheckpoint`
  - **metrics.json** - metryki ewaluacji (test_loss, test_mae, rmse)
  - **training_history.png** - wykres historii treningu

**Uwaga**: Pliki `.h5` są ignorowane przez Git (patrz `.gitignore`), więc nie będą commitowane do repozytorium.

## 🔧 Moduły

### data_loader.py
- `load_statsbomb_socceraction_data()`: Funkcja do wczytywania danych StatsBomb i konwersji do formatu SPADL
- Wczytuje dane z lokalnego folderu `data/statsbomb/data/`

### data_splitter.py
- `split_matches()`: Funkcja do podziału meczów na zbiory treningowy, walidacyjny i testowy
- Zapewnia, że każdy mecz znajduje się tylko w jednym zbiorze

### preprocessing.py
- `SequencePreprocessor`: Przetwarzanie zdarzeń na sekwencje
- Ekstrakcja posiadań piłki
- Generowanie cech: współrzędne, typ akcji, metryki podań, czas, presja
- Tworzenie etykiet (xG lub xThreat)

### xthreat.py
- `get_default_xt_model()`: Funkcja zwracająca domyślny model xThreat
- `xTModel`: Klasa modelu xThreat do wyceny zagrożenia

### action_valuation.py
- Funkcje do wyceny akcji piłkarskich
- Integracja z modelami xThreat

### model.py
- `LSTMSequenceModel`: Model LSTM z warstwami dropout
- `TransformerSequenceModel`: Model Transformer z attention
- `create_model()`: Factory function do tworzenia modeli

### train.py
- `ModelTrainer`: Klasa do trenowania i ewaluacji
- Callbacks: Early Stopping, ReduceLROnPlateau, ModelCheckpoint
- Metryki: MSE, MAE, RMSE
- Wizualizacja wyników trenowania

## 📈 Features (Cechy Wejściowe)

Dla każdej akcji w sekwencji:
1. **Współrzędne**: `x_start`, `y_start`, `x_end`, `y_end`
2. **Typ akcji**: podanie, prowadzenie, odbiór, strzał (encoded)
3. **Metryki podania**: długość, kąt
4. **Czas**: delta od poprzedniej akcji
5. **Presja**: czy akcja wykonana pod presją

## 🎯 Output (Etykieta)

Wartość sekwencji:
- **xG strzału** jeśli sekwencja kończy się strzałem
- **xThreat** wartość zagrożenia dla innych akcji

## 🧠 Architektury Modeli

### LSTM
```
Input → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(64) → Dense(1)
```

### Transformer
```
Input → Embedding → Positional Encoding → TransformerEncoder × N → 
GlobalPooling → Dense(64) → Dense(1)
```



## 📚 Wymagania

- Python 3.9+
- TensorFlow 2.10+
- Pandas, NumPy, scikit-learn
- matplotlib (do wizualizacji)
- socceraction (do przetwarzania danych SPADL)
- statsbombpy (lokalna kopia w projekcie)

## 🔍 Analiza Wyników

Po wytrenowaniu modelu możesz:
- Analizować średnią przewidywaną wartość sekwencji dla każdego zawodnika
- Identyfikować wysokowartościowe sekwencje akcji
- Porównywać zawodników na podstawie jakości ich akcji

## 📄 Licencja

Projekt edukacyjny. Dane StatsBomb podlegają ich własnej licencji.

## 🤝 Kontakt

Projekt stworzony jako szablon do analizy danych StatsBomb.

