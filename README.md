# StatsBomb Scout - Football Player Evaluation Model

Model oceny zawodników piłkarskich na podstawie sekwencji akcji z danych StatsBomb.

## 📋 Opis Projektu

Projekt implementuje pipeline machine learning do oceny zawodników piłkarskich poprzez analizę sekwencji akcji (podań, prowadzeń, odbiorów, strzałów). Model przewiduje wartość sekwencji akcji na podstawie expected goals (xG).

### Główne Funkcjonalności:
- Wczytywanie danych zdarzeń meczowych ze StatsBomb (JSON/CSV)
- Ekstrakcja faz posiadania piłki
- Generowanie sekwencji akcji o ustalonej długości
- Ekstrakcja cech dla każdej akcji (współrzędne, typ, długość/kąt podania, czas, presja)
- Trenowanie modeli sekwencyjnych (LSTM lub Transformer)
- Ewaluacja i analiza wyników

## 🏗️ Struktura Projektu

```
statsbomb-scout/
├── data/
│   ├── raw/              # Surowe dane JSON/CSV
│   └── processed/        # Przetworzone dane
├── models/               # Zapisane modele i wykresy
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Wczytywanie danych StatsBomb
│   ├── preprocessing.py  # Przetwarzanie na sekwencje
│   ├── model.py          # Architektura LSTM/Transformer
│   └── train.py          # Trenowanie i ewaluacja
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
Umieść pliki z danymi StatsBomb w katalogu `data/raw/`:
- Format JSON (zalecany): `events.json`
- Format CSV: `events.csv`

### 2. Konfiguracja
Edytuj `config.py` aby dostosować parametry:
```python
SEQUENCE_LENGTH = 10      # Długość sekwencji akcji
BATCH_SIZE = 32           # Rozmiar batcha
EPOCHS = 50               # Liczba epok trenowania
MODEL_TYPE = 'lstm'       # 'lstm' lub 'transformer'
```

### 3. Uruchomienie pipeline
```bash
python main.py
```

## 🔧 Moduły

### data_loader.py
- `StatsBombDataLoader`: Klasa do wczytywania danych z JSON/CSV
- Metody: `load_from_json()`, `load_from_csv()`, `filter_relevant_events()`

### preprocessing.py
- `SequencePreprocessor`: Przetwarzanie zdarzeń na sekwencje
- Ekstrakcja posiadań piłki
- Generowanie cech: współrzędne, typ akcji, metryki podań, czas, presja
- Tworzenie etykiet (xG)
- Podział na zbiory: train/val/test

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
- **0** jeśli brak strzału na końcu sekwencji

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

## 📝 TODO - Miejsca do Uzupełnienia

W kodzie znajdują się komentarze `# TODO:` w następujących miejscach:

1. **data_loader.py**: 
   - Parsowanie struktury JSON StatsBomb
   - Implementacja logiki dla wielu meczów

2. **preprocessing.py**:
   - Ekstrakcja faz posiadania piłki
   - Implementacja funkcji `create_features()`
   - Logika tworzenia etykiet (xG)

3. **main.py**:
   - Podanie ścieżki do plików danych
   - Odkomentowanie pipeline po przygotowaniu danych

## 📚 Wymagania

- Python 3.9+
- TensorFlow 2.10+
- Pandas, NumPy, scikit-learn
- matplotlib (do wizualizacji)
- statsbombpy (opcjonalnie, do API)

## 🔍 Analiza Wyników

Po wytrenowaniu modelu możesz:
- Analizować średnią przewidywaną wartość sekwencji dla każdego zawodnika
- Identyfikować wysokowartościowe sekwencje akcji
- Porównywać zawodników na podstawie jakości ich akcji

## 📄 Licencja

Projekt edukacyjny. Dane StatsBomb podlegają ich własnej licencji.

## 🤝 Kontakt

Projekt stworzony jako szablon do analizy danych StatsBomb.

