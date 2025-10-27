# Następne Kroki - StatsBomb Scout

## 🎯 Miejsca do uzupełnienia w kodzie

### 1. data_loader.py

#### `load_from_json()`
- [ ] Parsowanie struktury JSON StatsBomb
- [ ] Wyciąganie pól: `timestamp`, `location`, `type`, `player`, `team`
- [ ] Obsługa zagnieżdżonych struktur (np. `pass.end_location`)

Przykład struktury StatsBomb JSON:
```json
{
  "id": "...",
  "type": {"name": "Pass"},
  "timestamp": "00:00:12.123",
  "location": [60.0, 40.0],
  "player": {"name": "Player Name"},
  "pass": {
    "end_location": [80.0, 45.0],
    "length": 25.3,
    "angle": 0.5
  }
}
```

#### `filter_relevant_events()`
- [ ] Dokładna filtracja typów zdarzeń
- [ ] Możliwe typy do filtrowania: `Pass`, `Carry`, `Ball Receipt*`, `Shot`, `Duel`, `Pressure`

---

### 2. preprocessing.py

#### `extract_possessions()`
- [ ] Grupowanie zdarzeń po `possession_team`
- [ ] Wykrywanie zmiany posiadania
- [ ] Filtrowanie zbyt krótkich posiadań (< sequence_length)

Wskazówka:
```python
possessions = []
for (match_id, poss_id), group in events_df.groupby(['match_id', 'possession']):
    if len(group) >= self.sequence_length:
        possessions.append(group)
```

#### `create_features()`
Zaimplementuj ekstrakcję cech dla każdej akcji:

1. **Współrzędne** (4 cechy):
   ```python
   x_start = row['location'][0] if row['location'] else 0
   y_start = row['location'][1] if row['location'] else 0
   x_end = row.get('pass_end_location', [x_start, y_start])[0]
   y_end = row.get('pass_end_location', [x_start, y_start])[1]
   ```

2. **Typ akcji** (1 cecha - encoded):
   ```python
   action_types = {'Pass': 0, 'Carry': 1, 'Ball Receipt*': 2, 'Shot': 3}
   action_type = action_types.get(row['type'], -1)
   ```

3. **Metryki podania** (2 cechy):
   ```python
   if row['type'] == 'Pass':
       pass_length = row.get('pass_length', 0)
       pass_angle = row.get('pass_angle', 0)
   else:
       pass_length = 0
       pass_angle = 0
   ```

4. **Delta czasowa** (1 cecha):
   ```python
   if idx > 0:
       prev_time = pd.to_datetime(possession_df.iloc[idx-1]['timestamp'])
       curr_time = pd.to_datetime(row['timestamp'])
       time_delta = (curr_time - prev_time).total_seconds()
   else:
       time_delta = 0
   ```

5. **Pod presją** (1 cecha):
   ```python
   under_pressure = 1 if row.get('under_pressure', False) else 0
   ```

#### `create_labels()`
- [ ] Sprawdzanie czy sekwencja kończy się strzałem
- [ ] Wyciąganie wartości `shot.statsbomb_xg`
- [ ] Alternatywnie: wartość 1 jeśli shot, 0 w przeciwnym razie

```python
end_event = possession_df.iloc[end_idx - 1]
if end_event['type'] == 'Shot':
    label = end_event.get('shot_statsbomb_xg', 0.1)
else:
    label = 0.0
```

---

### 3. model.py

#### Opcjonalne usprawnienia:

**LSTM:**
- [ ] Eksperymentuj z liczbą warstw LSTM
- [ ] Dodaj Bidirectional LSTM
- [ ] Wypróbuj GRU zamiast LSTM

**Transformer:**
- [ ] Implementuj lepsze positional encoding (sinusoidalne)
- [ ] Dodaj attention visualization
- [ ] Eksperymentuj z liczbą heads i bloków

---

### 4. train.py

#### Dodatkowe callbacki:
- [ ] TensorBoard logging
- [ ] Custom metrics (R², correlation)
- [ ] Learning rate scheduler

```python
keras.callbacks.TensorBoard(log_dir='logs/'),
keras.callbacks.LearningRateScheduler(schedule_fn)
```

---

### 5. main.py

#### Pipeline do uzupełnienia:
1. [ ] Podaj ścieżkę do rzeczywistych danych
2. [ ] Odkomentuj pipeline
3. [ ] Dodaj logowanie
4. [ ] Implementuj save/load checkpoints

---

## 📊 Analiza wyników

### Metryki do śledzenia:
- **MSE** (Mean Squared Error) - podstawowa metryka
- **MAE** (Mean Absolute Error) - łatwiejsza interpretacja
- **RMSE** (Root MSE) - w tej samej skali co xG
- **R²** - jakość dopasowania
- **Korelacja** - między predykcją a prawdą

### Analiza per-player:
```python
# Grupuj sekwencje po zawodnikach
player_predictions = {}
for player in unique_players:
    player_sequences = get_sequences_for_player(player, sequences)
    predictions = model.predict(player_sequences)
    player_predictions[player] = {
        'mean_xg': predictions.mean(),
        'max_xg': predictions.max(),
        'high_value_sequences': (predictions > 0.2).sum(),
        'total_sequences': len(predictions)
    }

# Ranking zawodników
ranking = pd.DataFrame(player_predictions).T
ranking = ranking.sort_values('mean_xg', ascending=False)
print(ranking)
```

---

## 🔬 Eksperymenty do przeprowadzenia

### 1. Długość sekwencji
Przetestuj różne wartości `SEQUENCE_LENGTH`:
- [ ] 5 akcji
- [ ] 10 akcji (default)
- [ ] 15 akcji
- [ ] 20 akcji

### 2. Cechy wejściowe
Eksperymentuj z dodatkowymi cechami:
- [ ] Prędkość piłki
- [ ] Liczba obrońców w pobliżu
- [ ] Strefa boiska (defensywna/środkowa/ofensywna)
- [ ] Typ podania (ground/high/through)

### 3. Etykiety
Wypróbuj alternatywne definicje wartości sekwencji:
- [ ] Binarne (strzał / brak strzału)
- [ ] xG jako regressja (obecne)
- [ ] Kategorie (low/medium/high threat)
- [ ] xT (expected threat) zamiast xG

### 4. Architektury
- [ ] Porównaj LSTM vs Transformer
- [ ] Wypróbuj CNN-LSTM hybrydę
- [ ] Dodaj attention mechanism do LSTM
- [ ] Eksperymentuj z różnymi rozmiarami sieci

### 5. Augmentacja danych
- [ ] Odwróć kierunek akcji (mirror)
- [ ] Dodaj szum do współrzędnych
- [ ] Usuwaj losowe akcje z sekwencji

---

## 📚 Przydatne zasoby

### StatsBomb API:
- Dokumentacja: https://github.com/statsbomb/statsbombpy
- Open Data: https://github.com/statsbomb/open-data

### Przykłady użycia:
```python
from statsbombpy import sb

# Pobierz dostępne konkurencje
competitions = sb.competitions()

# Pobierz mecze z sezonu
matches = sb.matches(competition_id=11, season_id=90)

# Pobierz wydarzenia z meczu
events = sb.events(match_id=3788741)
```

### Machine Learning:
- TensorFlow docs: https://www.tensorflow.org/guide
- Keras examples: https://keras.io/examples/

---

## 🐛 Debugging

### Typowe problemy:

1. **Shape mismatch**
   - Sprawdź wymiary X: `(n_samples, sequence_length, n_features)`
   - Sprawdź wymiary y: `(n_samples,)`

2. **NaN values**
   - Wypełnij brakujące wartości: `df.fillna(0)`
   - Sprawdź czy wszystkie cechy są numeryczne

3. **Overfitting**
   - Zwiększ dropout rate
   - Dodaj regularizację L2
   - Użyj więcej danych

4. **Underfitting**
   - Zwiększ rozmiar sieci
   - Trenuj dłużej
   - Dodaj więcej cech

---

## ✅ Checklist implementacji

### Faza 1: Przygotowanie danych
- [ ] Pobranie danych StatsBomb
- [ ] Implementacja `extract_possessions()`
- [ ] Implementacja `create_features()`
- [ ] Implementacja `create_labels()`
- [ ] Walidacja kształtów danych

### Faza 2: Model baseline
- [ ] Uruchomienie LSTM z domyślnymi parametrami
- [ ] Trenowanie na małym zbiorze danych
- [ ] Ewaluacja baseline metrics
- [ ] Zapisanie wyników

### Faza 3: Optymalizacja
- [ ] Hyperparameter tuning
- [ ] Próba różnych architektur
- [ ] Cross-validation
- [ ] Selekcja najlepszego modelu

### Faza 4: Analiza
- [ ] Predykcje dla wszystkich zawodników
- [ ] Ranking zawodników
- [ ] Identyfikacja high-value sequences
- [ ] Wizualizacja wyników

### Faza 5: Dokumentacja
- [ ] Opis finalnego modelu
- [ ] Raport z wyników
- [ ] Wnioski i insights
- [ ] Propozycje przyszłych usprawnień

---

## 💡 Pomysły na rozszerzenia

1. **Multi-task learning**: Przewiduj zarówno xG jak i czy będzie assist
2. **Player embeddings**: Naucz reprezentacji zawodników
3. **Team style**: Analiza stylu gry zespołów
4. **Temporal analysis**: Jak wartość sekwencji zmienia się w czasie meczu
5. **Interactive dashboard**: Streamlit/Dash app do eksploracji

---

**Data ostatniej aktualizacji:** 2025-01-27

