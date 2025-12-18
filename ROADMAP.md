# Roadmap Projektu Statsbomb Scout

## Wprowadzenie

Ten plik zawiera opis zadań do wykonania oraz plan rozwoju projektu. Projekt dotyczy analizy danych piłkarskich z wykorzystaniem StatsBomb.

## Zadania do wykonania

### 1. Wyciąganie sekwencji zakończonych strzałem

- Opis: Zaimplementować funkcjonalność do ekstrakcji sekwencji akcji z meczu, które kończą się strzałem.
- Sposób realizacji:
    - Analiza danych eventów z StatsBomb.
    - Grupowanie akcji w sekwencje.
    - Filtrowanie sekwencji kończących się typem "shot".
- Priorytet: Wysoki

### 2. Wyciąganie sekwencji zakończonych golem

- Opis: Zaimplementować funkcjonalność do ekstrakcji sekwencji akcji z meczu, które kończą się golem.
- Sposób realizacji:
    - Podobnie jak powyżej, ale filtrowanie sekwencji kończących się typem "goal".
    - Uwzględnienie kontekstu meczu (np. drużyna, pozycja).
- Priorytet: Wysoki

### 3. Walidacja preprocessingu

- Opis: Zbadanie, jak generują się akcje na podstawie event groupingu.
- Sposób realizacji:
    - Przegląd kodu w `src/ml/preprocessing.py` lub podobnych plikach.
    - Uruchomienie testów w `tests/test_preprocessing.py`.
    - Analiza przykładowych danych wejściowych i wyjściowych.
    - Wizualizacja grupowania eventów w akcje.
- Priorytet: Średni

## Plan rozwoju

- **Faza 1:** Implementacja ekstrakcji sekwencji (punkty 1 i 2).
- **Faza 2:** Walidacja i optymalizacja preprocessingu (punkt 3).
- **Faza 3:** Integracja z modelami ML (np. LSTM, Transformer) do predykcji.
- **Faza 4:** Dodanie wizualizacji i raportów.

