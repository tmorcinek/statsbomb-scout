# Roadmap Projektu Statsbomb Scout

## Wprowadzenie

Ten plik zawiera opis zadań do wykonania oraz plan rozwoju projektu. Projekt dotyczy analizy danych piłkarskich z wykorzystaniem StatsBomb.

## Zadania do wykonania

1. **Dodanie paddingu i maskowania**  
   - Opis: Włączenie większej ilośći posiadań do sekwencji poprzez zastosowanie paddingu i maskowania w modelu.

2. **Refaktoring/Optymalizacja SequencePreprocessor**
   - Opis: Poprawa sequence preprocessor w celu zwiększenia wydajności i czytelności kodu oraz by łatwiej było stworzyć automat to testowania róznych ustawień jeśli chodzi o testowanie wyników.

3. **Refaktoring - dodanie kroków do SequencePreprocessor z możliwością zapisu każdego kroku**  
   - Opis: Rozszerzenie klasy o możliwość zapisywania wyników pośrednich na każdym etapie przetwarzania, np. po ekstrakcji posiadania, cech itp.

---


6. **Dodanie transformera z attention**  
   - Opis: Implementacja modelu Transformer z mechanizmem attention do predykcji na podstawie sekwencji akcji.

7. **Testowanie różnych labelek**  
   - Opis: Eksperymentowanie z różnymi sposobami tworzenia etykiet, np. na podstawie xG, xT, wyników meczu itp.

8. **Automat do testowania modeli z różnymi parametrami i zapisania najlepszego modelu do pliku**  
   - Opis: Stworzenie skryptu automatyzującego trening modeli z różnymi hiperparametrami, ewaluację i zapis najlepszego modelu.

5. **Zmiana sposobu wyliczania labelek: ostatni krok w sekwencji nieokreślony**  
   - Opis: Modyfikacja logiki tworzenia etykiet, gdzie ostatni krok w sekwencji jest nieokreślony (nie wiadomo czy został oddany celny strzał, czy w ogóle został oddany strzał).


## Plan rozwoju

- **Faza 1:** Optymalizacja i refaktoryzacja SequencePreprocessor (punkty 1-3).
- **Faza 2:** Walidacja etykiet i dodanie nowych modeli (punkty 4-6).
- **Faza 3:** Eksperymenty z etykietami i automatyzacja treningu (punkty 7-8).
- **Faza 4:** Integracja z wizualizacjami i raportami.
