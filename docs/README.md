# Dokumentacja Projektu

## 📚 Pliki Dokumentacji

### [ATTENTION_MECHANISM.md](ATTENTION_MECHANISM.md)
**Mechanizm Attention - Przewodnik Użytkownika**

Wyjaśnienie jak działa attention w modelach:
- Attention LSTM (tanh scoring)
- Transformer (tanh vs true multi-head attention)
- Masking i padding
- Interpretacja wag attention
- Wizualizacja najlepszych sekwencji
- Przykłady użycia

**Dla kogo:** Użytkownicy, którzy chcą zrozumieć modele i interpretować wyniki.

---

### [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
**Notatki Implementacyjne - Technical Deep Dive**

Szczegóły techniczne implementacji:
- Weryfikacja pipeline i maskingu
- Implementacja custom layers (AttentionLayer)
- Problem z key_dim w MultiHeadAttention
- Attention mask w Transformer
- Multi-output models (value + attention_weights)
- True attention vs tanh attention
- Testy maskingu
- Lessons learned i best practices

**Dla kogo:** Programiści rozwijający projekt, kontrybutorzy, osoby debugujące problemy.

---

## 🎯 Quick Links

### Chcę zrozumieć attention
→ Czytaj: [ATTENTION_MECHANISM.md](ATTENTION_MECHANISM.md)

### Chcę wytrenować modele
→ Komenda: `python train_models.py`  
→ Kod: `train_models.py`

### Chcę zmienić implementację
→ Czytaj: [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)  
→ Kod: `src/ml/models/`

### Chcę uruchomić testy
→ Komenda: `pytest tests/test_masking.py -v`

### Chcę przeanalizować wyniki
→ Notebooki: `analysis_matches.ipynb`, `analysis_final_match.ipynb`

---

## 📊 Struktura Projektu

```
statsbomb-scout/
├── docs/                           # 📚 Ta dokumentacja
│   ├── README.md                   # Ten plik
│   ├── ATTENTION_MECHANISM.md      # Wyjaśnienie attention
│   └── IMPLEMENTATION_NOTES.md     # Notatki techniczne
│
├── src/ml/models/                  # 🧠 Implementacje modeli
│   ├── lstm.py                     # LSTM baseline
│   ├── attention_lstm.py           # Attention LSTM
│   ├── transformer.py              # Transformer
│   └── model_factory.py            # Factory pattern
│
├── tests/                          # ✅ Testy
│   ├── test_masking.py             # Testy maskingu
│   └── ...
│
├── models/                         # 💾 Wytrenowane modele
│   ├── lstm/
│   ├── attention_lstm/
│   ├── transformer/
│   └── README.md                   # Dokumentacja struktury
│
├── train_models.py                 # 🚀 Główny skrypt treningu
├── analyze_results.py              # 📊 Analiza wyników
├── config.py                       # ⚙️ Konfiguracja
└── README.md                       # 📖 Główny README projektu
```

---

## 🔬 Research Questions

Dokumentacja pomaga odpowiedzieć na:

1. **Jak działa attention?**  
   → [ATTENTION_MECHANISM.md](ATTENTION_MECHANISM.md) - Sekcja "Jak Działa Attention?"

2. **Dlaczego padding ma wagę 0?**  
   → [ATTENTION_MECHANISM.md](ATTENTION_MECHANISM.md) - Sekcja "Masking i Padding"

3. **Czym różni się tanh attention od true attention?**  
   → [ATTENTION_MECHANISM.md](ATTENTION_MECHANISM.md) - Tabela porównawcza  
   → [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) - Sekcja 6

4. **Jak dodać własny model?**  
   → [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) - Sekcja "Lessons Learned"

5. **Dlaczego key_dim = d_model // num_heads?**  
   → [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) - Sekcja 3

6. **Jak testować masking?**  
   → [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) - Sekcja 7

---

## 🛠️ Maintenance

### Aktualizacja Dokumentacji

Po zmianach w kodzie, zaktualizuj odpowiednie sekcje:

- Nowy model → dodaj do [ATTENTION_MECHANISM.md](ATTENTION_MECHANISM.md) - "Porównanie Modeli"
- Zmiana w masking → zaktualizuj [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
- Nowy test → dodaj do [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) - "Testy Maskingu"

### Feedback

Jeśli dokumentacja jest niejasna lub brakuje informacji:
1. Dodaj pytanie do sekcji "Research Questions"
2. Rozszerz odpowiednią sekcję
3. Dodaj przykład kodu jeśli potrzeba

---

**Ostatnia aktualizacja:** Luty 2026
