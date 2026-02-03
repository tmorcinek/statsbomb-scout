# Models Directory

Ten folder zawiera wytrenowane modele oraz powiązane artefakty.

## 📁 Struktura

```
models/
├── lstm/                   # LSTM models
│   ├── best_model.keras
│   ├── metrics.json
│   └── *.png              # Visualizations
├── attention_lstm/         # Attention LSTM models  
│   ├── best_model.keras
│   ├── metrics.json
│   └── *.png
├── transformer/            # Transformer models
│   ├── best_model.keras
│   ├── metrics.json
│   └── *.png
└── xt_models/              # Expected Threat (xT) models
    └── default_xt_model.json
```

## 🔄 Automatyczne Zapisywanie

Podczas treningu przez `train_models.py`, każdy model jest zapisywany w osobnym folderze:

### **{model_type}_{name}_{timestamp}/**
- `best_model.keras` - Model z najniższą `val_loss` podczas treningu
- `metrics.json` - Metryki treningu i walidacji
- `*.png` - Wizualizacje najlepszych sekwencji
- `config.json` - Konfiguracja modelu

## 🚫 .gitignore

**Ważne**: Wszystkie pliki `.h5`, `.keras`, `.pb` są ignorowane przez Git (patrz `.gitignore`).
To znaczy, że modele **nie będą** commitowane do repozytorium, ponieważ są zazwyczaj duże (MB-GB).

## 📊 Expected Threat (xT) Models

Folder `xt_models/` zawiera:
- Wytrenowane siatki xT dla różnych lig/sezonów
- Format: `default_xt_model.json`
- Automatycznie cachowane po pierwszym wytrenowaniu
- Można je bezpiecznie commitować do repo (małe pliki)

## 🔧 Użycie

### Trenowanie modeli:
```bash
python train_models.py
```

### Załaduj wytrenowany model:
```python
from src.ml.models.model_factory import load_model

model = load_model('models/attention_lstm/best_model.keras')
predictions = model.predict(X_test)
```

### Załaduj xT model:
```python
from src.ml.xthreat import get_default_xt_model

xt_model = get_default_xt_model()
```

## 🧹 Czyszczenie

Aby usunąć wszystkie wytrenowane modele:
```bash
rm -rf models/lstm models/attention_lstm models/transformer
```

xT modele pozostaną nienaruszone w `xt_models/`.

