# Models Directory

Ten folder zawiera wytrenowane modele oraz powiązane artefakty.

## 📁 Struktura

```
models/
├── best_model.h5           # Najlepszy model z treningu (auto-zapisywany)
├── xt_models/              # Modele Expected Threat (xT)
│   ├── 4_16.pkl           # xT model dla konkretnej ligi/sezonu
│   └── default_xt_model.json
└── *.json                  # Metryki treningu
```

## 🔄 Automatyczne Zapisywanie

Podczas treningu (`ModelTrainer`), następujące pliki są automatycznie tworzone:

### **best_model.h5**
- Zapisywany przez `ModelCheckpoint` callback
- Zawiera wagi modelu z **najniższą** `val_loss` podczas całego treningu
- To jest model, którego powinieneś używać do predykcji!

### **Ręczne zapisywanie**
Możesz też zapisać model ręcznie:
```python
trainer.save_model("custom_name.h5")
```

## 🚫 .gitignore

**Ważne**: Wszystkie pliki `.h5`, `.keras`, `.pb` są ignorowane przez Git (patrz `.gitignore`).
To znaczy, że modele **nie będą** commitowane do repozytorium, ponieważ są zazwyczaj duże (MB-GB).

## 📊 Expected Threat (xT) Models

Folder `xt_models/` zawiera:
- Wytrenowane siatki xT dla różnych lig/sezonów
- Format: `{season_id}_{competition_id}.pkl`
- Automatycznie cachowane po pierwszym wytrenowaniu
- Można je bezpiecznie commitować do repo (małe pliki)

## 🔧 Użycie

### Załaduj najlepszy model:
```python
import tensorflow as tf

model = tf.keras.models.load_model('models/best_model.h5')
predictions = model.predict(X_test)
```

### Załaduj xT model:
```python
from src.xthreat import get_xt_model_for_competition

xt_model = get_xt_model_for_competition(season_id=4, competition_id=16)
```

## 🧹 Czyszczenie

Aby usunąć wszystkie wytrenowane modele:
```bash
rm models/*.h5 models/*.keras models/*.json
```

xT modele pozostaną nienaruszone w `xt_models/`.

