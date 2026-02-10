# Preprocessing Pipeline - Quick Reference

**Projekt:** StatsBomb Scout  
**Data:** 10 lutego 2026

---

## 🚀 TL;DR

```
StatsBomb Events (JSON)
    ↓ SPADL conversion
SPADL Actions (23 types)
    ↓ Group by possession
Possessions (avg 9 actions)
    ↓ Filter (≥6 actions)
Valid Possessions (75%)
    ↓ Feature engineering (46 features)
Normalized Features
    ↓ Sliding window (training) / Single (validation)
Sequences (6, 46)
    ↓ Label = max(xG, xT)
Training Data (X, y)
```

---

## 📊 8 Kroków Preprocessing

| # | Krok | Input | Output | Funkcja |
|---|------|-------|--------|---------|
| 1 | SPADL Conversion | StatsBomb JSON | SPADL Actions | `extract_actions_from_events()` |
| 2 | Enrich | Actions | Actions + xG, pressure | `enrich_actions_with_event_data()` |
| 3 | Extract Possessions | Actions | Dict[possession_id → actions] | `extract_possessions()` |
| 4 | Normalize Pitch | Possessions | Possessions (attacking L→R) | `normalize_pitch()` |
| 5 | Feature Engineering | Actions | Actions + 7 computed features | `_update_action()` |
| 6 | Normalize Features | Actions + features | (n_actions, 46) | `_normalize_features()` |
| 7 | Create Sequences | Features | (n_sequences, 6, 46) | `_create_sequences()` |
| 8 | Generate Labels | Actions | (n_sequences,) | `get_actions_value()` |

---

## 🔢 Numbers at a Glance

### Typical Match
```
Events:           2,500
Actions (SPADL):  2,300  (after conversion)
Possessions:      ~90    (grouped)
Valid (≥6):       ~67    (filtered, 75%)
Avg length:       9.2    (actions per possession)

Training mode:
  Sequences:      ~450   (sliding window)
  
Validation mode:
  Sequences:      ~67    (one per possession)
```

### Full Dataset (282 matches)
```
Training (70%):   197 matches → ~90,000 sequences
Validation (20%): 56 matches  → ~3,700 sequences
Test (10%):       28 matches  → ~1,900 sequences
```

---

## 🎯 46 Features Breakdown

```
[0-3]   Spatial (4):     start_x, start_y, end_x, end_y (normalized to [0,1])
[4-6]   Geometric (3):   distance, sin(angle), cos(angle)
[7]     Temporal (1):    time_diff (capped at 10s, normalized)
[8-10]  Contextual (3):  under_pressure, counterpress, opposite_action
[11-33] type_id (23):    pass, dribble, shot, tackle, ... (one-hot)
[34-39] result_id (6):   success, fail, offside, ... (one-hot)
[40-45] bodypart_id (6): foot, head, other, ... (one-hot)

Total: 4 + 3 + 1 + 3 + 23 + 6 + 6 = 46
```

---

## 🔄 Training vs Validation Mode

### Training Mode
```python
# Long possession (n ≥ 6): Sliding window
[a1, a2, a3, a4, a5, a6, a7, a8]
  → [a1,a2,a3,a4,a5,a6]  seq_1
  → [a2,a3,a4,a5,a6,a7]  seq_2
  → [a3,a4,a5,a6,a7,a8]  seq_3
Result: 3 sequences

# Short possession (n < 6): Padding
[a1, a2, a3, a4]
  → [a1,a2,a3,a4,0,0]  seq_1 (padded)
Result: 1 sequence
```

### Validation Mode
```python
# Always 1 sequence per possession

# Long possession (n ≥ 6): Last 6 actions
[a1, a2, a3, a4, a5, a6, a7, a8]
  → [a3,a4,a5,a6,a7,a8]  (last 6)

# Short possession (n < 6): Padding
[a1, a2, a3, a4]
  → [a1,a2,a3,a4,0,0]  (padded)
```

**Why different?**
- Training: Data augmentation (more samples)
- Validation: Realistic evaluation (1 prediction per possession)

---

## 🏷️ Label Generation

```python
label = max(sum(xG), xT_last)
```

### Examples

| Scenario | xG sum | xT last | Label | Why |
|----------|--------|---------|-------|-----|
| Goal | 0.85 | 0.20 | **0.85** | max(0.85, 0.20) |
| Shot missed | 0.23 | 0.18 | **0.23** | max(0.23, 0.18) |
| No shot, good position | 0.00 | 0.35 | **0.35** | max(0.00, 0.35) |
| No shot, bad position | 0.00 | 0.05 | **0.05** | max(0.00, 0.05) |

---

## ⚙️ Key Parameters

```python
SEQUENCE_LENGTH = 6           # Window size
MINIMUM_SEQUENCE_LENGTH = 6   # Min possession length
FIELD_LENGTH = 105            # SPADL pitch length
FIELD_WIDTH = 68              # SPADL pitch width
MAX_TIME_DIFF = 10.0          # Cap for time normalization
N_FEATURES = 46               # Total features per action
```

---

## 🔍 Quick Debugging

### Check possession extraction
```python
possessions = extract_possessions(game, events)
print(f"Total: {len(possessions)}")
print(f"Valid (≥6): {sum(1 for p in possessions.values() if len(p) >= 6)}")
```

### Check sequence shapes
```python
X, y, p, m = preprocessor.process_matches(matches, mode=PreprocessingMode.TRAINING)
print(f"X: {X.shape}")  # Should be (N, 6, 46)
print(f"y: {y.shape}")  # Should be (N,)
print(f"Labels range: [{y.min():.3f}, {y.max():.3f}]")
```

### Check feature normalization
```python
# All features should be in reasonable range
print(f"Feature mins: {X.min(axis=(0,1))}")  # Should be mostly 0
print(f"Feature maxs: {X.max(axis=(0,1))}")  # Should be mostly 1
```

---

## 🚨 Common Issues

### Issue: Too few sequences
```
Problem: Only 100 sequences from 200 matches
Cause: minimum_sequence_length too high
Fix: Lower to 4-5 or check data quality
```

### Issue: All labels are 0
```
Problem: Labels = [0.0, 0.0, 0.0, ...]
Cause: xG not calculated or xT model not loaded
Fix: Check xt_model initialization
```

### Issue: NaN in features
```
Problem: RuntimeWarning: invalid value in normalize
Cause: Missing coordinates in events
Fix: Check enrich_actions_with_event_data() merge
```

---

## 📚 Related Functions

### In `possessions_extraction.py`
- `extract_actions_from_events()` - SPADL conversion
- `enrich_actions_with_event_data()` - Add xG, pressure
- `extract_possessions()` - Group by possession
- `normalize_pitch()` - Rotate to L→R attack

### In `sequence_preprocessor.py`
- `_update_action()` - Feature engineering
- `_normalize_features()` - 46 features + one-hot
- `_create_sequences()` - Sliding window
- `_pad_sequence()` - Zero padding

### In `action_valuation.py`
- `calculate_xg_values()` - xG from StatsBomb
- `calculate_xt_values()` - xT from grid
- `get_actions_value()` - Label = max(xG, xT)

---

## 🎓 Key Insights

1. **~25% possessions discarded** (< 6 actions) - acceptable loss
2. **Sliding window in training** → 5-7x more sequences per match
3. **Padding with zeros** → works with LSTM masking
4. **Pitch normalization** → always attack L→R (important!)
5. **46 features** → 11 numerical + 35 categorical (one-hot)
6. **max(xG, xT)** → handles both shots and non-shot possessions

---

**Full documentation:** `docs/TRAINING_PIPELINE.md` (Section: Szczegółowy Pipeline Preprocessing)

