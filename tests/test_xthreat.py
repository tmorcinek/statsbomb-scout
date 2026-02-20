"""Test script for xThreat model utilities."""

import os

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch
from socceraction.xthreat import ExpectedThreat

from src.ml.preprocessing.xthreat import get_xt_model_for_competition, get_model_filename, get_default_xt_model


def test_get_model_filename():
    """Test that get_model_filename generates correct filename."""
    # Test basic functionality
    filename = get_model_filename(4, 16)
    assert filename == "4_16.pkl", f"Expected '4_16.pkl', got '{filename}'"

    # Test with different values
    filename2 = get_model_filename(281, 9)
    assert filename2 == "281_9.pkl", f"Expected '281_9.pkl', got '{filename2}'"

    # Test with large numbers
    filename3 = get_model_filename(12345, 67890)
    assert filename3 == "12345_67890.pkl", f"Expected '12345_67890.pkl', got '{filename3}'"

    print("✓ get_model_filename generates correct filenames")


def test_get_xt_model_for_competition_returns_expected_threat():
    """Test that get_xt_model_for_competition returns an ExpectedThreat instance."""
    # Using Champions League 2018/2019 (competition_id=16, season_id=4)
    model = get_xt_model_for_competition(season_id=4, competition_id=16)

    assert model is not None
    assert isinstance(model, ExpectedThreat)
    print("✓ get_xt_model_for_competition returns ExpectedThreat instance")


def test_get_xt_model_for_competition_has_grid():
    """Test that the returned model has xT grid with correct dimensions."""
    model = get_xt_model_for_competition(season_id=4, competition_id=16)

    assert hasattr(model, 'xT')
    assert model.xT is not None
    assert isinstance(model.xT, np.ndarray)

    # Check grid dimensions (l=16, w=12)
    assert model.xT.shape == (12, 16), f"Expected shape (12, 16), got {model.xT.shape}"

    print(f"✓ Model has xT grid with correct shape: {model.xT.shape}")


def test_get_xt_model_for_competition_creates_cache():
    """Test that the model is cached to disk after first training."""
    # Clean up any existing cache
    cache_filename = get_model_filename(4, 16)
    cache_path = os.path.join("models/xt_models", cache_filename)
    if os.path.exists(cache_path):
        os.remove(cache_path)

    # First call should train and cache
    model1 = get_xt_model_for_competition(season_id=4, competition_id=16)

    # Check that cache file was created
    assert os.path.exists(cache_path), "Cache file was not created"

    print("✓ Model cache file created successfully")


def test_get_xt_model_for_competition_uses_cache():
    """Test that subsequent calls use the cached model."""
    # First call (may train or load from cache)
    model1 = get_xt_model_for_competition(season_id=4, competition_id=16)

    # Second call should load from cache
    model2 = get_xt_model_for_competition(season_id=4, competition_id=16)

    # Both should be ExpectedThreat instances
    assert isinstance(model1, ExpectedThreat)
    assert isinstance(model2, ExpectedThreat)

    # Grids should be identical
    assert np.array_equal(model1.xT, model2.xT), "Cached model differs from original"

    print("✓ Cached model loads correctly and matches original")


def test_get_xt_model_for_competition_grid_values():
    """Test that the xT grid has valid probability values."""
    model = get_xt_model_for_competition(season_id=4, competition_id=16)

    # xT values should be between 0 and 1 (probabilities)
    assert np.all(model.xT >= 0), "Grid contains negative values"
    assert np.all(model.xT <= 1), "Grid contains values > 1"

    # Grid should not be all zeros or all ones
    assert np.any(model.xT > 0), "Grid is all zeros"
    assert np.any(model.xT < 1), "Grid is all ones"

    # xT should generally increase towards goal
    # Check that average xT in attacking third > defensive third
    attacking_third = model.xT[:, -5:].mean()  # Last 5 columns
    defensive_third = model.xT[:, :5].mean()   # First 5 columns

    assert attacking_third > defensive_third, \
        f"xT should increase towards goal: attacking={attacking_third:.4f}, defensive={defensive_third:.4f}"

    print(f"✓ xT grid values are valid (min: {model.xT.min():.4f}, max: {model.xT.max():.4f})")
    print(f"  Attacking third avg: {attacking_third:.4f}, Defensive third avg: {defensive_third:.4f}")


def test_visualize_xt_model():
    """Visualize the default xT model as a heatmap on a football pitch."""
    model = get_default_xt_model()

    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='gray', line_alpha=0.5)
    fig, ax = pitch.draw(figsize=(16, 10))

    pos = ax.imshow(model.xT, extent=[0, 120, 0, 80], aspect='auto',
                    cmap='Greens', alpha=0.9, origin='lower', vmin=0, vmax=0.15)

    n_rows, n_cols = model.xT.shape
    cell_width = 120 / n_cols
    cell_height = 80 / n_rows

    for i in range(n_rows + 1):
        ax.plot([0, 120], [i * cell_height, i * cell_height], color='black', linewidth=1)
    for j in range(n_cols + 1):
        ax.plot([j * cell_width, j * cell_width], [0, 80], color='black', linewidth=1)

    for i in range(n_rows):
        for j in range(n_cols):
            value = model.xT[i, j]
            x = (j + 0.5) * cell_width
            y = (i + 0.5) * cell_height
            ax.text(x, y, f'{value:.3f}', ha='center', va='center',
                    fontsize=16, fontweight='normal', color='black')

    plt.colorbar(pos, ax=ax, label='Expected Threat (xT)')
    ax.set_title('Expected Threat Model Visualization', fontsize=16, fontweight='bold')

    os.makedirs('data/test', exist_ok=True)
    output_path = 'data/test/xt_model_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ xT model visualized and saved to {output_path}")
    print(f"  Grid shape: {model.xT.shape}")
    print(f"  Value range: [{model.xT.min():.4f}, {model.xT.max():.4f}]")


if __name__ == "__main__":
    print("Running xThreat model tests...\n")

    results = []

    # Test utility function first
    print("--- Testing utility functions ---")
    results.append(("get_model_filename generates correct names", test_get_model_filename()))

    print("\n--- Testing get_xt_model_for_competition ---")
    print("Using Champions League 2018/2019 data (competition_id=16, season_id=4)\n")
    results.append(("Returns ExpectedThreat instance", test_get_xt_model_for_competition_returns_expected_threat()))
    results.append(("Has correct grid dimensions", test_get_xt_model_for_competition_has_grid()))
    results.append(("Creates cache file", test_get_xt_model_for_competition_creates_cache()))
    results.append(("Uses cached model", test_get_xt_model_for_competition_uses_cache()))
    results.append(("Valid grid values", test_get_xt_model_for_competition_grid_values()))

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'=' * 60}")

    if passed == total:
        print("✅ All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        for name, result in results:
            if not result:
                print(f"   - {name}")
