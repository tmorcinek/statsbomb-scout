"""
Analysis script for comparing trained models.

Usage:
    python analyze_results.py
    python analyze_results.py --comparison-file models/model_comparison.csv
    python analyze_results.py --plot
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(comparison_file: str = "models/model_comparison.csv") -> pd.DataFrame:
    """Load model comparison results."""
    df = pd.read_csv(comparison_file)
    return df


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    # Overall stats
    print(f"\nTotal models trained: {len(df)}")
    print(f"Successful: {(df['status'] == 'success').sum()}")
    print(f"Failed: {(df['status'] != 'success').sum()}")

    # Filter successful models
    df_success = df[df['status'] == 'success'].copy()

    if len(df_success) == 0:
        print("\nNo successful models to analyze.")
        return

    # Best models by metric
    print("\n" + "-"*80)
    print("BEST MODELS BY METRIC")
    print("-"*80)

    metrics = ['test_mae', 'test_rmse', 'test_loss']
    for metric in metrics:
        best = df_success.nsmallest(1, metric).iloc[0]
        print(f"\nBest {metric.upper()}:")
        print(f"  Model: {best['name']}")
        print(f"  Type: {best['model_type']}")
        print(f"  {metric}: {best[metric]:.6f}")
        print(f"  Directory: {best['output_dir']}")

    # Model type comparison
    print("\n" + "-"*80)
    print("COMPARISON BY MODEL TYPE")
    print("-"*80)

    grouped = df_success.groupby('model_type').agg({
        'test_mae': ['mean', 'std', 'min'],
        'test_rmse': ['mean', 'std', 'min'],
        'test_loss': ['mean', 'std', 'min']
    }).round(6)

    print(f"\n{grouped}")

    # Top 10 models
    print("\n" + "-"*80)
    print("TOP 10 MODELS (by test MAE)")
    print("-"*80)

    top10 = df_success.nsmallest(16, 'test_mae')[
        ['name', 'model_type', 'test_mae', 'test_rmse', 'val_mae']
    ]
    print(f"\n{top10.to_string(index=False)}")


def plot_comparison(df: pd.DataFrame, output_dir: str = "models/analysis"):
    """Create comparison plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Filter successful models
    df_success = df[df['status'] == 'success'].copy()

    if len(df_success) == 0:
        print("\nNo successful models to plot.")
        return

    sns.set_style("whitegrid")

    # 1. MAE comparison by model type
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Test MAE boxplot
    ax = axes[0, 0]
    df_success.boxplot(column='test_mae', by='model_type', ax=ax)
    ax.set_title('Test MAE by Model Type')
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Test MAE')
    plt.sca(ax)
    plt.xticks(rotation=45)

    # Test RMSE boxplot
    ax = axes[0, 1]
    df_success.boxplot(column='test_rmse', by='model_type', ax=ax)
    ax.set_title('Test RMSE by Model Type')
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Test RMSE')
    plt.sca(ax)
    plt.xticks(rotation=45)

    # Validation vs Test MAE
    ax = axes[1, 0]
    for model_type in df_success['model_type'].unique():
        subset = df_success[df_success['model_type'] == model_type]
        ax.scatter(subset['val_mae'], subset['test_mae'], label=model_type, alpha=0.6, s=100)
    ax.plot([df_success['val_mae'].min(), df_success['val_mae'].max()],
            [df_success['val_mae'].min(), df_success['val_mae'].max()],
            'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Validation MAE')
    ax.set_ylabel('Test MAE')
    ax.set_title('Validation vs Test MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Model ranking
    ax = axes[1, 1]
    top_models = df_success.nsmallest(15, 'test_mae')
    colors = {'lstm': 'blue', 'attention_lstm': 'green', 'transformer': 'red'}
    bar_colors = [colors.get(mt, 'gray') for mt in top_models['model_type']]

    ax.barh(range(len(top_models)), top_models['test_mae'], color=bar_colors, alpha=0.6)
    ax.set_yticks(range(len(top_models)))
    ax.set_yticklabels(top_models['name'], fontsize=8)
    ax.set_xlabel('Test MAE')
    ax.set_title('Top 15 Models by Test MAE')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_file = f"{output_dir}/model_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {output_file}")

    # 2. Detailed metrics heatmap
    if len(df_success) > 1:
        fig, ax = plt.subplots(figsize=(12, int(max(6, len(df_success) * 0.4))))

        # Prepare data for heatmap
        heatmap_data = df_success[['name', 'val_mae', 'val_rmse', 'test_mae', 'test_rmse']].set_index('name')

        # Normalize for better visualization
        heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

        sns.heatmap(heatmap_normalized, annot=heatmap_data.values, fmt='.4f',
                    cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Normalized Score'})
        ax.set_title('Model Performance Heatmap (lower is better)', fontsize=14, pad=20)

        plt.tight_layout()
        output_file = f"{output_dir}/metrics_heatmap.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved metrics heatmap to {output_file}")

    # 3. Model type averages
    fig, ax = plt.subplots(figsize=(10, 6))

    type_means = df_success.groupby('model_type')[['test_mae', 'test_rmse']].mean()
    type_stds = df_success.groupby('model_type')[['test_mae', 'test_rmse']].std()

    x = range(len(type_means))
    width = 0.35

    ax.bar([i - width/2 for i in x], type_means['test_mae'], width,
           yerr=type_stds['test_mae'], label='MAE', alpha=0.8, capsize=5)
    ax.bar([i + width/2 for i in x], type_means['test_rmse'], width,
           yerr=type_stds['test_rmse'], label='RMSE', alpha=0.8, capsize=5)

    ax.set_xlabel('Model Type')
    ax.set_ylabel('Error')
    ax.set_title('Average Test Errors by Model Type (with std)')
    ax.set_xticks(x)
    ax.set_xticklabels(type_means.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = f"{output_dir}/type_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved type comparison to {output_file}")

    print(f"\n✓ All plots saved to {output_dir}/")


def export_best_models(df: pd.DataFrame, output_file: str = "models/best_models.json"):
    """Export information about best models."""
    df_success = df[df['status'] == 'success'].copy()

    if len(df_success) == 0:
        print("\nNo successful models to export.")
        return

    best_models = {
        'best_overall': df_success.nsmallest(1, 'test_mae').iloc[0].to_dict(),
        'best_by_type': {},
        'top_10': df_success.nsmallest(10, 'test_mae').to_dict(orient='records')
    }

    for model_type in df_success['model_type'].unique():
        subset = df_success[df_success['model_type'] == model_type]
        best_models['best_by_type'][model_type] = subset.nsmallest(1, 'test_mae').iloc[0].to_dict()

    with open(output_file, 'w') as f:
        json.dump(best_models, f, indent=2)

    print(f"\n✓ Saved best models info to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze model training results')

    parser.add_argument(
        '--comparison-file',
        type=str,
        default='models/ComparisonPipeline_LSTM_DenseUnits.csv',
        help='Path to model comparison CSV file'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate comparison plots'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/analysis',
        help='Output directory for plots'
    )

    parser.add_argument(
        '--export-best',
        action='store_true',
        help='Export best models information to JSON'
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.comparison_file}...")
    df = load_results(args.comparison_file)

    # Print summary
    print_summary(df)

    # Generate plots
    if args.plot:
        print(f"\nGenerating comparison plots...")
        plot_comparison(df, args.output_dir)

    # Export best models
    if args.export_best:
        export_best_models(df)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
