import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.training.base import PipelineStep, Pipeline


class ModelComparisonConfig:

    def __init__(self, model1_name: str, model1_csv: str, model2_name: str, model2_csv: str):
        self.model1_name = model1_name
        self.model1_csv = model1_csv
        self.model2_name = model2_name
        self.model2_csv = model2_csv

    def __repr__(self):
        return f"ModelComparisonConfig({self.model1_name} vs {self.model2_name})"


class ComparisonData:

    def __init__(self, model1_name: str, model2_name: str,
                 df1: pd.DataFrame, df2: pd.DataFrame, merged: pd.DataFrame):
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.df1 = df1
        self.df2 = df2
        self.merged = merged


class LoadCSVsStep(PipelineStep):

    def __init__(self, config: ModelComparisonConfig):
        super().__init__()
        self.config = config

    def process(self, data: Any = None) -> Tuple[str, str, pd.DataFrame, pd.DataFrame]:
        print(f"Loading CSV files...")
        print(f"  Model 1: {self.config.model1_name} from {self.config.model1_csv}")
        print(f"  Model 2: {self.config.model2_name} from {self.config.model2_csv}")

        df1 = pd.read_csv(self.config.model1_csv)
        df2 = pd.read_csv(self.config.model2_csv)

        print(f"  Loaded {len(df1)} possessions from model 1")
        print(f"  Loaded {len(df2)} possessions from model 2")

        return self.config.model1_name, self.config.model2_name, df1, df2


class MergeAndCompareStep(PipelineStep):

    def process(self, data: Tuple[str, str, pd.DataFrame, pd.DataFrame]) -> ComparisonData:
        model1_name, model2_name, df1, df2 = data

        print(f"\nMerging dataframes on possession_id...")

        merged = pd.merge(
            df1[['possession_id', 'value', 'game_id', 'team_name', 'period', 'outcome']],
            df2[['possession_id', 'value']],
            on='possession_id',
            how='inner',
            suffixes=('_model1', '_model2')
        )

        print(f"  Found {len(merged)} common possessions")

        merged['value_diff'] = merged['value_model1'] - merged['value_model2']
        merged['value_diff_abs'] = np.abs(merged['value_diff'])
        merged['value_avg'] = (merged['value_model1'] + merged['value_model2']) / 2

        print(f"\nCalculating statistics...")
        print(f"  Mean absolute difference: {merged['value_diff_abs'].mean():.6f}")
        print(f"  Max absolute difference: {merged['value_diff_abs'].max():.6f}")
        print(f"  Min absolute difference: {merged['value_diff_abs'].min():.6f}")
        print(f"  Std of difference: {merged['value_diff'].std():.6f}")

        return ComparisonData(model1_name, model2_name, df1, df2, merged)


class AnalyzeAgreementStep(PipelineStep):

    def process(self, data: ComparisonData) -> Dict:
        print(f"\nAnalyzing model agreement...")

        merged = data.merged

        top_10_model1 = set(data.df1.nsmallest(10, 'value')['possession_id'])
        top_10_model2 = set(data.df2.nsmallest(10, 'value')['possession_id'])

        agreement_top10 = len(top_10_model1.intersection(top_10_model2))
        print(f"  Top 10 agreement: {agreement_top10}/10 possessions")

        correlation = merged['value_model1'].corr(merged['value_model2'])
        print(f"  Pearson correlation: {correlation:.6f}")

        mae = merged['value_diff_abs'].mean()
        rmse = np.sqrt((merged['value_diff'] ** 2).mean())
        print(f"  MAE: {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")

        model1_higher = (merged['value_diff'] > 0).sum()
        model2_higher = (merged['value_diff'] < 0).sum()
        same = (merged['value_diff'] == 0).sum()

        print(f"\n  Model 1 values higher: {model1_higher} possessions")
        print(f"  Model 2 values higher: {model2_higher} possessions")
        print(f"  Same value: {same} possessions")

        biggest_disagreements = merged.nlargest(10, 'value_diff_abs')[
            ['possession_id', 'value_model1', 'value_model2', 'value_diff', 'team_name', 'outcome']
        ]
        print(f"\n  Top 10 biggest disagreements:")
        print(biggest_disagreements.to_string())

        return {
            'model1_name': data.model1_name,
            'model2_name': data.model2_name,
            'total_compared': len(merged),
            'top10_agreement': agreement_top10,
            'correlation': float(correlation),
            'mae': float(mae),
            'rmse': float(rmse),
            'model1_higher_count': int(model1_higher),
            'model2_higher_count': int(model2_higher),
            'same_count': int(same),
            'mean_diff': float(merged['value_diff'].mean()),
            'std_diff': float(merged['value_diff'].std()),
            'biggest_disagreements': biggest_disagreements.to_dict('records')
        }


class VisualizeDifferencesStep(PipelineStep):

    def __init__(self, output_dir: str = "analysis_results"):
        super().__init__()
        self.output_dir = output_dir

    def process(self, data: Tuple[ComparisonData, Dict]) -> Dict:
        comparison_data, stats = data
        merged = comparison_data.merged

        print(f"\nCreating visualizations...")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        axes[0, 0].scatter(merged['value_model1'], merged['value_model2'], alpha=0.6)
        axes[0, 0].plot([merged['value_model1'].min(), merged['value_model1'].max()],
                        [merged['value_model1'].min(), merged['value_model1'].max()],
                        'r--', label='Perfect agreement')
        axes[0, 0].set_xlabel(f'{comparison_data.model1_name} value')
        axes[0, 0].set_ylabel(f'{comparison_data.model2_name} value')
        axes[0, 0].set_title('Value Predictions Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].hist(merged['value_diff'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', label='No difference')
        axes[0, 1].set_xlabel(f'Difference ({comparison_data.model1_name} - {comparison_data.model2_name})')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Value Differences')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].scatter(merged['value_avg'], merged['value_diff_abs'], alpha=0.6)
        axes[1, 0].set_xlabel('Average value')
        axes[1, 0].set_ylabel('Absolute difference')
        axes[1, 0].set_title('Absolute Difference vs Average Value')
        axes[1, 0].grid(alpha=0.3)

        top_n = min(20, len(merged))
        top_disagreements = merged.nlargest(top_n, 'value_diff_abs')
        x_pos = np.arange(top_n)
        axes[1, 1].barh(x_pos, top_disagreements['value_diff'], alpha=0.7)
        axes[1, 1].set_yticks(x_pos)
        axes[1, 1].set_yticklabels(top_disagreements['possession_id'].astype(str), fontsize=8)
        axes[1, 1].axvline(0, color='red', linestyle='--')
        axes[1, 1].set_xlabel(f'Difference ({comparison_data.model1_name} - {comparison_data.model2_name})')
        axes[1, 1].set_ylabel('Possession ID')
        axes[1, 1].set_title(f'Top {top_n} Biggest Disagreements')
        axes[1, 1].grid(alpha=0.3, axis='x')

        fig.suptitle(f'Model Comparison: {comparison_data.model1_name} vs {comparison_data.model2_name}',
                     fontsize=16, fontweight='bold')

        fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.05, wspace=0.25, hspace=0.3)

        output_path = Path(self.output_dir) / f"comparison_{comparison_data.model1_name}_vs_{comparison_data.model2_name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved visualization to {output_path}")

        stats['visualization_path'] = str(output_path)
        return stats


class SaveComparisonResultsStep(PipelineStep):

    def __init__(self, output_file: str):
        super().__init__()
        self.output_file = output_file

    def process(self, results: Dict) -> Dict:
        print(f"\nSaving comparison results to {self.output_file}...")

        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved successfully")
        return results


def create_comparison_pipeline(
        config: ModelComparisonConfig,
        output_dir: str = "analysis_results",
        output_file: Optional[str] = None
) -> Pipeline:

    if output_file is None:
        output_file = f"{output_dir}/comparison_{config.model1_name}_vs_{config.model2_name}.json"

    pipeline = Pipeline(name=f"ComparisonPipeline_{config.model1_name}_vs_{config.model2_name}")
    pipeline.add_step(LoadCSVsStep(config))
    pipeline.add_step(MergeAndCompareStep())

    analyze_step = AnalyzeAgreementStep()
    visualize_step = VisualizeDifferencesStep(output_dir)

    class CombineSteps(PipelineStep):
        def process(self, data: ComparisonData):
            stats = analyze_step.process(data)
            return visualize_step.process((data, stats))

    pipeline.add_step(CombineSteps())
    pipeline.add_step(SaveComparisonResultsStep(output_file))

    return pipeline


if __name__ == "__main__":
    config = ModelComparisonConfig(
        model1_name="transformer_baseline",
        model1_csv="generated/transformer_transformer_baseline_20260214_170158/top_possessions_match_3943043.csv",
        model2_name="attention_lstm_large",
        model2_csv="generated/attention_lstm_attention_lstm_large_20260214_170158/top_possessions_match_3943043.csv"
    )

    pipeline = create_comparison_pipeline(config)
    results = pipeline(None)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nKey metrics:")
    print(f"  Total possessions compared: {results['total_compared']}")
    print(f"  Top 10 agreement: {results['top10_agreement']}/10")
    print(f"  Correlation: {results['correlation']:.4f}")
    print(f"  MAE: {results['mae']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"\nVisualization: {results['visualization_path']}")
    print(f"Results JSON: analysis_results/comparison_{config.model1_name}_vs_{config.model2_name}.json")





