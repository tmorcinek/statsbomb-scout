#!/usr/bin/env python3
"""Analyze and compare best models from each category."""

import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def main():
    # Load all CSV files
    print("Loading CSV files...")
    df_main = pd.read_csv('models/model_comparison.csv')
    df_lstm = pd.read_csv('models/model_comparison_lstm.csv')
    df_transformer = pd.read_csv('models/model_comparison_transformer.csv')

    # Combine all dataframes
    df_all = pd.concat([df_main, df_lstm, df_transformer], ignore_index=True)

    # Remove duplicates (in case some models appear in multiple files)
    df_all = df_all.drop_duplicates(subset=['name', 'model_type'], keep='first')

    print("=" * 100)
    print("ALL MODELS LOADED")
    print("=" * 100)
    print(f"\nTotal models: {len(df_all)}")
    print(f"Model types: {df_all['model_type'].unique()}")
    print(f"\nLSTM models: {len(df_all[df_all['model_type'] == 'lstm'])}")
    print(f"Attention LSTM models: {len(df_all[df_all['model_type'] == 'attention_lstm'])}")
    print(f"Transformer models: {len(df_all[df_all['model_type'] == 'transformer'])}")
    print(f"BiGRU models: {len(df_all[df_all['model_type'] == 'bigru'])}")

    # Filter successful models only
    df_success = df_all[df_all['status'] == 'success'].copy()

    # Group by model type and find best model by test_mae
    print("\n" + "=" * 100)
    print("BEST MODELS BY TYPE (sorted by test_mae)")
    print("=" * 100)

    best_models = []

    for model_type in ['lstm', 'attention_lstm', 'transformer', 'bigru']:
        df_type = df_success[df_success['model_type'] == model_type].copy()
        if len(df_type) > 0:
            best = df_type.nsmallest(1, 'test_mae').iloc[0]
            best_models.append(best)
            print(f"\n{model_type.upper()}:")
            print(f"  Best model: {best['name']}")
            print(f"  test_mae: {best['test_mae']:.6f}")
            print(f"  test_rmse: {best['test_rmse']:.6f}")
            print(f"  val_mae: {best['val_mae']:.6f}")

    # Create comparison dataframe
    df_best = pd.DataFrame(best_models)

    # Select relevant columns
    comparison_cols = ['name', 'model_type', 'test_mae', 'test_rmse', 'val_mae', 'test_loss', 'val_loss']
    df_comparison = df_best[comparison_cols].copy()

    # Sort by test_mae
    df_comparison = df_comparison.sort_values('test_mae')

    print("\n" + "=" * 100)
    print("BEST MODELS COMPARISON TABLE")
    print("=" * 100)
    print(df_comparison.to_string(index=False))

    # Save to CSV
    output_file = 'models/best_models_comparison.csv'
    df_comparison.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")

    # More detailed analysis
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS")
    print("=" * 100)

    for idx, row in df_comparison.iterrows():
        print(f"\n{row['model_type'].upper()}: {row['name']}")
        print(f"  Test MAE:  {row['test_mae']:.6f}")
        print(f"  Test RMSE: {row['test_rmse']:.6f}")
        print(f"  Val MAE:   {row['val_mae']:.6f}")
        print(f"  Test Loss: {row['test_loss']:.6f}")
        print(f"  Val Loss:  {row['val_loss']:.6f}")
        generalization_gap = row['val_mae'] - row['test_mae']
        print(f"  Generalization gap (val_mae - test_mae): {generalization_gap:.6f}")

    # Analysis of LSTM variants
    print("\n" + "=" * 100)
    print("LSTM VARIANTS ANALYSIS")
    print("=" * 100)

    lstm_models = df_success[df_success['model_type'] == 'lstm'].copy()
    lstm_models = lstm_models.sort_values('test_mae')

    print(f"\nTop 5 LSTM models by test_mae:")
    top5_lstm = lstm_models.head(5)[['name', 'test_mae', 'test_rmse', 'val_mae']]
    print(top5_lstm.to_string(index=False))

    # Analysis of Transformer variants
    print("\n" + "=" * 100)
    print("TRANSFORMER VARIANTS ANALYSIS")
    print("=" * 100)

    transformer_models = df_success[df_success['model_type'] == 'transformer'].copy()
    transformer_models = transformer_models.sort_values('test_mae')

    print(f"\nAll Transformer models by test_mae:")
    all_transformers = transformer_models[['name', 'test_mae', 'test_rmse', 'val_mae']]
    print(all_transformers.to_string(index=False))

    # Analysis of Attention LSTM variants
    print("\n" + "=" * 100)
    print("ATTENTION LSTM VARIANTS ANALYSIS")
    print("=" * 100)

    attention_lstm_models = df_success[df_success['model_type'] == 'attention_lstm'].copy()
    attention_lstm_models = attention_lstm_models.sort_values('test_mae')

    print(f"\nAll Attention LSTM models by test_mae:")
    all_attention = attention_lstm_models[['name', 'test_mae', 'test_rmse', 'val_mae']]
    print(all_attention.to_string(index=False))

    # Mapping analysis from train_models.py
    print("\n" + "=" * 100)
    print("MODEL CONFIGURATION MAPPING")
    print("=" * 100)

    print("\n📋 LSTM Models:")
    print("  - Created with: create_lstm_configs()")
    print("  - Variants: small/baseline/large × dense64/dense128/no_dense")
    print("  - Parameters: lstm_units (32/64/128), dropout (0.15/0.2/0.3), dense_units (64/128/None)")

    print("\n📋 Attention LSTM Models:")
    print("  - Created with: create_default_configs()")
    print("  - Variants: small/baseline/large")
    print("  - Parameters: lstm_units (32/64/128), dropout (0.15/0.2/0.3)")
    print("  - Special: Returns attention_weights output")

    print("\n📋 Transformer Models:")
    print("  - Created with: create_transformers_config() and create_default_configs()")
    print("  - Variants: small/baseline/large, true_attention")
    print("  - Parameters: num_heads (2/4/8), d_model (64/128/256), ff_dim (256/512/1024)")
    print("  - Blocks: 2-3 transformer blocks")

    print("\n📋 BiGRU Models:")
    print("  - Created with: create_default_configs()")
    print("  - Variants: baseline/medium/large")
    print("  - Parameters: gru_units (64/96/128), attn_hidden (32/48/64)")


if __name__ == "__main__":
    df = pd.read_csv('model_analysis.csv')
    print(df.columns)
    filtered_df = df[['name', 'model_type', 'test_loss', 'test_rmse', 'test_mae']].copy()
    filtered_df = filtered_df.sort_values('test_loss')
    print(filtered_df.to_string(index=False))

    # main()

