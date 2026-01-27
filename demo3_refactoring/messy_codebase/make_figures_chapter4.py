# Figures for Chapter 4
# State-level visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Same paths as chapter 3 but defined again
DATA_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/data_processed/analysis_dataset.csv'
FIG_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/figures'

def load_data():
    return pd.read_csv(DATA_PATH)

def figure4_state_satisfaction_map():
    """Figure 4: Map of satisfaction by state"""
    # Would need geopandas for this
    # Skipping actual implementation
    print("Note: Map figure requires geopandas, skipping...")

def figure5_unemployment_satisfaction():
    """Figure 5: State unemployment vs mean satisfaction"""
    df = load_data()

    # Aggregate to state level
    state_df = df.groupby('state').agg({
        'satisfaction': 'mean',
        'unemployment_rate': 'first'  # Same for all in state
    }).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(state_df['unemployment_rate'], state_df['satisfaction'])
    ax.set_xlabel('State Unemployment Rate (%)', fontsize=12)
    ax.set_ylabel('Mean Life Satisfaction', fontsize=12)
    ax.set_title('Figure 5: State Unemployment and Life Satisfaction')

    # Add state labels (messy but reviewer asked for it)
    for _, row in state_df.iterrows():
        ax.annotate(row['state'][:2], (row['unemployment_rate'], row['satisfaction']),
                   fontsize=6, alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'{FIG_PATH}/fig5_unemployment.png', dpi=300)
    plt.savefig(f'{FIG_PATH}/fig5_unemployment.pdf')
    plt.close()
    print("Saved Figure 5")

def figure6_coefficient_plot():
    """Figure 6: Coefficient plot comparing models"""
    # Hard-coded coefficients from analysis (should read from results...)
    models = ['Model 1', 'Model 2', 'Model 3']
    income_coefs = [0.012, 0.011, 0.009]
    income_se = [0.002, 0.002, 0.002]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(models))
    ax.errorbar(x, income_coefs, yerr=[1.96*se for se in income_se], fmt='o', capsize=5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Income Coefficient')
    ax.set_title('Figure 6: Income Effect Across Model Specifications')

    plt.tight_layout()
    plt.savefig(f'{FIG_PATH}/fig6_coef_plot.png', dpi=300)
    plt.close()
    print("Saved Figure 6")

if __name__ == '__main__':
    figure5_unemployment_satisfaction()
    figure6_coefficient_plot()
