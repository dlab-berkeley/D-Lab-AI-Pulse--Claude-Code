# Figures for Chapter 3
# Main results visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/data_processed/analysis_dataset.csv'
FIG_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/figures'

def load_data():
    return pd.read_csv(DATA_PATH)

def figure1_satisfaction_distribution():
    """Figure 1: Distribution of life satisfaction"""
    df = load_data()

    fig, ax = plt.subplots(figsize=(8, 6))
    df['satisfaction'].hist(bins=10, edgecolor='black', ax=ax)
    ax.set_xlabel('Life Satisfaction (1-10)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Figure 1: Distribution of Life Satisfaction Scores')

    plt.tight_layout()
    plt.savefig(f'{FIG_PATH}/fig1_satisfaction_dist.png', dpi=300)
    plt.savefig(f'{FIG_PATH}/fig1_satisfaction_dist.pdf')  # For LaTeX
    plt.close()
    print("Saved Figure 1")

def figure2_satisfaction_by_education():
    """Figure 2: Satisfaction by education level"""
    df = load_data()

    # Create education categories
    bins = [0, 12, 14, 16, 25]
    labels = ['HS or less', 'Some college', 'Bachelor\'s', 'Graduate']
    df['edu_cat'] = pd.cut(df['education_years'], bins=bins, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    df.groupby('edu_cat')['satisfaction'].mean().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_xlabel('Education Level', fontsize=12)
    ax.set_ylabel('Mean Life Satisfaction', fontsize=12)
    ax.set_title('Figure 2: Life Satisfaction by Education Level')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{FIG_PATH}/fig2_satisfaction_education.png', dpi=300)
    plt.savefig(f'{FIG_PATH}/fig2_satisfaction_education.pdf')
    plt.close()
    print("Saved Figure 2")

def figure3_income_satisfaction_scatter():
    """Figure 3: Income vs Satisfaction"""
    df = load_data()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['income']/1000, df['satisfaction'], alpha=0.3, s=10)
    ax.set_xlabel('Household Income ($1000s)', fontsize=12)
    ax.set_ylabel('Life Satisfaction', fontsize=12)
    ax.set_title('Figure 3: Income and Life Satisfaction')

    # Add trend line
    z = np.polyfit(df['income'].dropna()/1000, df.loc[df['income'].notna(), 'satisfaction'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['income'].min()/1000, df['income'].max()/1000, 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Linear fit')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{FIG_PATH}/fig3_income_scatter.png', dpi=300)
    plt.savefig(f'{FIG_PATH}/fig3_income_scatter.pdf')
    plt.close()
    print("Saved Figure 3")

def make_all_chapter3_figures():
    print("Generating Chapter 3 figures...")
    figure1_satisfaction_distribution()
    figure2_satisfaction_by_education()
    figure3_income_satisfaction_scatter()
    print("Done!")

if __name__ == '__main__':
    make_all_chapter3_figures()
