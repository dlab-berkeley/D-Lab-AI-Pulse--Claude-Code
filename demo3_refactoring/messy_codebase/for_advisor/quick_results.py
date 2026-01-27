# Quick script to generate results for advisor meeting
# Not for thesis - just for discussion

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/PhDStudent/Desktop/thesis_final/data_processed/analysis_dataset.csv')

# Quick summary
print("N =", len(df))
print("\nSatisfaction by employment:")
print(df.groupby('employed')['satisfaction'].mean())

# Quick figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df['satisfaction'], bins=10)
axes[0].set_title('Satisfaction Distribution')

axes[1].scatter(df['income'], df['satisfaction'], alpha=0.3)
axes[1].set_title('Income vs Satisfaction')

plt.savefig('C:/Users/PhDStudent/Desktop/thesis_final/for_advisor/quick_fig.png')
print("\nSaved quick_fig.png")
