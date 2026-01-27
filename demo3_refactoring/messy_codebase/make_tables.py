# Generate LaTeX tables for thesis
# Run after all analyses complete

import pandas as pd
import numpy as np

OUTPUT_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/outputs'
TABLE_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/tables'

def table1_to_latex():
    """Convert summary stats to LaTeX"""
    df = pd.read_csv(f'{OUTPUT_PATH}/table1_summary.csv')

    latex = df.to_latex(index=False, float_format='%.2f')

    with open(f'{TABLE_PATH}/table1.tex', 'w') as f:
        f.write(latex)

    print("Generated table1.tex")

def combine_regression_results():
    """Create Table 2: Main regression results (Models 1-3)"""
    # This is hacky - parsing text output files
    # Should really store coefficients in structured format

    # Hard-coded for now (would parse from model output files)
    table_data = {
        'Variable': ['Age', 'Female', 'Education (years)', 'Income ($10k)',
                     'Employed', 'Trust Index', 'Constant', 'N', 'R-squared'],
        'Model 1': ['0.02**', '-0.15', '0.08***', '0.012***',
                    '', '', '4.52***', '2,847', '0.089'],
        'Model 2': ['0.02**', '-0.12', '0.07***', '0.011***',
                    '0.45***', '', '4.21***', '2,847', '0.112'],
        'Model 3': ['0.02*', '-0.10', '0.06***', '0.009***',
                    '0.38***', '0.52***', '3.15***', '2,847', '0.156']
    }

    df = pd.DataFrame(table_data)

    # Save as CSV for checking
    df.to_csv(f'{TABLE_PATH}/table2_regressions.csv', index=False)

    # Generate LaTeX
    latex = """\\begin{table}[htbp]
\\centering
\\caption{Determinants of Life Satisfaction}
\\label{tab:main_results}
\\begin{tabular}{lccc}
\\hline
 & Model 1 & Model 2 & Model 3 \\\\
\\hline
"""

    for _, row in df.iterrows():
        latex += f"{row['Variable']} & {row['Model 1']} & {row['Model 2']} & {row['Model 3']} \\\\\n"

    latex += """\\hline
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Notes: Robust standard errors in parentheses. * p<0.1, ** p<0.05, *** p<0.01
\\end{tablenotes}
\\end{table}
"""

    with open(f'{TABLE_PATH}/table2.tex', 'w') as f:
        f.write(latex)

    print("Generated table2.tex")

def make_all_tables():
    table1_to_latex()
    combine_regression_results()
    print("All tables generated!")

if __name__ == '__main__':
    make_all_tables()
