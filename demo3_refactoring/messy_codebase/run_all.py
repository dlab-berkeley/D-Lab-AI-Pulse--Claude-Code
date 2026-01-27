#!/usr/bin/env python
"""
Master script to reproduce all thesis results
Run this after downloading data

Order of operations:
1. Download all data (run manually - needs API keys)
2. Process survey data
3. Merge all data sources
4. Run Chapter 3 analysis
5. Run Chapter 4 analysis
6. Run robustness checks
7. Generate figures
8. Generate tables

Last updated: October 2024
"""

import os
import sys

# Set working directory
# os.chdir('/Users/old_laptop/thesis')
# os.chdir('/home/phd_student/Dropbox/thesis')
os.chdir('C:/Users/PhDStudent/Desktop/thesis_final')

print("="*70)
print("THESIS ANALYSIS PIPELINE")
print("="*70)

# Check that data exists
if not os.path.exists('data_raw/survey_responses_raw.csv'):
    print("ERROR: Raw survey data not found!")
    print("Run download_survey.py first")
    sys.exit(1)

print("\n[1/7] Processing survey data...")
exec(open('process_survey.py').read())

print("\n[2/7] Merging all data sources...")
exec(open('merge_all_data.py').read())

print("\n[3/7] Chapter 3 analysis...")
exec(open('analysis_chapter3.py').read())

print("\n[4/7] Chapter 4 analysis...")
exec(open('analysis_chapter4.py').read())

print("\n[5/7] Robustness checks...")
exec(open('analysis_robustness.py').read())

print("\n[6/7] Generating figures...")
exec(open('make_figures_chapter3.py').read())
exec(open('make_figures_chapter4.py').read())

print("\n[7/7] Generating tables...")
exec(open('make_tables.py').read())

print("\n" + "="*70)
print("PIPELINE COMPLETE")
print("="*70)
print("\nOutputs:")
print("  - outputs/*.txt    (regression results)")
print("  - figures/*.png    (figures)")
print("  - figures/*.pdf    (figures for LaTeX)")
print("  - tables/*.tex     (LaTeX tables)")

# TODO: Add the new interaction models
# TODO: Export for Stata (co-author request)
# TODO: Make paths relative so this works on any computer
