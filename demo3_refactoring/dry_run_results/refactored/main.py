#!/usr/bin/env python
"""
Main Entry Point for Thesis Analysis Pipeline.

This script orchestrates the complete thesis analysis workflow from
data processing through final outputs. It uses proper module imports
instead of exec() calls.

Usage:
    # Run full pipeline
    python main.py

    # Run specific stages
    python main.py --stage process
    python main.py --stage analysis
    python main.py --stage figures

    # Check configuration
    python main.py --check

Author: Refactored from original messy_codebase
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Import configuration and data loading
from config import Config
from data_loader import DataLoader


def check_prerequisites() -> bool:
    """
    Check that all prerequisites are met before running the pipeline.

    Returns:
        True if all checks pass, False otherwise
    """
    print("Checking prerequisites...")

    # Check raw survey data exists (required input)
    if not Config.SURVEY_RAW_FILE.exists():
        print(f"\nERROR: Raw survey data not found!")
        print(f"Expected at: {Config.SURVEY_RAW_FILE}")
        print("\nPlease run the download scripts first to obtain survey data.")
        return False

    print("  [OK] Raw survey data found")

    # Check directories exist (create if needed)
    Config.ensure_directories_exist()
    print("  [OK] Output directories ready")

    return True


def run_data_processing():
    """
    Stage 1: Process raw survey data into cleaned format.

    This corresponds to the original process_survey.py functionality.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: Processing Survey Data")
    print("=" * 70)

    # Import processing module
    try:
        from process_survey import process_all as process_survey
        process_survey()
        print("[OK] Survey data processed successfully")
    except ImportError:
        print("[SKIP] process_survey module not found")
    except Exception as e:
        print(f"[ERROR] Failed to process survey: {e}")
        raise


def run_data_merge():
    """
    Stage 2: Merge all data sources into analysis dataset.

    This corresponds to the original merge_all_data.py functionality.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: Merging Data Sources")
    print("=" * 70)

    try:
        from merge_all_data import create_final_dataset
        create_final_dataset()
        print("[OK] Data merged successfully")
    except ImportError:
        print("[SKIP] merge_all_data module not found")
    except Exception as e:
        print(f"[ERROR] Failed to merge data: {e}")
        raise


def run_chapter3_analysis():
    """
    Stage 3a: Run Chapter 3 analysis (individual-level determinants).
    """
    print("\n" + "=" * 70)
    print("STAGE 3a: Chapter 3 Analysis")
    print("=" * 70)

    try:
        from analysis_chapter3 import run_all_chapter3
        run_all_chapter3()
        print("[OK] Chapter 3 analysis complete")
    except ImportError:
        print("[SKIP] analysis_chapter3 module not found")
    except Exception as e:
        print(f"[ERROR] Chapter 3 analysis failed: {e}")
        raise


def run_chapter4_analysis():
    """
    Stage 3b: Run Chapter 4 analysis (contextual effects).
    """
    print("\n" + "=" * 70)
    print("STAGE 3b: Chapter 4 Analysis")
    print("=" * 70)

    try:
        from analysis_chapter4 import run_all_chapter4
        run_all_chapter4()
        print("[OK] Chapter 4 analysis complete")
    except ImportError:
        print("[SKIP] analysis_chapter4 module not found")
    except Exception as e:
        print(f"[ERROR] Chapter 4 analysis failed: {e}")
        raise


def run_robustness_checks():
    """
    Stage 4: Run robustness checks.
    """
    print("\n" + "=" * 70)
    print("STAGE 4: Robustness Checks")
    print("=" * 70)

    try:
        from analysis_robustness import run_all_robustness
        run_all_robustness()
        print("[OK] Robustness checks complete")
    except ImportError:
        print("[SKIP] analysis_robustness module not found")
    except Exception as e:
        print(f"[ERROR] Robustness checks failed: {e}")
        raise


def run_figure_generation():
    """
    Stage 5: Generate all figures.
    """
    print("\n" + "=" * 70)
    print("STAGE 5: Generating Figures")
    print("=" * 70)

    try:
        from make_figures_chapter3 import main as make_ch3_figures
        make_ch3_figures()
        print("  [OK] Chapter 3 figures")
    except ImportError:
        print("  [SKIP] make_figures_chapter3 module not found")
    except Exception as e:
        print(f"  [ERROR] Chapter 3 figures failed: {e}")

    try:
        from make_figures_chapter4 import main as make_ch4_figures
        make_ch4_figures()
        print("  [OK] Chapter 4 figures")
    except ImportError:
        print("  [SKIP] make_figures_chapter4 module not found")
    except Exception as e:
        print(f"  [ERROR] Chapter 4 figures failed: {e}")


def run_table_generation():
    """
    Stage 6: Generate all tables.
    """
    print("\n" + "=" * 70)
    print("STAGE 6: Generating Tables")
    print("=" * 70)

    try:
        from make_tables import main as make_tables
        make_tables()
        print("[OK] Tables generated")
    except ImportError:
        print("[SKIP] make_tables module not found")
    except Exception as e:
        print(f"[ERROR] Table generation failed: {e}")


def run_full_pipeline():
    """
    Run the complete analysis pipeline.
    """
    print("=" * 70)
    print("THESIS ANALYSIS PIPELINE")
    print("=" * 70)

    if not check_prerequisites():
        sys.exit(1)

    # Run all stages in order
    stages = [
        ("process", run_data_processing),
        ("merge", run_data_merge),
        ("chapter3", run_chapter3_analysis),
        ("chapter4", run_chapter4_analysis),
        ("robustness", run_robustness_checks),
        ("figures", run_figure_generation),
        ("tables", run_table_generation),
    ]

    for stage_name, stage_func in stages:
        try:
            stage_func()
        except Exception as e:
            print(f"\n[FATAL] Pipeline failed at stage '{stage_name}': {e}")
            sys.exit(1)

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nOutputs generated:")
    print(f"  Regression results: {Config.OUTPUT_DIR}/*.txt")
    print(f"  Figures:            {Config.FIGURES_DIR}/*.png")
    print(f"  Figures (PDF):      {Config.FIGURES_DIR}/*.pdf")
    print(f"  LaTeX tables:       {Config.TABLES_DIR}/*.tex")


def run_stage(stage: str):
    """
    Run a specific stage of the pipeline.

    Args:
        stage: Name of the stage to run
    """
    stage_map = {
        "process": run_data_processing,
        "merge": run_data_merge,
        "chapter3": run_chapter3_analysis,
        "chapter4": run_chapter4_analysis,
        "robustness": run_robustness_checks,
        "figures": run_figure_generation,
        "tables": run_table_generation,
        "analysis": lambda: (
            run_chapter3_analysis(),
            run_chapter4_analysis(),
            run_robustness_checks(),
        ),
    }

    if stage not in stage_map:
        print(f"Unknown stage: {stage}")
        print(f"Available stages: {', '.join(stage_map.keys())}")
        sys.exit(1)

    print(f"Running stage: {stage}")
    stage_map[stage]()


def check_config():
    """
    Print configuration and data status.
    """
    Config.print_config()
    print("\n")
    DataLoader.check_data_exists(verbose=True)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Thesis Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Run full pipeline
  python main.py --stage process  # Process survey data only
  python main.py --stage analysis # Run all analysis
  python main.py --check          # Check configuration

Stages:
  process    - Process raw survey data
  merge      - Merge all data sources
  chapter3   - Run Chapter 3 analysis
  chapter4   - Run Chapter 4 analysis
  robustness - Run robustness checks
  figures    - Generate figures
  tables     - Generate tables
  analysis   - Run all analysis stages (chapter3 + chapter4 + robustness)
        """,
    )

    parser.add_argument(
        "--stage",
        "-s",
        type=str,
        help="Run a specific stage only",
    )

    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        help="Check configuration and data status",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def main():
    """
    Main entry point.
    """
    args = parse_args()

    if args.check:
        check_config()
    elif args.stage:
        run_stage(args.stage)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
