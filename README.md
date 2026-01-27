# D-Lab Claude Code Workshop

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the materials for D-Lab's Claude Code AI Pulse workshop.

## Prerequisites

No prior experience with CLI tools required!

Check out D-Lab's [Workshop Catalog](https://dlab-berkeley.github.io/dlab-workshops/) to browse all workshops.

## Workshop Goals

This hands-on workshop introduces Claude Code, Anthropic's command-line interface for AI-assisted coding. You will learn how CLI-based AI tools differ from browser chat interfaces, how to configure Claude Code for your projects, and see practical demonstrations of code generation, documentation, and refactoring. Participants will leave with practical skills for integrating AI coding assistants into their research workflows.

## Learning Objectives

After completing this workshop, you will be able to:

1. **Understand CLI vs Browser AI tools** - Recognize when to use command-line tools (multi-file projects, direct file manipulation) versus web interfaces (quick questions, conceptual explanations).

2. **Install and configure Claude Code** - Set up the CLI tool and customize it for your projects using CLAUDE.md files and configuration options.

3. **Navigate models and context** - Understand different model options (Opus, Sonnet, Haiku) and manage conversation context effectively.

4. **Generate code from specifications** - Use Claude Code to create new code modules with proper structure and documentation.

5. **Document existing codebases** - Leverage Claude Code to read and understand unfamiliar code, generating documentation and explanations.

6. **Refactor and improve code** - Use AI assistance to clean up messy code, improve structure, and add error handling.

7. **Use subagents for verification** - Spawn subagents to independently verify results and catch errors through parallel review.

## Workshop Structure (~30 minutes)

1. **Introduction to CLI** (5 min) - What is a CLI, browser vs CLI differences
2. **Installation & Setup** (2 min) - Quick installation, first run
3. **Customization** (5 min) - CLAUDE.md, models, context, subagents
4. **Demo 1: Linear Regression** (5 min) - Building code from scratch
5. **Demo 2: Julia Documentation** (5 min) - Reading and documenting existing code
6. **Demo 3: Refactoring** (5 min) - Cleaning up a messy PhD codebase
7. **Demo 4: Data Consolidation** (bonus) - Merging economic data with subagent verification
8. **Q&A** (remaining time)

## Folder Structure

```
Claude Code Workshop/
├── README.md
├── slides/
│   ├── workshop_slides.tex      # LaTeX Beamer source
│   ├── workshop_slides.pdf      # Compiled slides (30 pages)
│   └── workshop_slides.md       # Original markdown slides
├── demo1_linear_regression/
│   └── productivity_study.csv   # Sample dataset
├── demo2_julia_documentation/
│   ├── SOURCE.md                # Instructions to download DSGE.jl
│   └── dry_run_results/         # Example output from demo
├── demo3_refactoring/
│   ├── messy_codebase/          # Realistic PhD project (19 Python files)
│   │   ├── download_*.py        # 4 data downloaders
│   │   ├── process_survey.py
│   │   ├── merge_all_data.py
│   │   ├── analysis_*.py        # Chapter-specific analysis
│   │   ├── make_figures_*.py    # Figure generation
│   │   ├── make_tables.py
│   │   ├── helpers.py
│   │   ├── run_all.py           # Uses exec() anti-pattern
│   │   ├── old_scripts/         # "Just in case" folder
│   │   └── for_advisor/         # Quick meeting scripts
│   └── dry_run_results/         # Example output from demo
├── demo4_data_consolidation/
│   └── SOURCE.md                # Instructions to download FRED/Michigan/SPF data
└── assets/
    └── example_CLAUDE.md
```

## Data Files

Some demo data files are too large for GitHub and must be downloaded separately:

- **Demo 2 (DSGE.jl)**: Download from [FRBNY-DSGE/DSGE.jl](https://github.com/FRBNY-DSGE/DSGE.jl)
- **Demo 4 (Economic Data)**: See `demo4_data_consolidation/SOURCE.md` for download links to FRED, Michigan Survey, and SPF data

## Installation

**Prerequisites:**
- Node.js (download from [nodejs.org](https://nodejs.org/))
- Windows users: WSL (run `wsl --install` in PowerShell)

**Install Claude Code:**
```bash
npm install -g @anthropic-ai/claude-code
claude
```

## Resources

- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [Claude.ai](https://claude.ai/) (Pro/Team subscription includes Claude Code)
- [Anthropic Console](https://console.anthropic.com/) (API keys)

---

# About the UC Berkeley D-Lab

D-Lab works with Berkeley faculty, research staff, and students to advance data-intensive social science and humanities research. Our goal at D-Lab is to provide practical training, staff support, resources, and space to enable you to use AI tools for your own research applications.

Visit the [D-Lab homepage](https://dlab.berkeley.edu/) to learn more about us.

# Contributors

* Bruno Cittolin Smaniotto
* Tom van Nuenen
* Claude (Anthropic)
