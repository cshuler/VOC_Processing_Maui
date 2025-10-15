# WRRC VOC Processing – Maui Fire Response

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)]()
[![Reproducible-Env](https://img.shields.io/badge/env-conda-blue.svg)]()

Post-processing pipeline and curated outputs for volatile organic compound (VOC) sampling conducted by the University of Hawaiʻi Water Resources Research Center (WRRC) in the aftermath of the 2023 Maui wildfires. This repo contains processing notebooks/scripts, a harmonized public data table, health-limit keys, and summary tables used to generate community-facing reports.

> For a visual overview of the fields used in the one-page reports, see the README figure below.


> **Community Water Info Hub**  
> For project background, sampling updates, and public resources, see:  
> https://www.wrrc.hawaii.edu/maui-post-fire-community-water-info-hub/



<p align="center">
  <img width="900" height="450" src=README_VOC_report_Explanation.jpg>
</p>



---

## Table of Contents
- [What’s in this repo](#whats-in-this-repo)
- [Background](#background)
- [Data and Structure](#data-and-structure)
- [Quickstart](#quickstart)
- [Processing Workflow](#processing-workflow)
- [Interpreting Results](#interpreting-results)
- [Data Dictionary](#data-dictionary)
- [Citation](#citation)
- [Acknowledgments & Funding](#acknowledgments--funding)
- [License](#license)
- [Contact](#contact)

---

## What’s in this repo

- `Scripts/` – Jupyter/Python code to clean, harmonize, and summarize VOC measurements.
- `MASTER_Maui_VOC_Sheet_PUBLIC.csv` – Public master table used for analysis/figures.
- `VOC_Health_Limits_Key.csv` – Health benchmarks and metadata for comparison.
- `Summary_Tables/` – Precomputed summaries for figures/reports.
- `PDFs_sample_results/` – Example output PDFs of sample reports.
- `README_VOC_report_Explanation.jpg` (and `.pdf`) – Field/label explanation graphic.
- `Paper Resources/` and `Resources/` – Supporting references and assets.

> License: MIT (see [License](#license)).

---

## Background

Following the August 2023 Maui wildfires, community water quality questions included possible VOC presence from fire-related emissions and subsequent system disturbances. This repository captures the post-processing steps used by WRRC to:
1) standardize results from multiple labs/rounds,
2) compare concentrations to health-based benchmarks,
3) organize report-ready tables and figures for public communication.

---

## Data and Structure

**Master public table**  
`MASTER_Maui_VOC_Sheet_PUBLIC.csv` contains sample-level VOC results and metadata (site, sample time, analyte, units, qualifiers, detection limits, etc.).  

**Health benchmarks**  
`VOC_Health_Limits_Key.csv` maps analytes to applicable health limits and notes (e.g., EPA MCLs, RSLs, state guidance where applicable), plus the reference/source fields.
