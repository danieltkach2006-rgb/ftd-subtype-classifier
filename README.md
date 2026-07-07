# FTD Subtype Classification Tool

A clinical decision support research prototype that predicts the likely protein pathology subtype (TDP-43, Tau, or FUS) in Frontotemporal Dementia based on clinical presentation and genetic testing results.

## Overview

Frontotemporal Dementia (FTD) is a group of neurodegenerative disorders caused by the abnormal accumulation of misfolded proteins — primarily TDP-43 (~50–60% of cases), Tau (~30–40%), or FUS (~10%). This tool uses a Random Forest classifier trained on synthetic patient data calibrated to published genotype-phenotype associations to predict the most likely underlying proteinopathy given a patient's clinical and genetic profile.

## Features

- **Input:** Age of onset, primary presenting symptoms (behavioral, language, motor), family history, and genetic mutation status (TARDBP, MAPT, GRN, C9orf72, FUS)
- **Output:** Predicted FTD subtype with probability scores for each proteinopathy
- **Model accuracy:** ~95.8% on held-out test data

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/ftd-subtype-classifier.git
cd ftd-subtype-classifier
pip install -r requirements.txt
python app.py
```

Then open your browser to **http://localhost:5000**

## Disclaimer

This is a research prototype trained on synthetic data calibrated to published genotype-phenotype associations. It is not a validated clinical diagnostic tool and should not be used for medical decision-making without physician oversight and confirmatory testing.
