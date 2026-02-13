# Aircraft Scheduling Optimization: Decision Focused Learning

This repository contains a minimal, runnable implementation of an end-to-end **predict-then-optimize** workflow for an Aircraft Arrival Scheduling Problem (ASP) using **PyEPO** + **Gurobi**, with:
- Gradient-free decision-focused methods: `blackboxOpt (dbb)` and `negativeIdentity (nid)`
- A simple PSO-based hyperparameter / parameter search baseline

## Quickstart (Google Colab)
1. Open `notebooks/colab_runner.ipynb`
2. Update `CSV_PATH` to your Drive path
3. Run all cells

> Note: `gurobipy` requires a valid Gurobi license.

## Credit
This code is adapted from the official repository:
- https://github.com/harrylui1995/ASP_E2EPO

Related paper:
Lui, G. N., & Demirel, S. (2025). *Gradient-based smart predict-then-optimize framework for aircraft arrival scheduling problem*. Journal of Open Aviation Science, 2(2).

## Local setup
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/asp-predict-then-optimize.git
cd asp-predict-then-optimize
pip install -r requirements.txt


