# Power Enhancement Methods for Composite Revenue Metrics in A/B Tests

This repository contains the simulation framework and analysis code for a thesis comparing variance-reduction methods for detecting treatment effects on composite continuous revenue metrics in online controlled experiments.

## Motivation

Revenue metrics in A/B tests are often composite: they aggregate components that differ in scale, respond heterogeneously to treatment, and follow heavy-tailed distributions. Simple difference-in-means tests can be underpowered in this setting, masking true effects or slowing iteration. This study evaluates methods that recover that lost power.

## Methods Compared

| Method | Mechanism |
|---|---|
| Baseline (Welch t-test) | Difference of means on raw user-level revenue |
| CUPED | Pre-experiment covariate adjustment to reduce residual variance |
| Stratification | Country-level post-stratification to remove between-group variance |
| Winsorization | 0.3th / 99.7th percentile clipping to tame heavy tails |
| CUPED + Winsorization | CUPED first, then Winsorize the adjusted residuals |
| CUPED + Stratification | CUPED first, then stratified estimator on adjusted metric |
| Winsorization + Stratification | Winsorize first, then stratified estimator |
| CUPED + Winsorization + Stratification | Full pipeline: adjust → clip → stratify |

## Evaluation

All methods are applied to the same simulated datasets and compared on three metrics:

- **Statistical power** — true positive rate across 1,000 A/B replications with an injected effect
- **CI width** — average 95% confidence interval width (precision)
- **False positive rate** — empirical Type I error from A/A tests (no injected effect), must stay near α = 0.05

## Simulation Design

Synthetic data is generated with configurable:
- Zero-inflation and heavy tails
- Pre-period / experiment-period correlation (drives CUPED gains)
- Component-level effect heterogeneity and scale imbalance
- Country-level stratification structure

This controlled setup allows ground-truth power measurement and isolation of each method's contribution.

## Repository Structure

```
simulation.py   # Data-generating process and experiment runner
methods.py      # Baseline, CUPED, Stratification, Winsorization, and combined estimators
scenarios.ipynb # Scenario definitions, result aggregation, and plots
requirements.txt
```