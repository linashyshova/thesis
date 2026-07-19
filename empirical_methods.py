"""
Empirical-application wrappers for variance-reduction methods.

All functions accept numpy arrays (not DataFrames) and return a dict with:
    effect, se, ci_lower, ci_upper, ci_width, pvalue, significant
CUPED-based methods additionally return:
    theta, variance_reduction

Formulas are identical to methods.py so that verify_against_methods_py()
produces exact numeric matches.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind

ALPHA = 0.05


# ---------------------------------------------------------------------------
# 1. Welch t-test
# ---------------------------------------------------------------------------

def welch(control: np.ndarray, treatment: np.ndarray, alpha: float = ALPHA) -> dict:
    """Welch two-sample t-test (treatment − control)."""
    result = ttest_ind(treatment, control, equal_var=False,
                       nan_policy="omit", alternative="two-sided")
    ci = result.confidence_interval(confidence_level=1 - alpha)
    ci_width = float(ci.high - ci.low)
    effect = float(np.nanmean(treatment) - np.nanmean(control))
    se = ci_width / (2 * norm.ppf(1 - alpha / 2))
    return dict(
        effect=effect,
        se=se,
        ci_lower=float(ci.low),
        ci_upper=float(ci.high),
        ci_width=ci_width,
        pvalue=float(result.pvalue),
        significant=bool(result.pvalue < alpha),
    )


# ---------------------------------------------------------------------------
# 2. CUPED
# ---------------------------------------------------------------------------

def cuped_method(control: np.ndarray, treatment: np.ndarray,
                 control_pre: np.ndarray, treatment_pre: np.ndarray,
                 alpha: float = ALPHA) -> dict:
    """CUPED adjustment; theta estimated from control arrays only."""
    cov = np.cov(control, control_pre)[0, 1]
    var_X = np.var(control_pre, ddof=1)
    mean_X = np.mean(control_pre)
    theta = float(cov / var_X)

    control_adj = control - theta * (control_pre - mean_X)
    treatment_adj = treatment - theta * (treatment_pre - mean_X)

    var_before = float(np.var(control, ddof=1))
    var_after = float(np.var(control_adj, ddof=1))
    variance_reduction = float(1.0 - var_after / var_before) if var_before > 0 else 0.0

    result = ttest_ind(treatment_adj, control_adj, equal_var=False,
                       alternative="two-sided")
    ci = result.confidence_interval(confidence_level=1 - alpha)
    ci_width = float(ci.high - ci.low)
    effect = float(np.mean(treatment_adj) - np.mean(control_adj))
    se = ci_width / (2 * norm.ppf(1 - alpha / 2))

    return dict(
        effect=effect,
        se=se,
        ci_lower=float(ci.low),
        ci_upper=float(ci.high),
        ci_width=ci_width,
        pvalue=float(result.pvalue),
        significant=bool(result.pvalue < alpha),
        theta=theta,
        variance_reduction=variance_reduction,
    )


# ---------------------------------------------------------------------------
# 3. Stratification
# ---------------------------------------------------------------------------

def _stratification_core(control: np.ndarray, treatment: np.ndarray,
                          control_strata: np.ndarray, treatment_strata: np.ndarray,
                          alpha: float = ALPHA) -> dict:
    """
    Shared stratification logic.
    control / treatment      : outcome arrays
    control_strata / treatment_strata : stratum labels (same length)
    """
    # Build per-stratum stats using pandas for convenience
    c_df = pd.DataFrame({'y': control, 's': control_strata})
    t_df = pd.DataFrame({'y': treatment, 's': treatment_strata})

    c_grp = c_df.groupby('s')['y'].agg(['mean', 'var', 'count'])
    t_grp = t_df.groupby('s')['y'].agg(['mean', 'var', 'count'])

    all_idx = c_grp.index.union(t_grp.index)
    c_grp = c_grp.reindex(all_idx, fill_value=0.0)
    t_grp = t_grp.reindex(all_idx, fill_value=0.0)

    mean0 = c_grp['mean'].to_numpy()
    var0  = c_grp['var'].to_numpy()
    n0    = c_grp['count'].to_numpy(dtype=float)

    mean1 = t_grp['mean'].to_numpy()
    var1  = t_grp['var'].to_numpy()
    n1    = t_grp['count'].to_numpy(dtype=float)

    nh    = n0 + n1
    N     = nh.sum()
    omega = nh / N

    diff_h  = mean1 - mean0
    tau     = float((omega * diff_h).sum())

    var_h   = var1 / n1 + var0 / n0   # var_h is 0 where n=0 (fill_value=0)
    var_tau = float(((omega ** 2) * var_h).sum())

    se = float(np.sqrt(var_tau))
    z  = float(norm.ppf(1 - alpha / 2))

    ci_width = float(2 * z * se)
    ci_lower = float(tau - z * se)
    ci_upper = float(tau + z * se)
    pvalue   = float(2 * (1 - norm.cdf(abs(tau / se))))

    return dict(
        effect=tau,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_width,
        pvalue=pvalue,
        significant=bool(pvalue < alpha),
    )


def stratification_method(control: np.ndarray, treatment: np.ndarray,
                          control_strata: np.ndarray, treatment_strata: np.ndarray,
                          alpha: float = ALPHA) -> dict:
    """Post-stratification estimator."""
    return _stratification_core(control, treatment,
                                control_strata, treatment_strata, alpha)


# ---------------------------------------------------------------------------
# 4. Winsorization
# ---------------------------------------------------------------------------

def winsorize_method(control: np.ndarray, treatment: np.ndarray,
                     lower_bound: float, upper_bound: float,
                     alpha: float = ALPHA) -> dict:
    """Winsorized Welch t-test."""
    control_w   = np.clip(control.astype(float),   lower_bound, upper_bound)
    treatment_w = np.clip(treatment.astype(float),  lower_bound, upper_bound)
    return welch(control_w, treatment_w, alpha=alpha)


# ---------------------------------------------------------------------------
# 5. Winsorization + CUPED
# ---------------------------------------------------------------------------

def winsorized_cuped_method(control: np.ndarray, treatment: np.ndarray,
                             control_pre: np.ndarray, treatment_pre: np.ndarray,
                             lower_bound: float, upper_bound: float,
                             alpha: float = ALPHA) -> dict:
    """
    Winsorize Y first, then CUPED.
    theta estimated from winsorized control Y and raw control X (matches methods.py).
    """
    control_w   = np.clip(control.astype(float),   lower_bound, upper_bound)
    treatment_w = np.clip(treatment.astype(float),  lower_bound, upper_bound)

    cov   = np.cov(control_w, control_pre.astype(float))[0, 1]
    var_X = np.var(control_pre.astype(float), ddof=1)
    mean_X = np.mean(control_pre.astype(float))
    theta = float(cov / var_X)

    control_adj   = control_w   - theta * (control_pre.astype(float)   - mean_X)
    treatment_adj = treatment_w - theta * (treatment_pre.astype(float) - mean_X)

    var_before = float(np.var(control_w, ddof=1))
    var_after  = float(np.var(control_adj, ddof=1))
    variance_reduction = float(1.0 - var_after / var_before) if var_before > 0 else 0.0

    result = ttest_ind(treatment_adj, control_adj, equal_var=False,
                       alternative="two-sided")
    ci = result.confidence_interval(confidence_level=1 - alpha)
    ci_width = float(ci.high - ci.low)
    effect = float(np.mean(treatment_adj) - np.mean(control_adj))
    se = ci_width / (2 * norm.ppf(1 - alpha / 2))

    return dict(
        effect=effect,
        se=se,
        ci_lower=float(ci.low),
        ci_upper=float(ci.high),
        ci_width=ci_width,
        pvalue=float(result.pvalue),
        significant=bool(result.pvalue < alpha),
        theta=theta,
        variance_reduction=variance_reduction,
    )


# ---------------------------------------------------------------------------
# 6. Winsorization + Stratification
# ---------------------------------------------------------------------------

def winsorized_stratification_method(control: np.ndarray, treatment: np.ndarray,
                                      control_strata: np.ndarray,
                                      treatment_strata: np.ndarray,
                                      lower_bound: float, upper_bound: float,
                                      alpha: float = ALPHA) -> dict:
    """Winsorize Y first, then stratify."""
    control_w   = np.clip(control.astype(float),  lower_bound, upper_bound)
    treatment_w = np.clip(treatment.astype(float), lower_bound, upper_bound)
    return _stratification_core(control_w, treatment_w,
                                control_strata, treatment_strata, alpha)


# ---------------------------------------------------------------------------
# 7. Stratification + CUPED
# ---------------------------------------------------------------------------

def stratified_cuped_method(control: np.ndarray, treatment: np.ndarray,
                             control_pre: np.ndarray, treatment_pre: np.ndarray,
                             control_strata: np.ndarray, treatment_strata: np.ndarray,
                             alpha: float = ALPHA) -> dict:
    """CUPED adjustment followed by post-stratification."""
    cov   = np.cov(control.astype(float), control_pre.astype(float))[0, 1]
    var_X = np.var(control_pre.astype(float), ddof=1)
    mean_X = np.mean(control_pre.astype(float))
    theta = float(cov / var_X)

    var_before = float(np.var(control.astype(float), ddof=1))

    control_adj   = control.astype(float)   - theta * (control_pre.astype(float)   - mean_X)
    treatment_adj = treatment.astype(float) - theta * (treatment_pre.astype(float) - mean_X)

    var_after = float(np.var(control_adj, ddof=1))
    variance_reduction = float(1.0 - var_after / var_before) if var_before > 0 else 0.0

    result = _stratification_core(control_adj, treatment_adj,
                                  control_strata, treatment_strata, alpha)
    result['theta'] = theta
    result['variance_reduction'] = variance_reduction
    return result


# ---------------------------------------------------------------------------
# 8. Winsorization + Stratification + CUPED
# ---------------------------------------------------------------------------

def winsorized_stratified_cuped_method(control: np.ndarray, treatment: np.ndarray,
                                        control_pre: np.ndarray, treatment_pre: np.ndarray,
                                        control_strata: np.ndarray,
                                        treatment_strata: np.ndarray,
                                        lower_bound: float, upper_bound: float,
                                        alpha: float = ALPHA) -> dict:
    """Winsorize Y, then CUPED (theta from winsorized control Y), then stratify."""
    control_w   = np.clip(control.astype(float),  lower_bound, upper_bound)
    treatment_w = np.clip(treatment.astype(float), lower_bound, upper_bound)

    cov   = np.cov(control_w, control_pre.astype(float))[0, 1]
    var_X = np.var(control_pre.astype(float), ddof=1)
    mean_X = np.mean(control_pre.astype(float))
    theta = float(cov / var_X)

    var_before = float(np.var(control_w, ddof=1))

    control_adj   = control_w   - theta * (control_pre.astype(float)   - mean_X)
    treatment_adj = treatment_w - theta * (treatment_pre.astype(float) - mean_X)

    var_after = float(np.var(control_adj, ddof=1))
    variance_reduction = float(1.0 - var_after / var_before) if var_before > 0 else 0.0

    result = _stratification_core(control_adj, treatment_adj,
                                  control_strata, treatment_strata, alpha)
    result['theta'] = theta
    result['variance_reduction'] = variance_reduction
    return result


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def verify_against_methods_py(df: pd.DataFrame,
                               lower_bound: float,
                               upper_bound: float,
                               split_col: str = 'is_treatment',
                               feature: str = 'spend',
                               pre_feature: str = 'history',
                               strat_feature: str = 'channel') -> dict:
    """
    Compare methods.py functions with empirical_methods.py functions.

    Returns {method_name: {'ci_width_match': bool, 'significant_match': bool}}.
    """
    import sys, os
    # Allow importing methods.py (which imports simulation) gracefully
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import methods as m
    except ImportError as e:
        return {'error': str(e)}

    control   = df.loc[~df[split_col], feature].values.astype(float)
    treatment = df.loc[df[split_col],  feature].values.astype(float)
    control_pre   = df.loc[~df[split_col], pre_feature].values.astype(float)
    treatment_pre = df.loc[df[split_col],  pre_feature].values.astype(float)
    control_strata   = df.loc[~df[split_col], strat_feature].values
    treatment_strata = df.loc[df[split_col],  strat_feature].values

    tol = 1e-8

    def _check(mpy_ci_width, mpy_reject, emp_result):
        return {
            'ci_width_match':   abs(mpy_ci_width - emp_result['ci_width']) < tol,
            'significant_match': int(emp_result['significant']) == mpy_reject,
            'ci_width_methods_py':  mpy_ci_width,
            'ci_width_empirical': emp_result['ci_width'],
        }

    results = {}

    # 1. Welch
    mpy = m.ttest(df, split_col, feature)
    emp = welch(control, treatment)
    results['Welch t-test'] = _check(mpy[0], mpy[1], emp)

    # 2. CUPED
    mpy = m.cuped(df, split_col, pre_feature, feature)
    emp = cuped_method(control, treatment, control_pre, treatment_pre)
    results['CUPED'] = _check(mpy[0], mpy[1], emp)

    # 3. Stratification
    mpy = m.stratification(df, split_col, feature, strat_feature)
    emp = stratification_method(control, treatment, control_strata, treatment_strata)
    results['Stratification'] = _check(mpy[0], mpy[1], emp)

    # 4. Winsorization
    mpy = m.winsorize(df, split_col, feature, lower_bound, upper_bound)
    emp = winsorize_method(control, treatment, lower_bound, upper_bound)
    results['Winsorization'] = _check(mpy[0], mpy[1], emp)

    # 5. Winsorization + CUPED
    mpy = m.winsorized_cuped(df, split_col, feature, pre_feature, lower_bound, upper_bound)
    emp = winsorized_cuped_method(control, treatment, control_pre, treatment_pre,
                                  lower_bound, upper_bound)
    results['Winsorization + CUPED'] = _check(mpy[0], mpy[1], emp)

    # 6. Winsorization + Stratification
    mpy = m.winsorized_stratification(df, split_col, feature, strat_feature,
                                      lower_bound, upper_bound)
    emp = winsorized_stratification_method(control, treatment,
                                           control_strata, treatment_strata,
                                           lower_bound, upper_bound)
    results['Winsorization + Stratification'] = _check(mpy[0], mpy[1], emp)

    # 7. Stratification + CUPED
    mpy = m.stratified_cuped(df, split_col, feature, pre_feature, strat_feature)
    emp = stratified_cuped_method(control, treatment, control_pre, treatment_pre,
                                  control_strata, treatment_strata)
    results['Stratification + CUPED'] = _check(mpy[0], mpy[1], emp)

    # 8. Winsorization + Stratification + CUPED
    mpy = m.winsorized_stratified_cuped(df, split_col, feature, pre_feature,
                                        strat_feature, lower_bound, upper_bound)
    emp = winsorized_stratified_cuped_method(control, treatment,
                                             control_pre, treatment_pre,
                                             control_strata, treatment_strata,
                                             lower_bound, upper_bound)
    results['Winsorization + Stratification + CUPED'] = _check(mpy[0], mpy[1], emp)

    return results
