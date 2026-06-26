import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import ttest_ind


def ttest(df, split_feature, feature, alpha=0.05):
    control = df.loc[df[split_feature] == False, feature]
    treatment = df.loc[df[split_feature] == True, feature]

    result = ttest_ind(
        control,
        treatment,
        equal_var=False,
        nan_policy="omit",
        alternative="two-sided",
    )

    ci_base = result.confidence_interval(confidence_level=1 - alpha)
    ci_width = ci_base.high - ci_base.low
    reject = int(result.pvalue < alpha)

    return ci_width, reject


def cuped(df, split_feature, pre_feature, feature, alpha=0.05):
    control = df.loc[df[split_feature] == False, feature].values
    treatment = df.loc[df[split_feature] == True, feature].values
    control_pre = df.loc[df[split_feature] == False, pre_feature].values
    treatment_pre = df.loc[df[split_feature] == True, pre_feature].values

    cov = np.cov(control, control_pre)[0, 1]
    var_X = np.var(control_pre, ddof=1)
    mean_X = np.mean(control_pre)
    theta = cov / var_X

    control_adj = control - theta * (control_pre - mean_X)
    treatment_adj = treatment - theta * (treatment_pre - mean_X)

    result = ttest_ind(control_adj, treatment_adj, equal_var=False, alternative="two-sided")

    ci = result.confidence_interval(confidence_level=1 - alpha)
    ci_width = ci.high - ci.low
    reject = int(result.pvalue < alpha)

    return ci_width, reject


def stratification(df, split_feature, feature, strat_feature, alpha=0.05):
    control = df[df[split_feature] == False]
    treatment = df[df[split_feature] == True]

    c = control.groupby([strat_feature])[feature].agg(['mean', 'var', 'count'])
    t = treatment.groupby([strat_feature])[feature].agg(['mean', 'var', 'count'])

    all_idx = c.index.union(t.index)
    c = c.reindex(all_idx, fill_value=0.0)
    t = t.reindex(all_idx, fill_value=0.0)

    mean0, variance0, n0 = c['mean'].to_numpy(), c['var'].to_numpy(), c['count'].to_numpy(dtype=float)
    mean1, variance1, n1 = t['mean'].to_numpy(), t['var'].to_numpy(), t['count'].to_numpy(dtype=float)

    nh = n0 + n1
    N = nh.sum()
    omega = nh / N

    diff_h = mean1 - mean0
    tau = float((omega * diff_h).sum())

    var_h = variance1 / n1 + variance0 / n0
    var_tau = float(((omega ** 2) * var_h).sum())

    se = np.sqrt(var_tau)

    z = norm.ppf(1 - alpha / 2)
    ci_width = float(2 * z * se)
    p_value = float(2 * (1 - norm.cdf(abs(tau / se))))

    reject_strat = int(p_value < alpha)

    return ci_width, reject_strat


def winsorize(df, split_feature, feature, alpha=0.05, trim_frac=0.01):
    control = df.loc[df[split_feature] == False, feature].values.astype(float)
    treatment = df.loc[df[split_feature] == True, feature].values.astype(float)

    lower_bound = np.quantile(control, trim_frac)
    upper_bound = np.quantile(control, 1 - trim_frac)

    control_w = np.clip(control, lower_bound, upper_bound)
    treatment_w = np.clip(treatment, lower_bound, upper_bound)

    result = ttest_ind(control_w, treatment_w, equal_var=False, alternative="two-sided")

    ci = result.confidence_interval(confidence_level=1 - alpha)
    ci_width = ci.high - ci.low
    reject = int(result.pvalue < alpha)

    return ci_width, reject


def winsorized_cuped(df, split_feature, feature, pre_feature, alpha=0.05, trim_frac=0.01):
    control = df.loc[df[split_feature] == False, feature].values.astype(float)
    treatment = df.loc[df[split_feature] == True, feature].values.astype(float)
    control_pre = df.loc[df[split_feature] == False, pre_feature].values.astype(float)
    treatment_pre = df.loc[df[split_feature] == True, pre_feature].values.astype(float)

    lower = np.quantile(control, trim_frac)
    upper = np.quantile(control, 1 - trim_frac)

    control_w = np.clip(control, lower, upper)
    treatment_w = np.clip(treatment, lower, upper)

    cov = np.cov(control_w, control_pre)[0, 1]
    var_X = np.var(control_pre, ddof=1)
    mean_X = np.mean(control_pre)
    theta = cov / var_X

    control_adj = control_w - theta * (control_pre - mean_X)
    treatment_adj = treatment_w - theta * (treatment_pre - mean_X)

    result = ttest_ind(control_adj, treatment_adj, equal_var=False, alternative="two-sided")

    ci = result.confidence_interval(confidence_level=1 - alpha)
    ci_width = ci.high - ci.low
    reject = int(result.pvalue < alpha)

    return ci_width, reject


def winsorized_stratification(df, split_feature, feature, strat_feature, alpha=0.05, trim_frac=0.01):
    control_vals = df.loc[df[split_feature] == False, feature].values.astype(float)

    lower = np.quantile(control_vals, trim_frac)
    upper = np.quantile(control_vals, 1 - trim_frac)

    df_w = df.copy()
    df_w[feature] = df_w[feature].astype(float).clip(lower, upper)

    return stratification(df_w, split_feature, feature, strat_feature, alpha)


def stratified_cuped(df, split_feature, feature, pre_feature, strat_feature, alpha=0.05):
    control_pre = df.loc[df[split_feature] == False, pre_feature].values
    mean_X = np.mean(control_pre)
    cov = np.cov(
        df.loc[df[split_feature] == False, feature].values.astype(float),
        control_pre
    )[0, 1]
    var_X = np.var(control_pre, ddof=1)
    theta = cov / var_X

    df_adj = df.copy()
    df_adj[feature] = df_adj[feature].astype(float) - theta * (df_adj[pre_feature] - mean_X)

    return stratification(df_adj, split_feature, feature, strat_feature, alpha)


def winsorized_stratified_cuped(df, split_feature, feature, pre_feature, strat_feature, alpha=0.05, trim_frac=0.01):
    control_vals = df.loc[df[split_feature] == False, feature].values.astype(float)

    lower = np.quantile(control_vals, trim_frac)
    upper = np.quantile(control_vals, 1 - trim_frac)

    df_w = df.copy()
    df_w[feature] = df_w[feature].astype(float).clip(lower, upper)

    control_w = df_w.loc[df_w[split_feature] == False, feature].values
    control_pre = df_w.loc[df_w[split_feature] == False, pre_feature].values
    cov = np.cov(control_w, control_pre)[0, 1]
    var_X = np.var(control_pre, ddof=1)
    mean_X = np.mean(control_pre)
    theta = cov / var_X

    df_w[feature] = df_w[feature] - theta * (df_w[pre_feature] - mean_X)

    return stratification(df_w, split_feature, feature, strat_feature, alpha)
