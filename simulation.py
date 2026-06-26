import numpy as np
import pandas as pd
from scipy.special import expit, logit as logit_fn
from scipy.stats import beta

DISCOUNT = 0.10
COUNTRIES = [
    "US",
    "GB",
    "JP",
    "KR",
    "DE",
    "IT",
    "HU",
    "GR",
    "NL",
    "UA",
]

PRICE_MU_LOG = 6.4
PRICE_SIGMA_LOG = 0.8
PRICE_MIN = 0
PRICE_MAX = 5000

COUNTRY_PARAMS = {
    "US": {"base": -1.7, "summer": 0.25, "discount": 0.35, "precision": 20},
    "GB": {"base": -1.9, "summer": 0.20, "discount": 0.30, "precision": 22},
    "JP": {"base": -1.4, "summer": 0.10, "discount": 0.40, "precision": 18},
    "KR": {"base": -1.5, "summer": 0.15, "discount": 0.38, "precision": 19},
    "DE": {"base": -2.2, "summer": 0.40, "discount": 0.25, "precision": 25},
    "IT": {"base": -1.9, "summer": 0.35, "discount": 0.28, "precision": 23},
    "HU": {"base": -2.5, "summer": 0.20, "discount": 0.32, "precision": 20},
    "GR": {"base": -2.2, "summer": 0.45, "discount": 0.27, "precision": 24},
    "NL": {"base": -1.7, "summer": 0.15, "discount": 0.30, "precision": 21},
    "UA": {"base": -2.5, "summer": 0.18, "discount": 0.33, "precision": 20},
}

SERVICE_FEE = {
    "US": 0.005,
    "GB": 0.015,
    "JP": 0.010,
    "KR": 0.010,
    "DE": 0.005,
    "IT": 0.005,
    "HU": 0.006,
    "GR": 0.005,
    "NL": 0.007,
    "UA": 0.006,
}


def get_conversion_distribution(country, is_discount, is_summer, summer_scale, discount_scale, precision_scale):
    p = COUNTRY_PARAMS[country]

    logit_mu = p["base"]
    if is_summer:
        logit_mu += p["summer"] * summer_scale
    if is_discount:
        logit_mu += p["discount"] * discount_scale

    mu = expit(logit_mu)
    nu = p["precision"] * precision_scale

    alpha = mu * nu
    beta_param = (1 - mu) * nu
    return beta(alpha, beta_param)


def sample_price(n, rng, price_tail):
    sigma = PRICE_SIGMA_LOG * price_tail
    prices = rng.lognormal(mean=PRICE_MU_LOG, sigma=sigma, size=n)
    return np.clip(prices, PRICE_MIN, PRICE_MAX)


def generate_dataset(
    n_per_group=100_000,
    seed=None,
    summer_scale=1.0,
    discount_scale=0.0,
    precision_scale=1.0,
    price_tail=1.0,
    n_visits=10,
):
    rng = np.random.default_rng(seed)
    frames = []

    for is_discount in [False, True]:
        n = n_per_group
        countries = rng.choice(COUNTRIES, size=n)

        conversions_pre = np.zeros(n, dtype=int)
        conversions = np.zeros(n, dtype=int)
        outcomes_pre = np.zeros(n)
        outcomes = np.zeros(n)

        for country in COUNTRIES:
            mask = countries == country
            count = mask.sum()
            if count == 0:
                continue

            dist_base = get_conversion_distribution(
                country, is_discount=False, is_summer=False,
                summer_scale=summer_scale, discount_scale=discount_scale, precision_scale=precision_scale,
            )
            user_propensity = dist_base.rvs(size=count, random_state=rng)

            p = COUNTRY_PARAMS[country]
            logit_shift = p["summer"] * summer_scale
            if is_discount:
                logit_shift += p["discount"] * discount_scale
            logit_p = logit_fn(np.clip(user_propensity, 1e-8, 1 - 1e-8))
            p_experiment = expit(logit_p + logit_shift)

            for _ in range(n_visits):
                prices_v = sample_price(n=count, rng=rng, price_tail=price_tail)
                conv_v = rng.binomial(1, user_propensity)
                conversions_pre[mask] += conv_v
                outcomes_pre[mask] += conv_v * prices_v

            for _ in range(n_visits):
                prices_v = sample_price(n=count, rng=rng, price_tail=price_tail)
                conv_v = rng.binomial(1, p_experiment)
                conversions[mask] += conv_v
                if is_discount and discount_scale > 0:
                    fee_v = SERVICE_FEE[country] * prices_v
                    outcomes[mask] += conv_v * ((1 - DISCOUNT) * prices_v - fee_v)
                else:
                    outcomes[mask] += conv_v * prices_v

        frames.append(pd.DataFrame({
            "country": countries,
            "is_discount": is_discount,
            "conversion_pre": conversions_pre,
            "outcome_pre": outcomes_pre,
            "conversion": conversions,
            "outcome": outcomes,
        }))

    return pd.concat(frames, ignore_index=True)
