from collections.abc import Callable

import numpy as np
from jaxtyping import Float

__version__ = "0.1.0"


def subsample(
    X: Float[np.ndarray, "n l_full d"],
    length: int,
    max_leading_time: int,
) -> Float[np.ndarray, "m n l d"]:
    """
    2.3 Generation of Near-border Sampl
    """
    X_new = []
    full_len = X.shape[1]
    for lt in range(max_leading_time):
        end_idx = full_len - lt
        start_idx = end_idx - length
        X_new.append(X[:, start_idx:end_idx, :])
    return np.asarray(X_new)


def _mix_temporal_neighbor(
    X: Float[np.ndarray, "l d"],
    Y: Float[np.ndarray, "l d"],
    score_X: float,
    score_Y: float,
    alpha: Float[np.ndarray, " k"],
) -> tuple[Float[np.ndarray, "k l d"], Float[np.ndarray, " k"]]:
    alpha_ = alpha[:, np.newaxis, np.newaxis]
    Z = alpha_ * X[np.newaxis, :, :] + (1.0 - alpha_) * Y[np.newaxis, :, :]
    score_Z = alpha * score_X + (1 - alpha) * score_Y
    return Z, score_Z


def synthesize(
    X: Float[np.ndarray, "m n l d"],
    scores: Float[np.ndarray, "m n"],
    imbalance_ratio: float,
    rng: np.random.Generator,
    beta_scale: float = 2.0,
) -> tuple[Float[np.ndarray, "k l d"], Float[np.ndarray, " k"]]:
    """
    2.4 Generation of Synthetic Samples
    """
    n_leading_time, n_samples = scores.shape
    s_sum: float = sum(scores)
    Z = []
    scores_Z = []
    for i in range(n_samples):
        for l in range(n_leading_time):  # noqa
            m = imbalance_ratio * scores[l, i] / s_sum
            alpha = rng.beta(beta_scale, beta_scale, size=m)
            z, score_z = _mix_temporal_neighbor(
                X[l, i], X[l + 1, i], scores[l, i], scores[l + 1, i], alpha
            )
            Z.append(z)
            scores_Z.append(score_z)
    return np.asarray(Z), np.asarray(scores_Z)


def resample(
    Z: Float[np.ndarray, "k l d"],
    scores: Float[np.ndarray, " k"],
    score_threshold: float,
    N: int,
    rng: np.random.Generator,
) -> Float[np.ndarray, "k2 l d"]:
    """
    2.5 Weighted Sampling Method
    """
    weights = np.clip(scores - score_threshold, 0.0, None)
    prob = weights / np.sum(weights)
    chosen_idx = rng.choice(len(Z), size=N, p=prob)
    return Z[chosen_idx]


def tsmote(
    X: Float[np.ndarray, "n l_full d"],
    score_fn: Callable,
    length: int,
    max_leading_time: int,
    imbalance_ratio: float,
    score_threshold: float,
    N: int,
    rng: np.random.Generator = np.random.default_rng(),
    beta_scale: float = 2.0,
):
    subsampled = subsample(X, length, max_leading_time)
    scores = score_fn(subsampled)
    z, z_scores = synthesize(subsampled, scores, imbalance_ratio, rng, beta_scale)
    resampled = resample(z, z_scores, score_threshold, N, rng)
    return resampled
