# ```python
import numpy as np
from typing import Any

def generate_hard_anomalies(n_samples: int, model: Any, X_train: np.ndarray) -> np.ndarray:
    """
    Generates anomalies that are difficult for an ECOD detector to identify.

    These "hard anomalies" are designed to have low anomaly scores by exploiting
    the core mechanism of ECOD, which evaluates each feature independently.

    Args:
        n_samples (int): The number of hard anomaly samples to generate.
        model (Any): The trained ECOD model. It must have a `predict_score` method.
                     This function uses the model conceptually but does not call its methods
                     during generation, as the generation logic is based on first principles
                     of how ECOD works. The user can use the model to verify the low scores
                     of the generated samples.
        X_train (np.ndarray): The training data (normal samples) used to train the model,
                              with shape (n_train_samples, n_features).

    Returns:
        np.ndarray: An array of generated hard anomalies with shape (n_samples, n_features).

    Anomaly Generation Policy:
    --------------------------
    This function exploits the fundamental weakness of ECOD: its assumption of feature independence.
    ECOD assigns a low anomaly score to a sample if each of its feature values is common or "central"
    within the distribution of that feature in the training data. It does not consider the
    plausibility of the *combination* of these feature values.

    The policy is to generate new samples by combining individually common feature values in a way
    that breaks the natural correlation structure of the data. Such a sample is a multivariate anomaly
    but will appear normal to a univariate detector like ECOD.

    1.  **Identify the 'Safe Zone' for Each Feature**: Instead of finding "borderline" samples (which
        may already have some features in the tails), we identify the opposite: the most "normal" or
        "central" range of values for each feature. A robust way to define this safe zone is the
        Interquartile Range (IQR), from the 25th percentile (Q1) to the 75th percentile (Q3). Any
        value within this range will, by definition, have a large ECDF tail probability (>= 0.25)
        and thus contribute very little to the final anomaly score.

    2.  **Construct Anomalies by Independent Sampling**: We generate each new anomaly by independently
        sampling a value for each feature from its respective "safe zone". For feature `j`, we pick a
        random value uniformly from the range [Q1_j, Q3_j] calculated from the training data.

    3.  **Resulting Hard Anomaly**: The resulting sample vector consists entirely of feature values
        that ECOD considers highly normal. However, because the sampling is done independently for each
        feature, the correlation structure present in `X_train` is completely destroyed. For example, in a
        dataset where height and weight are positively correlated, this method might combine a "normal"
        short height with a "normal" heavy weight, creating a combination that is physically impossible
        or at least highly anomalous, yet invisible to ECOD.
    """
    # 1. All package imports must be done inside the function.
    import numpy as np

    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array.")
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    _, n_features = X_train.shape

    # 2. Identify the 'Safe Zone' for each feature using the Interquartile Range (IQR).
    # We calculate the 25th (Q1) and 75th (Q3) percentiles for each feature column.
    # These values define the boundaries of the most common values, where ECOD is least
    # sensitive.
    # axis=0 ensures percentiles are computed column-wise (for each feature).
    q1_per_feature = np.percentile(X_train, 25, axis=0)
    q3_per_feature = np.percentile(X_train, 75, axis=0)

    # 3. Initialize an array to store the generated hard anomalies.
    hard_anomalies = np.zeros((n_samples, n_features))

    # 4. Construct anomalies by independent sampling from the 'Safe Zone' of each feature.
    # For each feature, we generate `n_samples` values by drawing from a uniform distribution
    # defined by its Q1 and Q3. This ensures every generated value is "in-distribution"
    # from a univariate perspective.
    for i in range(n_features):
        # The low and high bounds for the uniform distribution are the Q1 and Q3
        # of the i-th feature.
        low_bound = q1_per_feature[i]
        high_bound = q3_per_feature[i]
        
        # If Q1 and Q3 are the same (e.g., for a constant feature), we just use that value.
        if low_bound == high_bound:
            hard_anomalies[:, i] = low_bound
        else:
            hard_anomalies[:, i] = np.random.uniform(low=low_bound,
                                                     high=high_bound,
                                                     size=n_samples)

    # The resulting `hard_anomalies` array contains samples where each feature value
    # is individually common, but their combination is novel and likely anomalous
    # due to the broken correlation structure. ECOD will assign these a low score.
    return hard_anomalies
# ```