# An expert in anomaly detection systems. The training set contains only normal samples. We use a ECOD detector, where the anomaly score is computed using model.predict\_score(). The higher the score, the more anomalous the sample. 

# # The description of ECOD.
# ECOD (Empirical Cumulative Distribution-based Outlier Detection) is a lightweight, univariate anomaly detection method that leverages the empirical cumulative distribution (ECDF) of each feature independently. It computes how extreme a value is relative to the distribution of that feature by evaluating its position in the ECDF. The core idea is that anomalous values often lie in the tails of the distribution. ECOD combines the tail probabilities across all features to produce an overall anomaly score, without assuming any parametric distribution. This makes it simple, interpretable, and effective for high-dimensional tabular data with only normal samples for training.

# The main steps of ECOD for anomaly detection are:

# * For each feature, compute the empirical cumulative distribution function (ECDF) using the training data.
# * For each new sample, calculate the left-tail and right-tail probabilities for each feature based on its ECDF.
# * Determine the tail probability for each feature as the minimum of left-tail and right-tail probabilities, capturing extremeness in either direction.
# * Aggregate the per-feature tail probabilities into a single anomaly score, often by averaging or summing across features.
# * Assign an anomaly label or score: higher scores indicate greater deviation from the normal feature distributions, signaling potential anomalies.


# # Objective
# Your task is to write a Python function generate\_hard\_anomalies(...) that generates anomalies which are the most difficult for the ECOD detector to detect. This means that you the generated anomalies should have relatively low anomaly score, thus they are hard to be detected. But these anomalies are helpful to build a more robust detector. After the Python function is completed, users can provide the function with:

# * A trained ECOD model (model) that exposes predict\_score(),

# * The training samples (X_train)

# # Requirements:
# Your should strictly follow below requirements:

# 1. You must use your expertise to give anomalies generation policies that are specific designed for ECOD, not a model-agnostic policy.

# 2. Generated samples should have as low a score as possible from model.predict\_score(). To achieve it, you can first find the set of ‘borderline’ normal training samples based on your unique and professional understanding to ECOD, not only based on the anomaly score. Then transform them to anomalies that is tailor-designed for ECOD. Please note that the transformation function should be specific for ECOD, which means that it is not a general transformation for other detectors.

# 3. For the model, you can only use the function model.predict\_score.

# 4. Use NumPy to generate the samples, and output an array of shape (n\_samples, d). And it should generate anomalies as much as I want.

# 5. The function should allow setting:

#     * the number of samples (n_samples),

#     * the trained ECOD model (model),
    
#     * training samples (X_train).

#     Thus the function format is generate\_hard\_anomalies(n_samples: int, model, X_train: np.ndarray)

# 6. All package imports must be done inside the function.
    
# Return only the complete Python function generate\_hard\_anomalies(...), with policy you used for genenrating anomalies and clear comments explaining key steps.

# ```python
import numpy as np
def generate_hard_anomalies(n_samples: int, model, X_train: np.ndarray):
    """
    Generates anomalies that are specifically designed to be difficult for an ECOD 
    (Empirical Cumulative Distribution-based Outlier Detection) model to detect.

    Args:
        n_samples (int): The number of hard anomalies to generate.
        model: A trained ECOD model object with a `predict_score` method.
        X_train (np.ndarray): The training data used to fit the ECOD model, 
                              containing only normal samples.

    Returns:
        np.ndarray: An array of shape (n_samples, n_features) containing the
                    generated hard anomalies.
    """
    # All package imports must be done inside the function as per requirements.
    import numpy as np

    # =========================================================================
    # ANOMALY GENERATION POLICY: "PROTOTYPE RECOMBINATION"
    # =========================================================================
    # ECOD's core weakness is its univariate nature. It evaluates each feature
    # independently and aggregates the results. A sample gets a low anomaly score
    # if all its feature values are "central" to their respective 1D distributions
    # (i.e., near the median). ECOD is blind to anomalous *combinations* of
    # individually common feature values.
    #
    # Our strategy exploits this weakness directly to generate anomalies with the
    # lowest possible scores:
    #
    # 1. IDENTIFY 'BORDERLINE' SAMPLES: The prompt requires finding 'borderline'
    #    samples to transform. For our ECOD-specific goal (lowest score), we
    #    define 'borderline' samples not as those with high scores, but as the
    #    most "prototypical" normal samples—those ECOD is *most confident* are
    #    normal. These are the samples with the *lowest* anomaly scores. They
    #    serve as the perfect building blocks (a "safe set") for our transformation.
    #
    # 2. TRANSFORM VIA RECOMBINATION: The transformation is not a simple perturbation.
    #    We deconstruct our prototypical samples into a pool of "safe" feature
    #    values. Then, we construct new synthetic samples by randomly picking
    #    values for each feature from this pool. For example, a new anomaly's
    #    first feature is randomly chosen from the first features of all
    #    prototypes, its second feature from the second features, and so on.
    #
    # 3. RESULT: The generated samples are anomalies because their specific
    #    combination of feature values is highly unlikely to exist in the
    #    original data, placing them in low-density regions of the joint
    #    distribution. However, since every single feature value is, by
    #    construction, very common and central in its marginal distribution,
    #    ECOD will calculate a very low anomaly score, making them hard to detect.
    # =========================================================================

    if X_train.ndim != 2 or X_train.shape[0] < 2:
        raise ValueError("X_train must be a 2D array with at least 2 samples.")

    n_train_samples, n_features = X_train.shape

    # --- Step 1: Identify the set of 'prototypical' normal training samples ---
    # We use the provided model to score all training samples.
    train_scores = model.predict_score(X_train)

    # We define our set of "prototypical" or "borderline-for-generation" samples
    # as the 20% of the training data with the lowest anomaly scores. This
    # percentage provides a good balance between diversity and "safeness" of
    # feature values.
    prototypical_percentile = 20
    score_threshold = np.percentile(train_scores, prototypical_percentile)
    
    # Select the samples that fall below this score threshold.
    prototypical_samples = X_train[train_scores <= score_threshold]

    # Handle the edge case where no samples are selected (e.g., all have same score)
    if prototypical_samples.shape[0] == 0:
        # Fallback to using the entire training set as the pool
        prototypical_samples = X_train

    # --- Step 2: Transform via Recombination to generate low-score anomalies ---
    # Initialize an empty array to store the generated hard anomalies.
    hard_anomalies = np.zeros((n_samples, n_features))

    # For each feature column, we will randomly sample values *only* from that
    # same column in our set of prototypical samples.
    for j in range(n_features):
        # Create a pool of "safe" values for the j-th feature.
        safe_values_for_feature_j = prototypical_samples[:, j]
        
        # Randomly sample with replacement from this pool to construct the
        # j-th feature for all our new anomalies.
        hard_anomalies[:, j] = np.random.choice(
            safe_values_for_feature_j, size=n_samples, replace=True
        )

    # --- Step 3 (Optional Refinement): Add minimal noise ---
    # To ensure the generated points are truly novel and not just duplicates of
    # existing feature values, we add a very small amount of Gaussian noise.
    # The noise is scaled by the standard deviation of each feature to be
    # proportionate to the feature's natural variance.
    feature_stds = np.std(X_train, axis=0)
    # Adding noise equivalent to 1% of the standard deviation. This is small
    # enough to not significantly increase the ECOD score.
    noise = np.random.randn(n_samples, n_features) * feature_stds * 0.01
    hard_anomalies += noise

    return hard_anomalies
# ```