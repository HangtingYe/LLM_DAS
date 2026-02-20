# An expert in anomaly detection systems. The training set contains only normal samples. We use a PCA detector, where the anomaly score is computed using model.predict\_score(). The higher the score, the more anomalous the sample. 

# # The description of PCA.
# Principal Component Analysis (PCA) is a commonly used method for dimensionality reduction and anomaly detection. It identifies directions in the data (called principal components) along which the variance is maximized. In anomaly detection, PCA can be used to reconstruct each sample from the top principal components and compute the reconstruction error as the anomaly score. Samples with higher reconstruction error are considered more anomalous.

# The main steps of PCA for anomaly detection are:

# * Compute the mean of the training data.

# * Subtract the mean from each sample to center the data.

# * Compute the covariance matrix of the centered data.

# * Perform eigen-decomposition on the covariance matrix to obtain eigenvectors and eigenvalues.

# * Select the top eigenvectors corresponding to the largest eigenvalues to serve as the principal components.

# * Project each sample onto the principal components and reconstruct it.

# * Compute the reconstruction error for each sample as the anomaly score.

# * Rank samples by their anomaly scores to detect anomalies.

# # Objective
# Your task is to write a Python function generate\_hard\_anomalies(...) that generates anomalies which are the most difficult for the PCA detector to detect. This means that you the generated anomalies should have relatively low anomaly score, thus they are hard to be detected. But these anomalies are helpful to build a more robust detector. After the Python function is completed, users can provide the function with:

# * A trained PCA model (model) that exposes predict\_score(),

# * The training samples (X_train)

# # Requirements:
# Your should strictly follow below requirements:

# 1. You must use your expertise to give anomalies generation policies that are specific designed for PCA, not a model-agnostic policy.

# 2. Generated samples should have as low a score as possible from model.predict\_score(). To achieve it, you can first find the set of ‘borderline’ normal training samples based on your unique and professional understanding to PCA, not only based on the anomaly score. Then transform them to anomalies that is tailor-designed for PCA. Please note that the transformation function should be specific for PCA, which means that it is not a general transformation for other detectors.

# 3. For the model, you can only use the function model.predict\_score.

# 4. Use NumPy to generate the samples, and output an array of shape (n\_samples, d). And it should generate anomalies as much as I want.

# 5. The function should allow setting:

#     * the number of samples (n_samples),

#     * the trained PCA model (model),
    
#     * training samples (X_train).

#     Thus the function format is generate\_hard\_anomalies(n_samples: int, model, X_train: np.ndarray)

# 6. All package imports must be done inside the function.
    
# Return only the complete Python function generate\_hard\_anomalies(...), with policy you used for genenrating anomalies and clear comments explaining key steps.

# ```python
import numpy as np
def generate_hard_anomalies(n_samples: int, model, X_train: np.ndarray) -> np.ndarray:
    """
    Generates anomalies that are hard for a PCA-based anomaly detector to identify.

    This function implements a policy specifically designed to exploit the weakness of
    PCA-based anomaly detection, which relies on reconstruction error.

    Policy for Generating Hard Anomalies for PCA:
    ---------------------------------------------
    1.  PCA's Blind Spot: PCA-based anomaly detectors learn a low-dimensional
        subspace (the "principal subspace") that captures the maximum variance of the
        normal training data. The anomaly score is the reconstruction error, which
        is the distance of a sample to this subspace. Consequently, PCA is good at
        detecting samples that lie far from this subspace. Its primary weakness is
        detecting anomalies that lie *within* this learned subspace but are far from
        the cluster of normal data. Such points are "anomalous" because they are in
        an unobserved region, but have a low reconstruction error, fooling the detector.

    2.  Identification of 'Borderline' Normal Samples: Instead of using anomaly scores,
        we identify borderline samples from a geometric perspective. We define "borderline"
        samples as those that are furthest from the center (mean) of the normal data
        distribution. These samples represent the extremities of the data cloud along the
        principal components, as these are the directions of highest variance.

    3.  PCA-Specific Transformation (Extrapolation): We generate anomalies by taking these
        borderline samples and pushing them further away from the data center along the
        same direction. The transformation is an extrapolation:
            direction = borderline_sample - mean
            anomaly = borderline_sample + alpha * direction
        This moves the point along a line that is likely well-contained within the
        principal subspace. Because the generated anomaly stays very close to the
        subspace, its reconstruction error (the anomaly score) remains low, making it
        a "hard" anomaly for the PCA detector to identify.

    Args:
        n_samples (int): The number of hard anomaly samples to generate.
        model: A trained PCA model object that has a `predict_score` method.
               This argument is kept for API consistency, but the generation logic
               relies on the principles of PCA and the training data, not on querying
               the model.
        X_train (np.ndarray): The training data, containing only normal samples.
                              Shape (n_train_samples, n_features).

    Returns:
        np.ndarray: An array of generated hard anomalies of shape (n_samples, n_features).
    """
    # 1. All package imports must be done inside the function.
    import numpy as np

    # Ensure X_train is a numpy array
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
        
    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array.")

    # 2. Compute the center (mean) of the normal training data.
    data_mean = np.mean(X_train, axis=0)

    # 3. Identify the set of 'borderline' normal training samples.
    #    We define these as the samples furthest from the data mean in Euclidean distance.
    #    These samples lie at the edges of the normal data distribution.
    distances_from_mean = np.linalg.norm(X_train - data_mean, axis=1)
    
    # Get the indices of samples sorted by their distance from the mean, in descending order.
    borderline_indices = np.argsort(distances_from_mean)[::-1]

    # We will use the top 5% of the most distant samples as our base for generating anomalies.
    # This provides a diverse set of "extreme" directions to extrapolate along.
    num_borderline_candidates = max(1, int(len(borderline_indices) * 0.05))
    top_borderline_indices = borderline_indices[:num_borderline_candidates]

    # 4. Generate anomaly samples by extrapolation.
    generated_anomalies = []
    for _ in range(n_samples):
        # Randomly select a borderline sample to serve as the base for the anomaly.
        base_index = np.random.choice(top_borderline_indices)
        borderline_sample = X_train[base_index]

        # The transformation is an extrapolation along the vector from the mean to the
        # borderline sample. This direction is assumed to lie within the principal subspace.
        direction_vector = borderline_sample - data_mean

        # We push the sample along this direction by a small, random factor 'alpha'.
        # A small factor (e.g., 0.1 to 0.4) ensures the new point doesn't stray too far
        # from the learned linear subspace, thus keeping its reconstruction error low.
        alpha = np.random.uniform(0.1, 0.4)
        
        # Generate the hard anomaly.
        anomaly = borderline_sample + alpha * direction_vector
        generated_anomalies.append(anomaly)

    # 5. Return the generated samples as a NumPy array.
    return np.array(generated_anomalies)
# ```