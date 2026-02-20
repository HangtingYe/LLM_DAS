# An expert in anomaly detection systems, I've designed the following Python function. It implements a specialized policy to generate anomalies that are intrinsically difficult for a One-Class SVM (OCSVM) to detect, leveraging the geometric principles of how OCSVM constructs its decision boundary.

# The core policy is as follows:

# 1.  **Identify "Borderline" Normal Samples:** OCSVM's decision boundary is defined by a subset of training points called support vectors. These points are the "borderline" cases that the model considers closest to being anomalous. Since we cannot access the support vectors directly, we use the `predict_score()` function as a proxy. The normal training samples (`X_train`) that receive the *highest* anomaly scores are the ones closest to the boundary. We select a pool of these high-scoring normal samples as our "borderline" candidates.

# 2.  **OCSVM-Specific Perturbation:** A hard anomaly for an OCSVM is one that lies just outside its learned boundary of normality. To create such a sample, we take a borderline normal point and "push" it slightly across the boundary. The most logical direction to push it is "outward," away from the dense center of the normal data cloud. We achieve this by:
#     *   Calculating the geometric center (mean) of the training data.
#     *   For each selected borderline sample, we determine the vector pointing from the data center to the sample.
#     *   We then move the borderline sample a very small, controlled distance along this outward-pointing vector.

# This targeted perturbation creates a new point that is semantically an anomaly (as it has been moved away from the normal region) but is geometrically very close to the decision boundary. Consequently, the OCSVM will assign it a low anomaly score, making it a "hard anomaly" that is difficult to distinguish from normal data.

# ```python
import numpy as np
def generate_hard_anomalies(n_samples: int, model, X_train: np.ndarray) -> np.ndarray:
    """
    Generates hard-to-detect anomalies for a trained One-Class SVM (OCSVM) model.

    This function implements an OCSVM-specific strategy to generate anomalies
    that receive low anomaly scores, making them difficult to detect. These
    anomalies are valuable for testing and improving the robustness of the detector.

    Anomaly Generation Policy:
    1.  **Identify Borderline Samples:** The function first identifies "borderline"
        normal samples from the training set. In the context of OCSVM, these are
        the points that lie closest to the decision boundary. We use the anomaly
        score (`model.predict_score()`) as a proxy, selecting the training samples
        with the highest scores (e.g., the top 10%) as our borderline candidates.
        These points are analogous to OCSVM's support vectors.

    2.  **Directed Outward Perturbation:** To create a hard anomaly, we take a
        borderline sample and push it slightly across the decision boundary. This
        is done by moving it in a direction pointing away from the center of the
        normal data distribution (calculated as the mean of X_train). This
        targeted "outward" push ensures the generated sample lies just outside
        the learned normal region, resulting in an anomaly score that is only
        marginally higher than that of a normal sample. The magnitude of this
-       push is kept small and is scaled relative to the standard deviation of
        the training data to ensure the anomaly remains close to the boundary.

    Args:
        n_samples (int): The number of hard anomaly samples to generate.
        model: A trained OCSVM-like model object that has a `predict_score`
               method. The method should accept a NumPy array and return a
               score for each sample, where a higher score indicates a higher
               degree of anomaly.
        X_train (np.ndarray): The training data, containing only normal samples,
                              used to train the model. Shape: (n_train_samples, n_features).

    Returns:
        np.ndarray: An array of generated hard anomalies of shape (n_samples, n_features).
    """
    # 1. All package imports are done inside the function.
    import numpy as np

    # --- Step 1: Identify a pool of "borderline" normal samples ---
    # These are the training samples that the OCSVM model is least confident about,
    # i.e., those with the highest anomaly scores. They act as proxies for the support vectors
    # that define the decision boundary.
    try:
        train_scores = model.predict_score(X_train)
    except Exception as e:
        raise ValueError(f"Failed to execute model.predict_score(X_train). Error: {e}")

    # Use a high percentile (e.g., 90th) to define the threshold for what constitutes a "borderline" sample.
    # This creates a pool of candidates from which to generate anomalies.
    # Using a percentile makes the selection robust to the scale of the scores.
    borderline_threshold = np.percentile(train_scores, 90)
    borderline_candidates = X_train[train_scores >= borderline_threshold]

    # Handle the edge case where all scores are identical, resulting in an empty candidate pool.
    # In this scenario, we fall back to using the entire training set as candidates.
    if len(borderline_candidates) == 0:
        borderline_candidates = X_train
    
    # --- Step 2: Select base samples from the borderline pool for generation ---
    # We randomly sample (with replacement) from our pool of borderline candidates.
    # This allows us to generate any number of anomalies, even more than the number of candidates.
    num_candidates = len(borderline_candidates)
    selected_indices = np.random.choice(num_candidates, size=n_samples, replace=True)
    base_samples = borderline_candidates[selected_indices]

    # --- Step 3: Generate hard anomalies via small, directed "outward" perturbations ---
    # The core of the OCSVM-specific policy is to push these borderline samples slightly "outward,"
    # away from the center of the normal data distribution. This moves them just across
    # the decision boundary, making them anomalies with low anomaly scores.

    # Calculate the geometric center (mean) of the normal training data.
    data_center = np.mean(X_train, axis=0)

    # Calculate a measure of the data's overall scale to make the perturbation magnitude robust.
    # We use the mean of the standard deviation across all features.
    data_scale = np.mean(np.std(X_train, axis=0))
    if data_scale < 1e-9:  # Handle case with zero or near-zero variance
        data_scale = 1.0

    # Calculate the displacement vectors pointing from the center to the base samples.
    direction_vectors = base_samples - data_center

    # Normalize the direction vectors to have unit length.
    # Add a small epsilon (1e-9) to the norm to avoid division by zero for any point at the exact center.
    norms = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
    normalized_directions = direction_vectors / (norms + 1e-9)

    # Define a small, random perturbation magnitude for each sample.
    # This magnitude is a small fraction of the overall data scale. The small step size ensures
    # the generated anomaly stays very close to the boundary, making it "hard".
    # The range (e.g., 0.05 to 0.2) means we push it 5% to 20% of the average feature std dev.
    perturbation_magnitudes = data_scale * np.random.uniform(low=0.05, high=0.2, size=(n_samples, 1))

    # Apply the perturbation to create the final hard anomalies:
    # new_point = base_point + magnitude * direction
    hard_anomalies = base_samples + perturbation_magnitudes * normalized_directions

    return hard_anomalies
# ```