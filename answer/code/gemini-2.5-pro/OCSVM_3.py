# ```python
import numpy as np
def generate_hard_anomalies(n_samples: int, model, X_train: np.ndarray) -> np.ndarray:
    """
    Generates anomalies that are difficult for a trained One-Class SVM (OCSVM) model to detect.

    This function implements a specialized policy tailored to the mechanics of OCSVM.
    The generated anomalies are "hard" because they are designed to lie very close to the
    decision boundary on the anomalous side, resulting in low anomaly scores that
    are just above the threshold for being considered normal.

    OCSVM-Specific Anomaly Generation Policy: "Outward Boundary Crossing"
    ---------------------------------------------------------------------
    The core idea is to identify normal samples that are already close to the decision
    boundary and then push them just across it. This is achieved in three steps:

    1.  Identify 'Borderline' Normal Samples: In OCSVM, the decision boundary is defined
        by the support vectors. While we cannot access them directly, we can infer which
        training samples are most likely to be support vectors or close to them. These are
        the "borderline" samples that lie on or just inside the boundary. They will have
        the highest anomaly scores among all the normal training samples. We identify
        these by scoring the entire training set and selecting the samples with the
        highest scores (e.g., the top 20%).

    2.  Define a Perturbation Direction: The OCSVM boundary encloses the dense region
        of normal data. Therefore, the most efficient way to cross this boundary from
        the inside is to move directly away from the center of the data cloud. We
        calculate this direction for each borderline sample by taking the vector pointing
        from the data's centroid (mean of X_train) to the borderline sample.

    3.  Generate the Anomaly: To create a hard anomaly, we take a borderline sample and
        move it a small, controlled distance (epsilon) along the pre-defined outward
        direction. This perturbation is just large enough to push the sample over the
        decision boundary, making it an anomaly, but small enough to ensure its anomaly
        score remains low, making it difficult to distinguish from normal samples.

    Args:
        n_samples (int): The number of hard anomalies to generate.
        model: A trained OCSVM model object that has a `predict_score()` method.
               Higher scores from this method indicate a higher degree of anomaly.
        X_train (np.ndarray): The training data (normal samples) used to train the model,
                              with shape (n_training_samples, n_features).

    Returns:
        np.ndarray: An array of generated hard anomalies with shape (n_samples, n_features).
    """
    # 1. All package imports must be done inside the function.
    import numpy as np

    # --- Step 1: Identify 'Borderline' Normal Samples ---
    # Score all training samples to find those closest to the decision boundary.
    # These samples will have the highest anomaly scores within the training set.
    train_scores = model.predict_score(X_train)

    # We define "borderline" samples as those in the top 20% of scores.
    # Using a percentile is robust and adapts to the score distribution.
    # This threshold identifies samples that are likely support vectors or near them.
    percentile_threshold = np.percentile(train_scores, 80)
    borderline_indices = np.where(train_scores >= percentile_threshold)[0]
    
    # In the rare case that all scores are identical, all samples are borderline.
    if len(borderline_indices) == 0:
        borderline_indices = np.arange(len(X_train))
        
    borderline_samples = X_train[borderline_indices]

    # --- Step 2: Define a Perturbation Direction ---
    # Calculate the center of the normal data distribution. This serves as an anchor
    # from which to push the borderline samples outwards.
    data_center = np.mean(X_train, axis=0)

    # --- Step 3: Generate Anomalies ---
    generated_anomalies = []
    for _ in range(n_samples):
        # Randomly select a borderline sample as the base for our new anomaly.
        base_idx = np.random.randint(0, len(borderline_samples))
        base_sample = borderline_samples[base_idx]

        # Calculate the direction vector: from the data center to the base sample.
        # Moving along this vector pushes the sample away from the core of the
        # normal data distribution, making it the most likely direction to cross
        # the OCSVM boundary.
        direction_vector = base_sample - data_center

        # Normalize the direction vector to ensure the perturbation magnitude is
        # controlled solely by epsilon, not by the sample's original distance from the center.
        norm = np.linalg.norm(direction_vector)
        if norm > 1e-6:  # Avoid division by zero for samples at the center
            direction_vector /= norm

        # Epsilon is the small step size for the perturbation. We use a small
        # random value to introduce variety while ensuring the generated sample
        # remains close to the boundary (i.e., a "hard" anomaly).
        # A small multiplier (e.g., 0.1) of the data's standard deviation can also be a good heuristic,
        # but a simple uniform value often works well. Here we use a range relative to the direction vector's unit length.
        epsilon = np.random.uniform(0.05, 0.2)

        # Create the new hard anomaly by taking a small step from the base sample
        # in the calculated outward direction.
        hard_anomaly = base_sample + epsilon * direction_vector
        generated_anomalies.append(hard_anomaly)

    return np.array(generated_anomalies)

# ```