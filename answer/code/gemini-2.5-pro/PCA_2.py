# ```python
import numpy as np
def generate_hard_anomalies(n_samples: int, model, X_train: np.ndarray) -> np.ndarray:
    """
    Generates anomalies that are difficult for a PCA-based anomaly detector to detect.

    These "hard" anomalies are designed to have low reconstruction error, and thus a low anomaly
    score, by being positioned far from the center of the normal data but still close to the
    principal subspace learned by the PCA model.

    Args:
        n_samples (int): The number of anomaly samples to generate.
        model: The trained PCA detector. The function only uses this to adhere to the API,
               its internal parameters are not accessed. The user would use
               `model.predict_score()` on the output of this function to verify the
               generated samples have low scores.
        X_train (np.ndarray): The training data, containing only normal samples.
                              Shape: (n_train_samples, n_features).

    Returns:
        np.ndarray: An array of generated hard anomaly samples.
                    Shape: (n_samples, n_features).

    ----------------------------------------------------------------------------------------------
    Anomaly Generation Policy (PCA-Specific):
    ----------------------------------------------------------------------------------------------
    1.  **Understanding the PCA Detector's Weakness**: PCA-based anomaly detection works by
        calculating the reconstruction error. A sample has a low error (and thus a low anomaly
        score) if it lies close to the principal subspace (the hyperplane spanned by the
        principal components). This subspace captures the directions of highest variance in the
        training data. Therefore, an anomaly that is "hiding" within or very close to this
        subspace will be difficult for PCA to detect.

    2.  **Identifying 'Borderline' Normal Samples**: Instead of just using samples with high
        anomaly scores, we identify 'borderline' samples in a way that is geometrically
        meaningful for PCA. We define these as the normal samples that are furthest from the
        data's center (mean). These points are "extreme" but are still considered normal, and
        they play a significant role in defining the orientation and extent of the principal
        subspace.

    3.  **PCA-Specific Transformation**: To create a hard anomaly, we take a 'borderline' sample
        and push it further away from the data center *along the direction it already lies*.
        This transformation has two key effects:
        a.  It moves the point into a sparse, previously unobserved region of the feature
            space, making it a legitimate anomaly by definition (i.e., it's a novel outlier).
        b.  Crucially, because the perturbation is applied along a vector that is already a
            part of the high-variance data manifold (the line from the center to an extreme
            point), the new point remains very close to the principal subspace.
        This results in a sample that is semantically anomalous but has a low reconstruction
        error, making it a "hard" anomaly for the PCA detector.
    ----------------------------------------------------------------------------------------------
    """
    # 1. All package imports must be done inside the function.
    import numpy as np

    if n_samples == 0:
        return np.array([])

    if X_train.shape[0] == 0:
        raise ValueError("X_train cannot be empty.")

    # 2. Compute the center of the training data. This serves as a robust proxy for the
    #    mean learned by the PCA model during its fitting process.
    data_center = np.mean(X_train, axis=0)

    # 3. Identify 'borderline' normal samples. These are the samples furthest from the data center,
    #    as they define the boundaries of the normal data distribution along high-variance directions.
    #    - Calculate the Euclidean distance of each training sample from the center.
    distances = np.linalg.norm(X_train - data_center, axis=1)

    #    - Identify the indices of the top 10% furthest samples to serve as our pool of
    #      borderline candidates. We ensure we have at least one candidate.
    n_candidates = max(1, int(0.1 * len(X_train)))
    borderline_indices = np.argsort(distances)[-n_candidates:]
    borderline_samples = X_train[borderline_indices]

    # 4. Select a set of starting points for our anomalies by randomly sampling (with replacement)
    #    from the pool of borderline samples. This allows generating n_samples even if n_samples > n_candidates.
    random_indices = np.random.choice(len(borderline_samples), size=n_samples, replace=True)
    start_points = borderline_samples[random_indices]

    # 5. Generate the hard anomalies by applying the PCA-specific transformation.
    #    - First, determine a reasonable magnitude for the perturbation. We scale it based on the
    #      average distance of the borderline points themselves. Pushing by 20%-40% of this
    #      distance is a good heuristic.
    avg_borderline_dist = np.mean(distances[borderline_indices])
    
    generated_anomalies = []
    for point in start_points:
        #    - Calculate the unit vector pointing from the data center to the borderline point.
        direction = point - data_center
        norm = np.linalg.norm(direction)

        # This check is for safety, but highly unlikely to be triggered since we select
        # points furthest from the center.
        if norm == 0:
            continue
        
        unit_direction = direction / norm

        #    - Define a random perturbation magnitude to create variety in the anomalies.
        #      We push the point by a factor of 0.2 to 0.4 of the average borderline distance.
        perturbation_magnitude = avg_borderline_dist * np.random.uniform(0.2, 0.4)

        #    - Create the new anomaly by moving the original borderline point further along its direction.
        anomaly = point + unit_direction * perturbation_magnitude
        generated_anomalies.append(anomaly)

    return np.array(generated_anomalies)
# ```