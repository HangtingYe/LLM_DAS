# ```python
import numpy as np
def generate_hard_anomalies(n_samples: int, model, X_train: np.ndarray) -> np.ndarray:
    """
    Generates anomalies that are difficult for a PCA-based anomaly detector to detect.

    Args:
        n_samples (int): The number of anomaly samples to generate.
        model: The trained PCA detector. This function is designed for models where
               the anomaly score is based on reconstruction error, but it only
               requires the model to be passed as an argument per the problem
               specification. The generation logic itself does not call model methods.
        X_train (np.ndarray): The training data, containing only normal samples.
                              Shape (n_train_samples, d).

    Returns:
        np.ndarray: An array of generated hard anomalies of shape (n_samples, d).
    """
    # All package imports must be done inside the function.
    import numpy as np

    # ==============================================================================
    # Anomaly Generation Policy (PCA-Specific)
    # ==============================================================================
    #
    # 1. PCA's Weakness: PCA-based anomaly detection identifies anomalies by
    #    measuring the reconstruction error. A sample has a high error (and is
    #    thus anomalous) if it lies far from the low-dimensional subspace
    #    (hyperplane) spanned by the principal components of the normal training data.
    #    The weakness is that a point can be a significant outlier in terms of its
    #    position *within* this subspace, but if it lies close to the subspace itself,
    #    its reconstruction error will be very low, making it a "hard" anomaly.
    #
    # 2. Identifying "Borderline" Normal Samples: To create compelling anomalies,
    #    we start from the edges of the normal data distribution. In the context of
    #    PCA, which is centered around the data's mean, the most extreme or
    #    "borderline" normal points are those furthest from the mean of the training
    #    data. These points define the outer boundary of the normal data cloud.
    #
    # 3. Finding "Safe" Directions for Perturbation: The key to generating
    #    low-score anomalies is to move a point in a direction that is parallel to
    #    the PCA subspace. We can approximate such directions by taking the vector
    #    between two random points from the training data. Since the training data
    #    itself defines the subspace, a vector connecting any two points within it
    #    will also largely lie within that same subspace.
    #
    # 4. Generation Strategy:
    #    a. Select a "borderline" normal sample (a point far from the data's center)
    #       as a starting point.
    #    b. Determine a "safe" direction by taking the vector between two other
    #       randomly chosen normal samples.
    #    c. Create the hard anomaly by pushing the borderline sample far out along
    #       this safe direction. The resulting point is far from the original data
    #       cloud (making it a semantic anomaly) but remains close to the principal
    #       component subspace (giving it a low reconstruction error).
    #
    # This method is specifically tailored to exploit how PCA computes anomaly scores
    # and does not rely on a generic, model-agnostic approach.
    # ==============================================================================

    # --- Step 0: Input validation ---
    if not isinstance(n_samples, int) or n_samples < 0:
        raise ValueError("n_samples must be a non-negative integer.")
    if n_samples == 0:
        return np.array([]).reshape(0, X_train.shape[1])
    if not isinstance(X_train, np.ndarray) or X_train.ndim != 2:
        raise ValueError("X_train must be a 2D NumPy array.")
    if len(X_train) < 2:
        raise ValueError("X_train must contain at least 2 samples to define a direction.")

    n_train_samples, n_features = X_train.shape

    # --- Step 1: Identify "borderline" normal samples ---
    # These are samples that are furthest from the center of the training data distribution.
    train_mean = np.mean(X_train, axis=0)
    distances_from_mean = np.linalg.norm(X_train - train_mean, axis=1)

    # We create a pool of borderline candidates to select our starting points from.
    # The pool size is the larger of n_samples or 10% of the training set.
    n_borderline_candidates = max(n_samples, int(0.1 * n_train_samples))
    n_borderline_candidates = min(n_borderline_candidates, n_train_samples) # Cap at training set size.
    
    # Get the indices of the points with the largest distances.
    borderline_indices = np.argsort(distances_from_mean)[-n_borderline_candidates:]
    
    # We use the maximum distance to scale our perturbation later.
    max_normal_distance = distances_from_mean[borderline_indices[-1]]

    # --- Step 2: Generate the hard anomalies ---
    generated_anomalies = []
    
    while len(generated_anomalies) < n_samples:
        # a. Randomly select a borderline sample as the starting point.
        start_point_idx = np.random.choice(borderline_indices)
        x_start = X_train[start_point_idx]

        # b. Randomly select two distinct points from the training set to define a direction.
        idx_a, idx_b = np.random.choice(n_train_samples, 2, replace=False)
        x_a, x_b = X_train[idx_a], X_train[idx_b]

        # c. Calculate the direction vector. This direction is likely to lie within
        #    the principal subspace of the normal data.
        direction = x_b - x_a
        
        # d. Normalize the direction vector to have a unit length.
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-9:
            # This is rare but can happen if two samples are identical.
            # We skip this iteration and try again.
            continue
        
        unit_direction = direction / direction_norm

        # e. Define a scaling factor 'alpha' to push the point far away.
        # We make the perturbation proportional to the size of the existing data cloud
        # by using `max_normal_distance`. The random factor ensures variety.
        # A factor between 1.5 and 3.0 ensures the new point is clearly outside
        # the original distribution.
        alpha = max_normal_distance * (1.5 + np.random.rand() * 1.5)

        # f. Generate the anomaly by moving the starting point along the safe direction.
        x_anomaly = x_start + alpha * unit_direction
        generated_anomalies.append(x_anomaly)

    return np.array(generated_anomalies).reshape(n_samples, n_features)

# ```