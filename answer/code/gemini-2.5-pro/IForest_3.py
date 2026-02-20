# ```python
import numpy as np
def generate_hard_anomalies(n_samples: int, model, X_train: 'np.ndarray') -> 'np.ndarray':
    """
    Generates anomalies that are difficult for a trained Isolation Forest model to detect.

    This function implements a policy specifically designed to exploit the mechanics of the
    Isolation Forest algorithm. The generated 'hard' anomalies have low anomaly scores,
    making them likely to be misclassified as normal samples.

    Anomaly Generation Policy:
    --------------------------
    The core idea behind Isolation Forest (IForest) is that anomalies are "few and different,"
    making them easy to isolate with a few random splits. Consequently, they have short average
    path lengths in the trees, leading to high anomaly scores. Conversely, normal points are
    located in dense regions and require many splits to be isolated, resulting in long path
    lengths and low anomaly scores.

    This function generates hard anomalies by creating synthetic points that mimic the
    characteristics of the most "normal" points in the training set.

    1.  **Identify 'Core' Normal Samples:** We first identify the most normal samples within the
        training data. These are the points that the IForest model finds hardest to isolate,
        meaning they lie in the densest regions of the data distribution and have the lowest
        anomaly scores. These are our "borderline" or "core" normal samples.

    2.  **Generate Anomalies via Recombination:** Instead of adding noise or creating points
        far from the data distribution (which would be easy for IForest to detect), we
        generate new samples by recombining features from pairs of these 'core' normal samples.
        A new point is created by taking a random subset of feature values from one core sample
        and the remaining feature values from a second core sample.

    Why this is a hard anomaly for IForest:
    -   The generated sample is an anomaly because it is a synthetic point that does not exist
        in the original training data.
    -   However, it is constructed entirely from feature values that are common and normal.
    -   For any given feature, the value of the synthetic point is a value from a dense region.
        Therefore, any single random split is unlikely to isolate it.
    -   The model needs to perform many splits across multiple dimensions to finally isolate this
        recombined point, leading to a long path length and, consequently, a low anomaly score,
        making it "hard" to detect.

    Args:
        n_samples (int): The number of hard anomaly samples to generate.
        model: A trained Isolation Forest model object that has a `predict_score` method.
               The method should return a score where a higher value indicates a more
               anomalous sample.
        X_train (np.ndarray): The training data used to train the model, with shape
                              (n_train_samples, n_features).

    Returns:
        np.ndarray: An array of generated hard anomalies with shape (n_samples, n_features).
    """
    # 1. All package imports must be done inside the function.
    import numpy as np

    # --- Step 1: Identify 'core' normal samples from the training set ---
    # According to the IForest principle, the most normal samples are those that are
    # hardest to isolate, meaning they have the longest path lengths and thus the
    # lowest anomaly scores.
    # We compute the anomaly scores for all training samples.
    train_scores = model.predict_score(X_train)

    # We sort the training samples by their anomaly score in ascending order.
    # The samples at the beginning of this sorted list are the most "normal".
    sorted_indices = np.argsort(train_scores)
    sorted_X_train = X_train[sorted_indices]

    # We define the pool of "core" normal samples. Using the 20% most normal samples
    # provides a robust base for generating new points.
    core_sample_pool_size = max(10, int(0.2 * len(X_train))) # Ensure a minimum pool size
    core_samples = sorted_X_train[:core_sample_pool_size]

    n_features = X_train.shape[1]
    generated_anomalies = []

    # --- Step 2: Generate anomalies through feature recombination ---
    # We create `n_samples` new points by combining features from pairs of core samples.
    for _ in range(n_samples):
        # Randomly select two distinct 'parent' samples from the core pool.
        # Sampling with replacement is used to allow for a large number of combinations.
        parent_indices = np.random.choice(len(core_samples), 2, replace=True)
        parent1 = core_samples[parent_indices[0]]
        parent2 = core_samples[parent_indices[1]]

        # Create a random binary mask. This mask determines which features are taken
        # from parent1 and which are from parent2.
        # A value of 1 in the mask means the feature comes from parent1.
        # A value of 0 means the feature comes from parent2.
        mask = np.random.randint(0, 2, size=n_features).astype(bool)

        # In the rare case the mask is all ones or all zeros, the generated sample
        # would be an exact copy of a parent. We flip one bit to ensure recombination.
        if mask.all() or not mask.any():
            flip_idx = np.random.randint(0, n_features)
            mask[flip_idx] = not mask[flip_idx]

        # Construct the new hard anomaly by taking features from parents based on the mask.
        # This is an efficient way to perform the recombination:
        # new_sample[i] = parent1[i] if mask[i] is True else parent2[i]
        new_anomaly = np.where(mask, parent1, parent2)
        
        generated_anomalies.append(new_anomaly)

    # Convert the list of generated samples into a NumPy array and return.
    return np.array(generated_anomalies)

# ```