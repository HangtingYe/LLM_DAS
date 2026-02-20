# ```python
import numpy as np
from typing import Any
def generate_hard_anomalies(n_samples: int, model: Any, X_train: np.ndarray) -> np.ndarray:
    """
    Generates anomalies that are difficult for a trained One-Class SVM (OCSVM) to detect.

    The generated anomalies are designed to have low anomaly scores, placing them near the
    decision boundary of the normal class.

    Args:
        n_samples (int): The number of hard anomaly samples to generate.
        model (Any): The trained OCSVM detector. Must have a `predict_score` method
                     that returns a higher score for more anomalous samples.
        X_train (np.ndarray): The training data (normal samples) used to train the model,
                              with shape (n_train_samples, n_features).

    Returns:
        np.ndarray: An array of generated hard anomalies with shape (n_samples, n_features).
    """
    # =================================================================================
    # Anomaly Generation Policy for OCSVM
    # =================================================================================
    # OCSVM works by creating a boundary (a hyperplane in a high-dimensional feature
    # space) that encloses the majority of the normal training data. The samples that
    # lie on this boundary are the support vectors. Hard-to-detect anomalies are those
    # that lie just outside this boundary.
    #
    # Our strategy is to first identify the "borderline" normal samples from the
    # training set. These are the normal samples that the model finds most "unusual",
    # meaning they have the highest anomaly scores among all normal samples and are
    # likely to be the support vectors defining the decision boundary.
    #
    # Once we have these borderline samples, we push them slightly "outward" to cross
    # the decision boundary and become anomalies. The "outward" direction is crucial
    # and is what makes this policy specific to OCSVM. We define it as the direction
    # pointing from the center of the normal data cloud towards the borderline sample.
    # By moving a borderline sample further along this vector, we are effectively
    # pushing it away from the dense region of normal data and just across the
    # learned boundary, thus creating a hard-to-detect anomaly.
    # =================================================================================

    # All package imports must be done inside the function.
    import numpy as np

    if not hasattr(model, 'predict_score'):
        raise TypeError("The provided 'model' object must have a 'predict_score' method.")
        
    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array.")

    # --- Step 1: Identify 'borderline' normal training samples ---
    # We use the provided model to score the original training data.
    # The normal samples with the highest scores are the ones closest to the
    # decision boundary. These are our candidates for transformation.
    train_scores = model.predict_score(X_train)

    # We select a pool of borderline samples, for instance, those in the top 5%
    # of anomaly scores within the training set. This gives us a diverse set
    # of points along the boundary to start from.
    # Using a percentile is more robust than picking a fixed number of points.
    score_threshold = np.percentile(train_scores, 95)
    borderline_indices = np.where(train_scores >= score_threshold)[0]
    
    # Handle the edge case where the percentile might not capture any samples
    # (e.g., if all scores are identical). In this case, we fall back to using
    # the sample with the single highest score.
    if len(borderline_indices) == 0:
        borderline_indices = [np.argmax(train_scores)]
        
    borderline_samples = X_train[borderline_indices]

    # --- Step 2: Determine the 'outward' direction for transformation ---
    # We approximate the center of the normal data distribution by calculating the mean
    # of the training samples. The vector from this center to a borderline sample
    # represents the 'outward' direction.
    data_center = np.mean(X_train, axis=0)

    # --- Step 3: Generate hard anomalies by transforming borderline samples ---
    generated_anomalies = []
    n_borderline = len(borderline_samples)

    for i in range(n_samples):
        # Randomly select a borderline sample to serve as the base for our anomaly.
        # This ensures variety in the generated anomalies.
        base_sample = borderline_samples[i % n_borderline]

        # Calculate the direction vector pointing from the data center to the base sample.
        # Normalizing this vector is good practice to control the step size.
        direction_vector = base_sample - data_center
        norm = np.linalg.norm(direction_vector)
        if norm > 1e-8:  # Avoid division by zero
            direction_vector /= norm

        # Define a small perturbation magnitude to push the sample across the boundary.
        # This magnitude should be small enough to create a "hard" anomaly but large
        # enough to reliably cross the boundary. Scaling it by the data's standard
        # deviation makes it more adaptive to different datasets. We add a small
        # random factor to introduce more variability.
        perturbation_scale = 0.1 * np.mean(np.std(X_train, axis=0))
        perturbation = direction_vector * perturbation_scale * (1 + np.random.uniform(-0.1, 0.1))

        # Create the hard anomaly by adding the perturbation to the base sample.
        hard_anomaly = base_sample + perturbation
        generated_anomalies.append(hard_anomaly)

    return np.array(generated_anomalies)
# ```