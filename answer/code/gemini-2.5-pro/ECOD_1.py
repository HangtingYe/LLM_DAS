# ```python
def generate_hard_anomalies(n_samples: int, model, X_train: "np.ndarray") -> "np.ndarray":
    """
    Generates anomalies that are hard for an ECOD (Empirical Cumulative 
    Distribution-based Outlier Detection) model to detect.

    This function implements a policy specifically designed to exploit the
    working principle of ECOD. ECOD assesses the anomaly of a sample by
    evaluating how extreme each feature value is in its univariate distribution,
    independently of other features. A hard anomaly for ECOD is therefore a
    sample whose feature values are individually common (i.e., not in the
    tails of their distributions) but whose *combination* of feature values is
    novel or unseen.

    Policy for Generating Hard Anomalies:
    1.  **Identify Prototypical Normals:** First, we identify a pool of the
        'most normal' samples from the training set. These are the samples
        that ECOD assigns the very lowest anomaly scores to. These samples
        consist of feature values that are all very central to their respective
        empirical distributions (i.e., close to the median).
    2.  **Create Anomalies by Recombination:** We then generate new samples
        by creating novel combinations of these 'normal' feature values. For each
        feature of a new anomaly, we randomly pick a value from the
        corresponding feature column of our 'prototypical normals' pool.
        This ensures that every feature value in the generated sample is, by
        itself, extremely common and non-anomalous. However, the resulting
        vector is a new combination of feature values that likely does not
        exist in the training data, thus representing a subtle, combinational
        anomaly.

    This method creates anomalies with low ECOD scores because ECOD's
    feature-independence assumption prevents it from effectively detecting
    anomalous combinations of individually normal values.

    Args:
        n_samples (int): The number of hard anomalies to generate.
        model: A trained ECOD model instance with a `predict_score` method.
        X_train (np.ndarray): The training data used to fit the ECOD model,
                              of shape (n_train_samples, n_features).

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
        
    n_features = X_train.shape[1]

    # --- Step 1: Identify the set of 'prototypical' normal samples ---
    # These are the training samples that ECOD considers the most normal (lowest scores).
    # We use them as a source of "safe" feature values.
    
    # Calculate anomaly scores for the entire training set
    train_scores = model.predict_score(X_train)

    # Define a threshold to select the most normal samples. We'll use the 20th
    # percentile as a robust way to define the pool of "prototypical normals".
    # This avoids being overly sensitive to a few samples with extremely low scores.
    score_threshold = np.percentile(train_scores, 20)
    
    # Filter the training samples to get our pool of prototypical normals
    prototypical_normals_mask = train_scores <= score_threshold
    prototypical_normals = X_train[prototypical_normals_mask]

    if prototypical_normals.shape[0] == 0:
        # Fallback in case no samples are below the threshold, e.g., if all scores are identical.
        # In this rare case, we use the 10 samples with the lowest scores.
        lowest_score_indices = np.argsort(train_scores)[:10]
        prototypical_normals = X_train[lowest_score_indices]
        if prototypical_normals.shape[0] == 0:
             # If X_train is very small, use the whole set
             prototypical_normals = X_train

    # --- Step 2: Generate hard anomalies by recombination ---
    # We construct new samples by picking feature values from the pool of
    # prototypical normals. This creates novel combinations of individually
    # common values.
    
    # Initialize the array to store the generated hard anomalies
    hard_anomalies = np.zeros((n_samples, n_features))
    
    n_prototypical = prototypical_normals.shape[0]

    # For each feature, generate all random choices at once for efficiency
    for j in range(n_features):
        # For the j-th feature, randomly select n_samples indices from the
        # pool of prototypical normals.
        random_indices = np.random.randint(0, n_prototypical, size=n_samples)
        
        # Extract the j-th feature values from the randomly selected prototypical samples
        # and assign them to the j-th column of our new anomalies array.
        hard_anomalies[:, j] = prototypical_normals[random_indices, j]

    return hard_anomalies
# ```