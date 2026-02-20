import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

class RandomForestWrapper:
    def __init__(self, n_estimators=None, max_depth=None, random_state=None):
        """
        A wrapper around sklearn's RandomForestClassifier
        with a unified interface.
        """
        self.model = RandomForestClassifier(
            # n_estimators=n_estimators,
            # max_depth=max_depth,
            random_state=random_state
        )
        self.best_params_ = None

    def fit(self, X_train, y_train):
        """
        Fit the RandomForest model on training data.
        """
        self.model.fit(X_train, y_train)

    def predict_score(self, X):
        """
        Predict probability of label = 1 for each sample.
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        """
        Predict binary labels.
        """
        return (self.predict_score(X) >= threshold).astype(int)

    def tune(self, X_train, y_train, param_grid=None, search_type="random", n_iter=20, cv=3, scoring="roc_auc", random_state=None, n_jobs=-1):
        """
        Hyperparameter tuning with GridSearchCV or RandomizedSearchCV.

        Parameters
        ----------
        param_grid : dict or None
            Parameter grid for search. If None, use default search space.
        search_type : {"grid", "random"}
            Type of search.
        n_iter : int
            Number of parameter settings sampled (for random search).
        cv : int
            Cross-validation folds.
        scoring : str
            Scoring metric.
        random_state : int or None
            Random seed.
        n_jobs : int
            Parallel jobs.
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "bootstrap": [True, False]
            }

        if search_type == "grid":
            searcher = GridSearchCV(
                self.model,
                param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs
            )
        else: 
            searcher = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                random_state=random_state,
                n_jobs=n_jobs
            )

        searcher.fit(X_train, y_train)
        self.model = searcher.best_estimator_
        self.best_params_ = searcher.best_params_
        return self.best_params_
