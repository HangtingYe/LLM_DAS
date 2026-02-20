import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

class CatBoostWrapper:
    def __init__(self, iterations=200, depth=6, learning_rate=0.1, random_state=None, use_gpu=False, verbose=0):
        """
        A wrapper around CatBoostClassifier with a unified interface.

        Parameters
        ----------
        iterations : int
            Number of boosting iterations.
        depth : int
            Depth of trees.
        learning_rate : float
            Learning rate.
        random_state : int or None
            Random seed.
        use_gpu : bool
            Whether to use GPU.
        verbose : int
            Verbosity level for CatBoost.
        """
        # self.model = CatBoostClassifier(
        #     iterations=iterations,
        #     depth=depth,
        #     learning_rate=learning_rate,
        #     random_seed=random_state,
        #     task_type="GPU" if use_gpu else "CPU",
        #     verbose=verbose,
        #     loss_function="Logloss",
        #     eval_metric="AUC"
        # )
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=random_state,
            task_type="GPU" if use_gpu else "CPU",
            verbose=verbose,
            loss_function="Logloss"  # 注意不要写 AUC
        )
        self.best_params_ = None

    def fit(self, X_train, y_train, cat_features=None):
        """
        Fit CatBoost model.
        """
        self.model.fit(X_train, y_train, cat_features=cat_features)

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

    def tune(self, X_train, y_train, param_grid=None, search_type="random", n_iter=20, cv=3, scoring="roc_auc", random_state=None, n_jobs=-1, cat_features=None):
        """
        Hyperparameter tuning with GridSearchCV or RandomizedSearchCV.
        """
        if param_grid is None:
            param_grid = {
                "depth": [4, 6, 8, 10],
                "iterations": [200, 500, 1000],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "l2_leaf_reg": [1, 3, 5, 7, 10],
                "border_count": [32, 64, 128]
            }

        base_model = CatBoostClassifier(
            random_seed=random_state,
            task_type=self.model.get_param("task_type"),
            verbose=0,
            loss_function="Logloss",
            eval_metric="AUC"
        )

        if search_type == "grid":
            searcher = GridSearchCV(
                base_model,
                param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs
            )
        else:
            searcher = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                random_state=random_state,
                n_jobs=n_jobs
            )

        searcher.fit(X_train, y_train, cat_features=cat_features)
        self.model = searcher.best_estimator_
        self.best_params_ = searcher.best_params_
        return self.best_params_
