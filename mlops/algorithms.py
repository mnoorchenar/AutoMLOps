"""Algorithm registry for AutoMLOps — multiple categories for classification & regression."""
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    PassiveAggressiveClassifier, LinearRegression, Ridge, Lasso,
    ElasticNet, BayesianRidge, HuberRegressor, SGDRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier,
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor,
)
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


# ── Shared verbosity helper ────────────────────────────────────────────────────
_SILENT = {"verbosity": 0}          # XGBoost
_LGBM_SILENT = {"verbose": -1}      # LightGBM


ALGORITHMS = {
    # ══════════════════════════════════════════════════════════════════════
    #  CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════
    "classification": {

        "Linear Models": {
            "Logistic Regression": {
                "class": LogisticRegression,
                "params": {"max_iter": 1000, "random_state": 42},
                "description": "L2-regularised linear classifier, interpretable baseline.",
                "color": "#3b82f6",
            },
            "Logistic Regression (L1)": {
                "class": LogisticRegression,
                "params": {"penalty": "l1", "solver": "saga", "max_iter": 1000, "random_state": 42},
                "description": "Sparse logistic regression via L1 regularisation.",
                "color": "#60a5fa",
            },
            "Ridge Classifier": {
                "class": RidgeClassifier,
                "params": {"alpha": 1.0},
                "description": "Ridge-regression-based classifier; fast on high-dim data.",
                "color": "#93c5fd",
            },
            "SGD Classifier": {
                "class": SGDClassifier,
                "params": {"max_iter": 1000, "random_state": 42},
                "description": "Stochastic Gradient Descent for large-scale linear classification.",
                "color": "#bfdbfe",
            },
            "Passive Aggressive": {
                "class": PassiveAggressiveClassifier,
                "params": {"max_iter": 1000, "random_state": 42},
                "description": "Online learning algorithm suited to text/streaming data.",
                "color": "#dbeafe",
            },
            "Linear Discriminant Analysis": {
                "class": LinearDiscriminantAnalysis,
                "params": {},
                "description": "Finds linear combinations that maximise class separation.",
                "color": "#eff6ff",
            },
        },

        "Tree-Based": {
            "Decision Tree": {
                "class": DecisionTreeClassifier,
                "params": {"max_depth": 10, "random_state": 42},
                "description": "Interpretable tree of if-else rules.",
                "color": "#22c55e",
            },
            "Random Forest": {
                "class": RandomForestClassifier,
                "params": {"n_estimators": 100, "random_state": 42},
                "description": "Bagging of decision trees; robust, low variance.",
                "color": "#4ade80",
            },
            "Extra Trees": {
                "class": ExtraTreesClassifier,
                "params": {"n_estimators": 100, "random_state": 42},
                "description": "Extremely randomised trees; faster than Random Forest.",
                "color": "#86efac",
            },
            "Quadratic Discriminant Analysis": {
                "class": QuadraticDiscriminantAnalysis,
                "params": {},
                "description": "Non-linear discriminant analysis with quadratic boundary.",
                "color": "#bbf7d0",
            },
        },

        "Ensemble / Boosting": {
            "Gradient Boosting": {
                "class": GradientBoostingClassifier,
                "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
                "description": "Sequential boosting of shallow trees; high accuracy.",
                "color": "#f59e0b",
            },
            "AdaBoost": {
                "class": AdaBoostClassifier,
                "params": {"n_estimators": 100, "random_state": 42},
                "description": "Adaptive boosting; up-weights misclassified samples.",
                "color": "#fbbf24",
            },
            "Bagging Classifier": {
                "class": BaggingClassifier,
                "params": {"n_estimators": 50, "random_state": 42},
                "description": "Bootstrap aggregating of any base estimator.",
                "color": "#fcd34d",
            },
            "XGBoost": {
                "class": XGBClassifier,
                "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42, **_SILENT},
                "description": "Optimised gradient boosting with regularisation; competition favourite.",
                "color": "#d97706",
            },
            "LightGBM": {
                "class": LGBMClassifier,
                "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42, **_LGBM_SILENT},
                "description": "Leaf-wise boosting; extremely fast on large datasets.",
                "color": "#b45309",
            },
        },

        "Support Vector Machines": {
            "SVC (RBF Kernel)": {
                "class": SVC,
                "params": {"kernel": "rbf", "probability": True, "random_state": 42},
                "description": "Non-linear SVM with radial basis function kernel.",
                "color": "#a855f7",
            },
            "SVC (Polynomial)": {
                "class": SVC,
                "params": {"kernel": "poly", "degree": 3, "probability": True, "random_state": 42},
                "description": "SVM with polynomial kernel; captures feature interactions.",
                "color": "#c084fc",
            },
            "SVC (Linear)": {
                "class": SVC,
                "params": {"kernel": "linear", "probability": True, "random_state": 42},
                "description": "Linear SVM; interpretable weights, good on text features.",
                "color": "#d8b4fe",
            },
            "LinearSVC": {
                "class": LinearSVC,
                "params": {"max_iter": 2000, "random_state": 42},
                "description": "Faster linear SVM implementation via liblinear.",
                "color": "#ede9fe",
            },
        },

        "Probabilistic": {
            "Gaussian Naive Bayes": {
                "class": GaussianNB,
                "params": {},
                "description": "Assumes Gaussian feature distribution; very fast baseline.",
                "color": "#ec4899",
            },
            "Bernoulli Naive Bayes": {
                "class": BernoulliNB,
                "params": {},
                "description": "NB for binary/boolean features; popular in text classification.",
                "color": "#f472b6",
            },
            "Complement Naive Bayes": {
                "class": ComplementNB,
                "params": {},
                "description": "Improved NB variant, particularly strong on imbalanced text data.",
                "color": "#fbcfe8",
            },
        },

        "Instance-Based (KNN)": {
            "KNN (k=3)": {
                "class": KNeighborsClassifier,
                "params": {"n_neighbors": 3},
                "description": "Majority vote from 3 nearest neighbours.",
                "color": "#06b6d4",
            },
            "KNN (k=5)": {
                "class": KNeighborsClassifier,
                "params": {"n_neighbors": 5},
                "description": "Majority vote from 5 nearest neighbours.",
                "color": "#22d3ee",
            },
            "KNN (k=9)": {
                "class": KNeighborsClassifier,
                "params": {"n_neighbors": 9},
                "description": "Majority vote from 9 nearest neighbours; smoother boundary.",
                "color": "#67e8f9",
            },
        },

        "Neural Networks": {
            "MLP (Small)": {
                "class": MLPClassifier,
                "params": {"hidden_layer_sizes": (64,), "max_iter": 500, "random_state": 42},
                "description": "Single hidden-layer neural network.",
                "color": "#f43f5e",
            },
            "MLP (Medium)": {
                "class": MLPClassifier,
                "params": {"hidden_layer_sizes": (128, 64), "max_iter": 500, "random_state": 42},
                "description": "Two hidden-layer neural network.",
                "color": "#fb7185",
            },
            "MLP (Deep)": {
                "class": MLPClassifier,
                "params": {"hidden_layer_sizes": (256, 128, 64), "max_iter": 500, "random_state": 42},
                "description": "Three hidden-layer neural network with ReLU activations.",
                "color": "#fda4af",
            },
        },
    },

    # ══════════════════════════════════════════════════════════════════════
    #  REGRESSION
    # ══════════════════════════════════════════════════════════════════════
    "regression": {

        "Linear Models": {
            "Linear Regression": {
                "class": LinearRegression,
                "params": {},
                "description": "Ordinary least-squares; interpretable baseline.",
                "color": "#3b82f6",
            },
            "Ridge Regression": {
                "class": Ridge,
                "params": {"alpha": 1.0},
                "description": "L2-regularised linear regression; handles multicollinearity.",
                "color": "#60a5fa",
            },
            "Lasso": {
                "class": Lasso,
                "params": {"alpha": 0.1, "max_iter": 2000},
                "description": "L1 regularisation produces sparse feature weights.",
                "color": "#93c5fd",
            },
            "ElasticNet": {
                "class": ElasticNet,
                "params": {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 2000},
                "description": "Combines L1 and L2 regularisation.",
                "color": "#bfdbfe",
            },
            "Bayesian Ridge": {
                "class": BayesianRidge,
                "params": {},
                "description": "Probabilistic Bayesian linear regression with automatic regularisation.",
                "color": "#dbeafe",
            },
            "Huber Regressor": {
                "class": HuberRegressor,
                "params": {"max_iter": 200},
                "description": "Robust to outliers via Huber loss function.",
                "color": "#eff6ff",
            },
        },

        "Tree-Based": {
            "Decision Tree Regressor": {
                "class": DecisionTreeRegressor,
                "params": {"max_depth": 10, "random_state": 42},
                "description": "Recursive partitioning for regression.",
                "color": "#22c55e",
            },
            "Random Forest Regressor": {
                "class": RandomForestRegressor,
                "params": {"n_estimators": 100, "random_state": 42},
                "description": "Averaged predictions of many trees; low variance.",
                "color": "#4ade80",
            },
            "Extra Trees Regressor": {
                "class": ExtraTreesRegressor,
                "params": {"n_estimators": 100, "random_state": 42},
                "description": "Extremely randomised regression trees; fast.",
                "color": "#86efac",
            },
        },

        "Ensemble / Boosting": {
            "Gradient Boosting Regressor": {
                "class": GradientBoostingRegressor,
                "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
                "description": "Sequential boosting minimising regression loss.",
                "color": "#f59e0b",
            },
            "AdaBoost Regressor": {
                "class": AdaBoostRegressor,
                "params": {"n_estimators": 100, "random_state": 42},
                "description": "Adaptive boosting for regression.",
                "color": "#fbbf24",
            },
            "Bagging Regressor": {
                "class": BaggingRegressor,
                "params": {"n_estimators": 50, "random_state": 42},
                "description": "Bootstrap aggregating for regression.",
                "color": "#fcd34d",
            },
            "XGBoost Regressor": {
                "class": XGBRegressor,
                "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42, **_SILENT},
                "description": "Regularised gradient boosting; excellent out-of-the-box performance.",
                "color": "#d97706",
            },
            "LightGBM Regressor": {
                "class": LGBMRegressor,
                "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42, **_LGBM_SILENT},
                "description": "Leaf-wise boosting regressor; fast and memory-efficient.",
                "color": "#b45309",
            },
        },

        "Support Vector Machines": {
            "SVR (RBF)": {
                "class": SVR,
                "params": {"kernel": "rbf"},
                "description": "Non-linear support vector regression.",
                "color": "#a855f7",
            },
            "SVR (Linear)": {
                "class": SVR,
                "params": {"kernel": "linear"},
                "description": "Linear support vector regression.",
                "color": "#c084fc",
            },
        },

        "Instance-Based (KNN)": {
            "KNN Regressor (k=3)": {
                "class": KNeighborsRegressor,
                "params": {"n_neighbors": 3},
                "description": "Average of 3 nearest neighbours.",
                "color": "#06b6d4",
            },
            "KNN Regressor (k=5)": {
                "class": KNeighborsRegressor,
                "params": {"n_neighbors": 5},
                "description": "Average of 5 nearest neighbours.",
                "color": "#22d3ee",
            },
        },

        "Neural Networks": {
            "MLP Regressor (Small)": {
                "class": MLPRegressor,
                "params": {"hidden_layer_sizes": (64,), "max_iter": 500, "random_state": 42},
                "description": "Single hidden-layer neural network for regression.",
                "color": "#f43f5e",
            },
            "MLP Regressor (Medium)": {
                "class": MLPRegressor,
                "params": {"hidden_layer_sizes": (128, 64), "max_iter": 500, "random_state": 42},
                "description": "Two hidden-layer neural network for regression.",
                "color": "#fb7185",
            },
        },
    },
}


# ── Hyperparameter search grids (keyed by model class name) ───────────────────
HPO_GRIDS: dict[str, dict] = {
    # Linear Models
    "LogisticRegression":    {"C": [0.001, 0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "saga"], "max_iter": [500, 1000]},
    "RidgeClassifier":       {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "SGDClassifier":         {"loss": ["hinge", "log_loss", "modified_huber"], "alpha": [0.0001, 0.001, 0.01]},
    "Ridge":                 {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso":                 {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "ElasticNet":            {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "HuberRegressor":        {"epsilon": [1.1, 1.35, 1.5, 2.0], "alpha": [0.0001, 0.001, 0.01, 0.1]},
    # Tree-Based
    "DecisionTreeClassifier":{"max_depth": [3, 5, 7, 10, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "criterion": ["gini", "entropy"]},
    "DecisionTreeRegressor": {"max_depth": [3, 5, 7, 10, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]},
    "RandomForestClassifier":{"n_estimators": [50, 100, 200, 300], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10], "max_features": ["sqrt", "log2"]},
    "RandomForestRegressor": {"n_estimators": [50, 100, 200, 300], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10], "max_features": ["sqrt", "log2", None]},
    "ExtraTreesClassifier":  {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
    "ExtraTreesRegressor":   {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
    # Boosting
    "GradientBoostingClassifier": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 4, 5, 6], "subsample": [0.7, 0.8, 0.9, 1.0]},
    "GradientBoostingRegressor":  {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 4, 5, 6], "subsample": [0.7, 0.8, 0.9, 1.0]},
    "AdaBoostClassifier":  {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0]},
    "AdaBoostRegressor":   {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0], "loss": ["linear", "square", "exponential"]},
    "XGBClassifier":  {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 4, 5, 6, 7], "subsample": [0.7, 0.8, 0.9], "colsample_bytree": [0.7, 0.8, 0.9]},
    "XGBRegressor":   {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 4, 5, 6, 7], "subsample": [0.7, 0.8, 0.9], "colsample_bytree": [0.7, 0.8, 0.9]},
    "LGBMClassifier": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [-1, 5, 10, 20], "num_leaves": [15, 31, 63, 127], "subsample": [0.7, 0.8, 0.9, 1.0]},
    "LGBMRegressor":  {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [-1, 5, 10, 20], "num_leaves": [15, 31, 63, 127], "subsample": [0.7, 0.8, 0.9, 1.0]},
    # SVM
    "SVC": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.001, 0.01, 0.1]},
    "SVR": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"], "epsilon": [0.01, 0.1, 0.5, 1.0]},
    # KNN
    "KNeighborsClassifier": {"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
    "KNeighborsRegressor":  {"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
    # MLP
    "MLPClassifier": {"hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (256, 128)], "learning_rate_init": [0.001, 0.005, 0.01], "alpha": [0.0001, 0.001, 0.01], "activation": ["relu", "tanh"]},
    "MLPRegressor":  {"hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (256, 128)], "learning_rate_init": [0.001, 0.005, 0.01], "alpha": [0.0001, 0.001, 0.01], "activation": ["relu", "tanh"]},
}


def get_hpo_grid(cls) -> dict:
    """Return the hyperparameter search grid for a model class, or {} if none defined."""
    return HPO_GRIDS.get(cls.__name__, {})


def get_algorithm(task: str, category: str, name: str) -> dict:
    """Retrieve algorithm config by task / category / name."""
    try:
        return ALGORITHMS[task][category][name]
    except KeyError:
        raise ValueError(f"Algorithm not found: task={task}, category={category}, name={name}")


def list_algorithms(task: str) -> dict:
    """Return the algorithm tree for the given task type."""
    if task not in ALGORITHMS:
        raise ValueError(f"Unknown task: {task}")
    return ALGORITHMS[task]


def all_algorithm_names(task: str) -> list[str]:
    """Flat list of all algorithm names for a given task."""
    names = []
    for cat in ALGORITHMS[task].values():
        names.extend(cat.keys())
    return names


def algorithms_for_json(task: str | None = None) -> dict:
    """Return ALGORITHMS (or a task subset) as a JSON-serializable dict.

    Removes the non-serializable ``"class"`` key and converts tuples to lists.
    """
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k != "class"}
        if isinstance(obj, (list, tuple)):
            return [_clean(i) for i in obj]
        return obj

    src = ALGORITHMS if task is None else ALGORITHMS[task]
    return _clean(src)
