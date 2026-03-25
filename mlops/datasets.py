"""Dataset loaders for AutoMLOps demo."""
import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    load_diabetes, fetch_california_housing
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASETS = {
    # ── Classification ────────────────────────────────────────────────────
    "Iris Flowers": {
        "task": "classification",
        "description": "Classic 3-class flower species classification (150 samples, 4 features)",
        "loader": load_iris,
        "icon": "🌸",
        "difficulty": "Easy",
    },
    "Wine Quality": {
        "task": "classification",
        "description": "Wine cultivar identification from chemical analysis (178 samples, 13 features)",
        "loader": load_wine,
        "icon": "🍷",
        "difficulty": "Easy",
    },
    "Breast Cancer": {
        "task": "classification",
        "description": "Tumour malignancy detection (569 samples, 30 features)",
        "loader": load_breast_cancer,
        "icon": "🔬",
        "difficulty": "Medium",
    },
    "Handwritten Digits": {
        "task": "classification",
        "description": "Digit recognition 0-9 from pixel images (1797 samples, 64 features)",
        "loader": load_digits,
        "icon": "✍️",
        "difficulty": "Medium",
    },
    # ── Regression ────────────────────────────────────────────────────────
    "Diabetes Progression": {
        "task": "regression",
        "description": "Disease progression prediction from physiological measurements (442 samples, 10 features)",
        "loader": load_diabetes,
        "icon": "💉",
        "difficulty": "Medium",
    },
    "California Housing": {
        "task": "regression",
        "description": "House price prediction from socio-economic data (20640 samples, 8 features)",
        "loader": fetch_california_housing,
        "icon": "🏠",
        "difficulty": "Hard",
    },
}


def load_dataset(name: str, test_size: float = 0.2, random_state: int = 42):
    """Load a dataset and return train/test splits with metadata."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    cfg = DATASETS[name]
    data = cfg["loader"]()

    X, y = data.data, data.target

    feature_names = (
        list(data.feature_names) if hasattr(data, "feature_names") else
        [f"feature_{i}" for i in range(X.shape[1])]
    )
    target_names = (
        list(data.target_names) if hasattr(data, "target_names") else None
    )

    # Stratify classification splits so each class is proportionally
    # represented in the test set — avoids overly easy / hard partitions.
    stratify = y if cfg["task"] == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    metadata = {
        "name": name,
        "task": cfg["task"],
        "description": cfg["description"],
        "icon": cfg["icon"],
        "difficulty": cfg["difficulty"],
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "feature_names": feature_names,
        "target_names": target_names,
        "n_classes": len(np.unique(y)) if cfg["task"] == "classification" else None,
    }

    return X_train, X_test, y_train, y_test, metadata
