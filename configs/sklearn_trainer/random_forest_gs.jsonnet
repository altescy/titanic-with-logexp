local estimator = {
  "@type": "sklearn_model",
  "@model": "ensemble.RandomForestClassifier"
  };
local param_grid = {
  "n_estimators": [10, 50, 100],
  "criterion": ["gini", "entropy"],
  "max_depth": [null, 16, 32],
  "min_samples_split": [2, 4],
  "min_samples_leaf": [1, 4],
  "max_features": ["sqrt", "log2"],
  "max_leaf_nodes": [null, 32],
  "random_state": [10, 100],
  "class_weight": [null, "balanced", "balanced_subsample"],
};

local pdpipeline = import 'pdpipeline.jsonnet';
{
  "random_seed": 0,
  "train_path": "./data/train.csv",
  "test_path": "./data/test.csv",
  "pdpipeline": pdpipeline,
  "model": {
    "@type": "sklearn_model",
    "@model": "model_selection.GridSearchCV",
    "estimator": estimator,
    "param_grid": param_grid,
    "n_jobs": -1,
    "verbose": 1,
  },
  "cross_validate": {
    "cv": 5,
    "scoring": {
      "accuracy": "accuracy",
      "precision": "precision_macro",
      "recall": "recall_macro",
      "fscore": "f1_macro"
    },
    "return_train_score": true
  }
}
