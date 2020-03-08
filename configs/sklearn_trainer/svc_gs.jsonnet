local estimator = {
  "@type": "sklearn_model",
  "@model": "svm.SVC"
  };
local param_grid = {
  "C": [0.5, 1.0, 5.0],
  "class_weight": [null, "balanced"],
  "gamma": ["scale", "auto"],
  "kernel": ["rbf"],
  "random_state": [10, 100],
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
