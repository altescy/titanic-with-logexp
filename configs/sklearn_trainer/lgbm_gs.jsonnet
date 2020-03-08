local estimator = {
  "@type": "sklearn_model",
  "@module": "lightgbm",
  "@model": "LGBMClassifier",
  "boosting_type": "gbdt",
  "objective": "binary",
  "subsample_freq": 1,
  "min_split_gain": 0.5,
  "min_child_weight": 1,
  "scale_pos_weight": 1,
};
local param_grid = {
  "learning_rate": [0.005, 0.01],
  "n_estimators": [10, 50, 100],
  "max_depth": [-1, 4, 16],
  "max_bin": [256, 512],
  "num_leaves": [2, 16, 64],
  "random_state" : [100],
  "colsample_bytree" : [0.1, 0.4, 0.7],
  "subsample" : [0.2, 0.4, 0.9],
  "subsample_for_bin": [100, 200],
  "min_child_samples": [1, 5, 10],
  "reg_alpha" : [0.5, 1.0, 1.5],
  "reg_lambda" : [0.5, 1.0, 2.0],
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
