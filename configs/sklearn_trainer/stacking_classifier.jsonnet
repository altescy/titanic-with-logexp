local random_forest = {
  "@type": "sklearn_model",
  "@model": "ensemble.RandomForestClassifier",
  "class_weight": null,
  "criterion": "gini",
  "max_depth": null,
  "max_features": "sqrt",
  "max_leaf_nodes": 32,
  "min_samples_leaf": 4,
  "min_samples_split": 2,
  "n_estimators": 100,
  "random_state": 10,
};
local lgbm = {
  "@type": "sklearn_model",
  "@module": "lightgbm",
  "@model": "LGBMClassifier",
  "boosting_type": "gbdt",
  "colsample_bytree": 0.7,
  "learning_rate": 0.01,
  "max_bin": 256,
  "max_depth": 4,
  "min_child_samples": 10,
  "min_child_weight": 1,
  "min_split_gain": 0.5,
  "n_estimators": 100,
  "objective": "binary",
  "num_leaves": 16,
  "random_state": 100,
  "reg_alpha": 0.5,
  "reg_lambda": 0.5,
  "scale_pos_weight": 1,
  "subsample": 0.9,
  "subsample_for_bin": 200,
  "subsample_freq": 1,
};
local svc = {
  "@type": "sklearn_model",
  "@model": "svm.SVC",
  "C": 20,
  "class_weight": null,
  "gamma": "scale",
  "kernel": "rbf",
  "probability": true,
  "random_state": 10
};

local pdpipeline = import 'pdpipeline.jsonnet';
{
  "random_seed": 0,
  "train_path": "./data/train.csv",
  "test_path": "./data/test.csv",
  "pdpipeline": pdpipeline,
  "model": {
    "@type": "sklearn_model",
    "@model": "ensemble.StackingClassifier",
    "estimators": [
      ["lgbm", lgbm],
      ["rfc", random_forest],
      ["svc", svc],
    ],
    "final_estimator": {
      "@type": "sklearn_model",
      "@model": "linear_model.LogisticRegression",
      "C": 5,
    },
    "cv": 5,
    "passthrough": false,
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
