local pdpipeline = import 'pdpipeline.jsonnet';
{
  "random_seed": 0,
  "train_path": "./data/train.csv",
  "test_path": "./data/test.csv",
  "pdpipeline": pdpipeline,
  "model": {
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
