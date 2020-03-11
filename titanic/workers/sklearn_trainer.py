from __future__ import annotations
import json
import pickle

import colt
import logexp
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection._search import BaseSearchCV

import titanic
from titanic.logger import create_logger

logger = create_logger(__name__)


@titanic.ex.worker("sklearn_trainer")
class SklearnTrainer(logexp.BaseWorker):
    def config(self):
        self.random_seed = 0

        self.train_path = "./data/train.csv"
        self.test_path = "./data/test.csv"

        self.pdpipeline = {
            "@type":
            "pdp:pd_pipeline",
            "stages": [
                {
                    "@type": "pdp:fill_na",
                    "columns": ["Age", "Fare"],
                    "fill_type": "median"
                },
                {
                    "@type": "pdp:fill_na",
                    "columns": ["Embarked"],
                    "fill_type": "mode"
                },
                {
                    "@type": "pdp:col_drop",
                    "columns": ["PassengerId", "Cabin", "Name", "Ticket"]
                },
                {
                    "@type": "pdp:encode"
                },
            ],
        }

        self.model = {
            "@type": "sklearn_model",
            "@model": "ensemble.RandomForestClassifier",
        }

        self.cross_validate = {
            "cv": 5,
            "scoring": {
                "accuracy": "accuracy",
                "precision": "precision_macro",
                "recall": "recall_macro",
                "fscore": "f1_macro",
            },
            "return_train_score": True,
        }

    def run(self) -> logexp.Report:
        logger.info("params: %s", repr(self.params.to_json()))

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        logger.info("load datasets")
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        logger.info("datasets:")
        logger.info("%s:\n%s", self.train_path, train_df.info())
        logger.info("%s:\n%s", self.test_path, test_df.info())

        logger.info("build ndarray")
        pdpipeline = colt.build(self.pdpipeline)
        pdpipeline.fit(train_df)

        y_train = train_df.pop("Survived").to_numpy(dtype=np.float)
        X_train = pdpipeline.transform(train_df).to_numpy(dtype=np.float)
        X_test = pdpipeline.transform(test_df).to_numpy(dtype=np.float)

        logger.info("build model")
        model = colt.build(self.model)
        if isinstance(model, BaseSearchCV):
            grid = model
            logger.info("[ GS ] start grid-search")
            grid.fit(X_train, y_train)

            logger.info("[ GS ] best params: %s", repr(grid.best_params_))
            logger.info("[ GS ] best score: %s", repr(grid.best_score_))

            with self.storage.open("best_params.json", "w") as f:
                json.dump(grid.best_params_, f)

            model = grid.best_estimator_

        logger.info("model: %s", repr(model))

        logger.info("start cross-validation: %s", repr(self.cross_validate))
        cv_scores = cross_validate(model, X_train, y_train,
                                   **self.cross_validate)
        cv_score_mean = {key: val.mean() for key, val in cv_scores.items()}
        cv_score_std = {key: val.std() for key, val in cv_scores.items()}
        for key in cv_scores:
            mean = cv_score_mean[key]
            std = cv_score_std[key]
            logger.info("[ CV ]  %s : %f +/- %f", key, mean, std)

        logger.info("start training model")
        model.fit(X_train, y_train)

        logger.info("save model")
        with self.storage.open("model.pkl", "wb") as f:
            pickle.dump(model, f)

        logger.info("make predictions")
        test_pred_df = pd.DataFrame()
        test_pred_df["PassengerId"] = test_df["PassengerId"]
        test_pred_df["Survived"] = model.predict(X_test).astype(int)

        logger.info("save predictions")
        with self.storage.open("submit.csv", "w") as f:
            test_pred_df.to_csv(f, index=False)

        report = logexp.Report()
        report["cv_score"] = {
            "mean": cv_score_mean,
            "std": cv_score_std,
            "all": {key: val.tolist()
                    for key, val in cv_scores.items()},
        }

        return report
