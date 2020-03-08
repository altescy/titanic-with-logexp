import logexp

from titanic.utils.jsonnet import load_params_from_jsonnet


# setup to use Jsonnet for params
setattr(
    logexp.executor.Executor,
    "_load_params",
    lambda _self, path: load_params_from_jsonnet(path)
)


# setup logexp experiment
ex = logexp.Experiment("titanic")
