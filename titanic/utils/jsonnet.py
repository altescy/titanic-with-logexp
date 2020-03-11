import typing as tp

import json
from logexp.params import Params
import os
from pathlib import Path

from _jsonnet import evaluate_file


def _is_encodable(value: str) -> bool:
    return (value == "") or (value.encode("utf-8", "ignore") != b"")


def _environment_variables() -> tp.Dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items() if _is_encodable(value)
    }


def load_params_from_jsonnet(path: str) -> Params:
    ext_vars = _environment_variables()
    jsondict = json.loads(evaluate_file(str(path), ext_vars=ext_vars))
    return Params.from_json(jsondict)
