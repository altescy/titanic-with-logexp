from __future__ import annotations
import re


def camel_to_snake(s: str) -> str:
    underscored = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', underscored).lower()
