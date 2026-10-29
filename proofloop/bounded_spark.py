"""Versioned low-reasoning default around the existing journaled Meta adapter.

Only an omitted Spark reasoning keyword changes. An explicit None preserves
the provider default; explicit low, medium or high remain caller choices. The
caller still controls its existing output-token cap. SAM requests, admission,
lease/binding guards, attempt accounting and no-retry behavior are inherited.
This configuration does not establish a quality or accuracy improvement.
"""
from __future__ import annotations

from typing import Any

from .inference_journal import JournaledMetaClient


BOUNDED_SPARK_VERSION = "bounded_spark_low_reasoning_v1"
DEFAULT_REASONING_EFFORT = "low"


class BoundedSparkClient(JournaledMetaClient):
    """Use explicit low reasoning only when spark_json's keyword is absent."""

    def spark_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = DEFAULT_REASONING_EFFORT
        return super().spark_json(*args, **kwargs)
