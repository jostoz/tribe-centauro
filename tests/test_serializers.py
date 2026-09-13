"""Tests de serialización JSON-safe y de la taxonomía de errores."""

import json
import math

import numpy as np
import pytest

from service.errors import InvalidInput, JobNotFound, MediaUnsupported
from service.serializers import downsample, to_jsonable


def test_numpy_scalars_become_native():
    payload = {
        "i": np.int64(3),
        "f": np.float64(1.5),
        "b": np.bool_(True),
        "arr": np.arange(5),
    }
    out = to_jsonable(payload)

    assert type(out["i"]) is int
    assert type(out["f"]) is float
    assert type(out["b"]) is bool
    assert out["arr"] == [0, 1, 2, 3, 4]

    json.dumps(out)  # no debe lanzar


def test_non_finite_becomes_null():
    out = to_jsonable({"nan": float("nan"), "inf": float("inf")})
    assert out["nan"] is None
    assert out["inf"] is None
    json.dumps(out)


def test_long_sequences_are_truncated():
    out = to_jsonable(np.arange(10_000))
    assert len(out) == 2048


def test_downsample_reduces_to_requested_points():
    series = np.arange(600, dtype=float)
    out = downsample(series, max_points=120)
    assert len(out) == 120


def test_downsample_preserves_shape_of_trend():
    series = np.linspace(0, 100, 500)
    out = downsample(series, max_points=50)
    assert out[0] < out[-1], "una tendencia monótona debe seguir siendo monótona"
    assert out[0] == pytest.approx(0, abs=5)


def test_downsample_passthrough_for_short_series():
    out = downsample([1.0, 2.0, 3.0], max_points=120)
    assert out == [1.0, 2.0, 3.0]


def test_error_payload_is_machine_actionable():
    err = InvalidInput("fuente ambigua", hint="usa una sola fuente")
    payload = err.as_dict()

    assert payload["code"] == "INVALID_INPUT"
    assert payload["retryable"] is False
    assert payload["hint"]
    json.dumps(payload)


def test_job_not_found_is_not_retryable():
    assert JobNotFound("j1").as_dict()["retryable"] is False


def test_media_unsupported_carries_code():
    assert MediaUnsupported("formato raro").as_dict()["code"] == "MEDIA_UNSUPPORTED"
