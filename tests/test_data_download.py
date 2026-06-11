"""
@module: tests.test_data_download
@depends: data.download
@exports:
@data_flow: manifest source string -> provider spec
"""

from __future__ import annotations

import pytest

from data.download import parse_source


def test_parse_http_source():
    spec = parse_source("https://example.com/data/file.parquet")

    assert spec.provider == "http"
    assert spec.resource == "https://example.com/data/file.parquet"
    assert spec.file_name == ""


def test_parse_kaggle_dataset_source():
    spec = parse_source("kaggle://datasets/demo-owner/hier-ts/train.parquet")

    assert spec.provider == "kaggle"
    assert spec.resource_type == "datasets"
    assert spec.resource == "demo-owner/hier-ts"
    assert spec.file_name == "train.parquet"


def test_parse_kaggle_competition_source():
    spec = parse_source("kaggle://competitions/m5-forecasting-accuracy/sales_train_validation.csv")

    assert spec.provider == "kaggle"
    assert spec.resource_type == "competitions"
    assert spec.resource == "m5-forecasting-accuracy"
    assert spec.file_name == "sales_train_validation.csv"


def test_parse_source_rejects_invalid_scheme():
    with pytest.raises(ValueError, match="Unsupported dataset source"):
        parse_source("s3://bucket/file.parquet")


def test_parse_source_rejects_invalid_kaggle_shape():
    with pytest.raises(ValueError, match="Kaggle sources must use"):
        parse_source("kaggle://datasets/demo-owner")