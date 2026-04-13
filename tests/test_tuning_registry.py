"""Sanity checks for the tuning registry.

Ensures the registry entries don't drift out of sync with the
authoritative dataclass defaults over time, and that the schema
is internally consistent.
"""

from dataclasses import fields

import pytest

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.tuning import (
    ParamSpec,
    Sensitivity,
    TuningCategory,
    all_params,
    get_param,
    params_by_category,
    params_by_dataclass,
)

_DATACLASSES = {
    "CTMConfig": CTMConfig,
    "iPEPSConfig": iPEPSConfig,
}


def _dataclass_default(cls, field_name):
    for f in fields(cls):
        if f.name == field_name:
            if f.default_factory is not None and callable(f.default_factory):  # type: ignore[misc]
                return f.default_factory()
            return f.default
    raise LookupError(field_name)


class TestRegistrySchema:
    def test_all_specs_are_param_spec(self):
        for p in all_params():
            assert isinstance(p, ParamSpec)

    def test_fully_qualified_names_unique(self):
        names = [p.fully_qualified_name() for p in all_params()]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_dataclass_names_are_known(self):
        unknown = {
            p.dataclass_name
            for p in all_params()
            if p.dataclass_name not in _DATACLASSES
        }
        assert not unknown, f"Unknown dataclass(es) referenced: {unknown}"


class TestRegistryDefaultsMatchDataclasses:
    @pytest.mark.parametrize(
        "spec", all_params(), ids=lambda p: p.fully_qualified_name()
    )
    def test_default_matches_dataclass(self, spec):
        cls = _DATACLASSES[spec.dataclass_name]
        actual = _dataclass_default(cls, spec.name)
        assert spec.default == actual, (
            f"{spec.fully_qualified_name()}: registry says {spec.default!r}, "
            f"dataclass says {actual!r}"
        )


class TestRegistryQueryAPI:
    def test_get_param_round_trip(self):
        for p in all_params():
            assert get_param(p.fully_qualified_name()) is p

    def test_get_param_missing_returns_none(self):
        assert get_param("CTMConfig.definitely_does_not_exist") is None

    def test_params_by_category_partition(self):
        covered = set()
        for cat in TuningCategory:
            for p in params_by_category(cat):
                covered.add(p.fully_qualified_name())
        all_names = {p.fully_qualified_name() for p in all_params()}
        assert covered == all_names

    def test_params_by_dataclass(self):
        for cls_name in _DATACLASSES:
            ps = params_by_dataclass(cls_name)
            for p in ps:
                assert p.dataclass_name == cls_name


class TestRegistryHintSanity:
    @pytest.mark.parametrize(
        "spec", all_params(), ids=lambda p: p.fully_qualified_name()
    )
    def test_hint_sensitivity_is_enum(self, spec):
        assert isinstance(spec.hint.sensitivity, Sensitivity)

    @pytest.mark.parametrize(
        "spec", all_params(), ids=lambda p: p.fully_qualified_name()
    )
    def test_categorical_has_values(self, spec):
        from tenax.tuning import Scale

        if spec.hint.scale == Scale.CATEGORICAL:
            # Categorical scale should list allowed values — unless the
            # param is a free-form schedule-like structure.
            if spec.hint.values is not None:
                assert len(spec.hint.values) >= 2

    @pytest.mark.parametrize(
        "spec", all_params(), ids=lambda p: p.fully_qualified_name()
    )
    def test_numeric_range_sane(self, spec):
        if spec.hint.range is not None:
            lo, hi = spec.hint.range
            assert lo <= hi
