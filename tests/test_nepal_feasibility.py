"""Offline safety checks for bounded pilot requests; no real coverage claims."""

import json

import pytest

from landcover.nepal_feasibility import (
    candidate_geometry,
    daily_limit,
    pilot_budget,
    scene_inventory,
)


def quota(value, dimensions=None):
    return {
        "metrics": [
            {
                "metric": "earthengine.googleapis.com/daily_eecu_usage_time",
                "consumerQuotaLimits": [
                    {
                        "quotaBuckets": [
                            {"effectiveLimit": str(value), "dimensions": dimensions or {}}
                        ]
                    }
                ],
            }
        ]
    }


@pytest.mark.parametrize("value", [0, -1])
def test_reject_exhausted_daily_quota(value):
    with pytest.raises(ValueError, match="daily EECU quota"):
        daily_limit(quota(value))


def test_positive_cap_and_incomplete_snapshot():
    assert daily_limit(quota(3600)) == 3600
    assert daily_limit(quota(9223372036854775807)) is None
    assert daily_limit(quota(7200)) == 7200
    with pytest.raises(ValueError, match="incomplete"):
        daily_limit({**quota(3600), "nextPageToken": "more"})
    with pytest.raises(ValueError, match="daily EECU quota"):
        daily_limit(quota(3600, {"region": "not-global"}))


def test_pilot_area_and_bounding_pixel_limits():
    box = [[0, 0], [6000, 0], [6000, 6000], [0, 6000], [0, 0]]
    assert pilot_budget(36_000_000, box) == 10201
    assert pilot_budget(200_000_000, [[0, 0], [20000, 40000]], True) == 223780
    with pytest.raises(ValueError, match="invalid corridor area"):
        pilot_budget(float("nan"), box, True)
    with pytest.raises(ValueError, match="50 square"):
        pilot_budget(51_000_000, box)
    with pytest.raises(ValueError, match="30000"):
        pilot_budget(1_000_000, [[0, 0], [200_000, 200_000]])


def test_geometry_requires_finite_sourced_coordinate_shape(tmp_path):
    path = tmp_path / "candidate.geojson"
    geometry = {"type": "LineString", "coordinates": [[85, 28], [85.01, 28.01]]}
    path.write_text(json.dumps({"type": "Feature", "geometry": geometry}))
    assert candidate_geometry(path) == geometry
    geometry["coordinates"][1][0] = float("nan")
    path.write_text(json.dumps(geometry))
    with pytest.raises(ValueError, match="invalid"):
        candidate_geometry(path)


def test_scene_inventory_requires_matching_quality_and_bounded_size():
    record = dict(
        ids=["a", "b"], times=[0, 1000], tiles=["t", "t"], cloud_percent=[0, 50], score_ids=["a"]
    )
    result = scene_inventory(record)
    assert result[0]["cloud_score_available"] is True
    assert result[1]["cloud_score_available"] is False
    assert result[1]["acquired_at_utc"] == "1970-01-01T00:00:01+00:00"
    with pytest.raises(ValueError, match="matching Cloud"):
        scene_inventory({**record, "score_ids": []})
    with pytest.raises(ValueError, match="oversized"):
        scene_inventory({**record, "ids": list(map(str, range(61)))})


def test_shared_download_grid_and_google_size_limit():
    from landcover.nepal_feasibility import download_grid

    grid = download_grid([[300001, 3100001], [305001, 3095001]])
    assert grid == {"width": 501, "height": 501, "transform": [10, 0, 300000, 0, -10, 3100010]}
    with pytest.raises(ValueError, match="Google's synchronous"):
        download_grid([[0, 0], [10000, 10000]])


def test_yaml_fallback_dates_remain_serializable(tmp_path):
    from datetime import date

    from landcover.nepal import validate_acquisition

    result = validate_acquisition(
        {
            "event_name": "test",
            "event_date": date(2026, 8, 26),
            "event_source": "https://example.org/event",
            "source_checked_on": date(2026, 10, 8),
            "project": "test",
            "bbox": [85, 27, 86, 29],
            "pre_start": date(2026, 8, 1),
            "pre_end": date(2026, 8, 26),
            "post_start": date(2026, 8, 27),
            "post_end": date(2026, 9, 21),
            "fallback_post_end": date(2026, 10, 8),
            "collection": "COPERNICUS/S2_HARMONIZED",
            "crs": "EPSG:32645",
        }
    )
    assert result["fallback_post_end"] == "2026-10-08"
    assert json.loads(json.dumps(result))["event_date"] == "2026-08-26"


def test_offline_helpers_import_without_optional_earth_engine_dependencies(monkeypatch):
    import builtins
    import importlib.util

    from landcover import nepal_feasibility

    original_import = builtins.__import__

    def without_nepal_dependencies(name, *args, **kwargs):
        if name == "ee" or name == "google" or name.startswith("google."):
            raise ModuleNotFoundError("optional Nepal dependencies are unavailable")
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(
        "offline_nepal_helpers", nepal_feasibility.__file__
    )
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", without_nepal_dependencies)
        spec.loader.exec_module(module)
        assert module.pilot_budget(36000000, [[0, 0], [6000, 6000]]) == 10201
