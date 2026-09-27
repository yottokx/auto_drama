import pytest

from scripts.record.record_player import player_url, successor


def test_successor_checks_lineage_and_order():
    ready = {"build_id": "first", "status": "ready",
             "next_build": {"id": "second", "chapter_number": 2}}
    assert successor(ready, "first", 1) == "second"
    with pytest.raises(ValueError):
        successor(ready, "other", 1)
    with pytest.raises(ValueError):
        successor(ready, "first", 2)
    for identifier in ["../evil", "first", "", "日本語"]:
        with pytest.raises(ValueError):
            successor({**ready, "next_build": {"id": identifier, "chapter_number": 2}}, "first", 1)
    for status in ["waiting", "complete"]:
        assert successor({"build_id": "first", "status": status}, "first", 1) is None


def test_player_url_rejects_non_http():
    assert player_url("http://127.0.0.1:8000/player/build/").endswith("/build/")
    for url in ["file:///tmp/index.html", "invalid", "http://localhost/?unexpected=1"]:
        with pytest.raises(ValueError):
            player_url(url)
