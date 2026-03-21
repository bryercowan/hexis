"""Tests for TrajectoryStore: CRUD, dedup, import, export."""

import base64
import json

import pytest

from hexis.data.schemas import TrajectoryRecord
from hexis.data.trajectory_store import TrajectoryStore


def _make_record(label: str = "test", content: str = "hello") -> TrajectoryRecord:
    b64 = base64.b64encode(content.encode()).decode()
    return TrajectoryRecord(
        screenshot_b64=b64,
        action={"action": "left_click", "coordinate": [100, 200]},
        expert_label=label,
    )


@pytest.fixture
def store(tmp_path):
    return TrajectoryStore(base_dir=tmp_path / "trajectories")


class TestTrajectoryStore:
    def test_add_returns_true(self, store):
        rec = _make_record()
        assert store.add(rec) is True

    def test_add_duplicate_returns_false(self, store):
        rec = _make_record()
        assert store.add(rec) is True
        assert store.add(rec) is False

    def test_count(self, store):
        assert store.count("test") == 0
        store.add(_make_record("test", "a"))
        store.add(_make_record("test", "b"))
        assert store.count("test") == 2

    def test_count_total(self, store):
        store.add(_make_record("a", "x"))
        store.add(_make_record("b", "y"))
        assert store.count() == 2

    def test_labels(self, store):
        store.add(_make_record("alpha", "1"))
        store.add(_make_record("beta", "2"))
        labels = store.labels()
        assert "alpha" in labels
        assert "beta" in labels

    def test_query(self, store):
        store.add(_make_record("test", "data1"))
        store.add(_make_record("test", "data2"))
        records = store.query("test")
        assert len(records) == 2
        assert all(isinstance(r, TrajectoryRecord) for r in records)

    def test_query_empty(self, store):
        assert store.query("nonexistent") == []

    def test_add_batch(self, store):
        records = [_make_record("batch", f"item{i}") for i in range(5)]
        added = store.add_batch(records)
        assert added == 5
        assert store.count("batch") == 5

    def test_add_batch_dedup(self, store):
        records = [_make_record("batch", "same")] * 3
        added = store.add_batch(records)
        assert added == 1

    def test_train_val_split(self, store):
        for i in range(20):
            store.add(_make_record("split", f"sample{i}"))
        train, val = store.train_val_split("split", val_fraction=0.2)
        assert len(train) + len(val) == 20
        assert len(val) >= 1

    def test_train_val_split_empty(self, store):
        train, val = store.train_val_split("empty")
        assert train == []
        assert val == []

    def test_export_for_sft(self, store, tmp_path):
        store.add(_make_record("expert", "data1"))
        store.add(_make_record("expert", "data2"))
        out_path = tmp_path / "sft_export.jsonl"
        store.export_for_sft("expert", out_path, "do something")
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2
        entry = json.loads(lines[0])
        assert "screenshot_b64" in entry
        assert entry["conditioning_text"] == "do something"

    def test_export_for_router(self, store, tmp_path):
        store.add(_make_record("expert_a", "d1"))
        store.add(_make_record("none", "d2"))
        out_path = tmp_path / "router_export.jsonl"
        store.export_for_router(out_path)
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2
        entries = [json.loads(l) for l in lines]
        names = [e["expert_name"] for e in entries]
        assert "expert_a" in names
        assert "__none__" in names

    def test_import_jsonl(self, store, tmp_path):
        jsonl = tmp_path / "import.jsonl"
        records = []
        for i in range(3):
            b64 = base64.b64encode(f"import{i}".encode()).decode()
            records.append(json.dumps({
                "screenshot_b64": b64,
                "action": {"click": [i, i]},
            }))
        jsonl.write_text("\n".join(records))
        added = store.import_jsonl(jsonl, "imported")
        assert added == 3
        assert store.count("imported") == 3

    def test_stats(self, store):
        store.add(_make_record("a", "x"))
        store.add(_make_record("b", "y"))
        store.add(_make_record("b", "z"))
        stats = store.stats()
        assert stats["a"] == 1
        assert stats["b"] == 2

    def test_persistence(self, tmp_path):
        """Store data persists across instances."""
        store1 = TrajectoryStore(base_dir=tmp_path / "persist")
        store1.add(_make_record("test", "persist"))

        store2 = TrajectoryStore(base_dir=tmp_path / "persist")
        assert store2.count("test") == 1
        # Dedup persists too
        assert store2.add(_make_record("test", "persist")) is False
