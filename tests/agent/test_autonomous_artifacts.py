from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from app.agent.artifacts import AutonomousArtifactStore
from app.agent import artifacts
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository


ACCOUNT = 900000101
GROUP = 900000001


def source(engine, *, group=GROUP, name="source"):
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=group, group_name="synthetic", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=900000201, nickname="synthetic", group_card="")
        row = MessageRepository(session).add_group_message(platform_msg_id=name, group_id=group, user_id=900000201,
            timestamp=datetime.now(UTC), plain_text="合成来源", raw_json={}, msg_type="text", reply_to_msg_id=None, mentioned_bot=False)
        session.flush()
        return row.id


def store(tmp_path, engine, *, sources=(), scope=None, account=ACCOUNT):
    return AutonomousArtifactStore(tmp_path / "artifacts", account_id=account, scope=scope or f"group:{GROUP}",
                                   engine=engine, source_message_ids=lambda: sources)


def photo(root, name="provider.png"):
    root.mkdir(parents=True, exist_ok=True)
    output = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(output, "PNG")
    path = root / name
    path.write_bytes(output.getvalue())
    return path


def test_actual_image_manifest_and_new_store_resume_same_owned_artifact(sqlite_engine, tmp_path):
    origin = source(sqlite_engine)
    library = store(tmp_path, sqlite_engine, sources=(origin,))
    original = photo(library.root)
    artifact_id = library.add(original, origin="autonomous_generated", model="synthetic-image-model")
    actual = library.get(artifact_id)
    assert actual is not None and actual != original and actual.read_bytes() == original.read_bytes()
    original.unlink()
    resumed = store(tmp_path, sqlite_engine)
    assert resumed.get(artifact_id) == actual
    listing = resumed.list()
    assert len(listing) == 1 and listing[0]["id"] == artifact_id
    assert listing[0]["type"] == "image/png"
    assert set(listing[0]) == {"id", "origin", "type", "size", "hash"}
    metadata = json.loads(next(library.root.glob(artifact_id + ".meta.json")).read_text(encoding="utf-8"))
    assert metadata["account_id"] == ACCOUNT and metadata["scope"] == f"group:{GROUP}"
    assert metadata["source_message_ids"] == [origin]
    assert metadata["sha256"] == listing[0]["hash"]
    assert resumed.describe(artifact_id) == listing[0]
    assert resumed.describe("art-missing") is None


@pytest.mark.parametrize("suffix,body,mime", [
    (".txt", "实际计算结果是42", "text/plain"), (".csv", "x,y\n1,2", "text/csv"),
    (".json", '{"result":42}', "application/json"), (".md", "# 合成结果", "text/markdown"),
])
def test_computation_text_outputs_can_be_owned_without_exposing_paths(sqlite_engine, tmp_path, suffix, body, mime):
    library = store(tmp_path, sqlite_engine)
    sandbox = library.root / "sandbox" / "synthetic-run"
    sandbox.mkdir(parents=True)
    path = sandbox / ("result" + suffix)
    path.write_text(body, encoding="utf-8")
    artifact_id = library.add(path, origin="sandbox_computed")
    actual = library.get(artifact_id)
    assert actual.read_text(encoding="utf-8") == body
    assert library.list()[0]["type"] == mime
    assert "synthetic-run" not in json.dumps(library.list()) and body not in json.dumps(library.list())


def test_account_and_scope_cannot_reuse_each_others_artifacts(sqlite_engine, tmp_path):
    library = store(tmp_path, sqlite_engine)
    artifact_id = library.add(photo(library.root), origin="generated")
    assert store(tmp_path, sqlite_engine, account=ACCOUNT + 1).get(artifact_id) is None
    assert store(tmp_path, sqlite_engine, scope=f"group:{GROUP + 1}").get(artifact_id) is None
    assert store(tmp_path, sqlite_engine, scope="qzone_publish").get(artifact_id) is None
    assert store(tmp_path, sqlite_engine, scope="qzone_publish").list() == ()


@pytest.mark.parametrize("state", ["deleted", "uncertain", "blocked", "reserved"])
def test_source_withdrawal_revokes_durable_artifact_reads(sqlite_engine, tmp_path, state):
    source_id = source(sqlite_engine)
    library = store(tmp_path, sqlite_engine, sources=(source_id,))
    artifact_id = library.add(photo(library.root), origin="generated")
    with session_scope(sqlite_engine) as session:
        row = session.get(Message, source_id)
        row.raw_json = {"delivery_state": state}
    assert library.get(artifact_id) is None and library.list() == ()


def test_public_scopes_never_accept_group_derived_images(sqlite_engine, tmp_path):
    group_source = source(sqlite_engine)
    library = store(tmp_path, sqlite_engine, sources=(group_source,), scope="qzone_publish")
    with pytest.raises(ValueError):
        library.add(photo(library.root), origin="generated")


def test_group_cannot_claim_other_groups_canonical_sources(sqlite_engine, tmp_path):
    other_source = source(sqlite_engine, group=GROUP + 1)
    library = store(tmp_path, sqlite_engine, sources=(other_source,))
    with pytest.raises(ValueError):
        library.add(photo(library.root), origin="generated")


def test_hash_tampering_and_manifest_path_tampering_fail_closed(sqlite_engine, tmp_path):
    library = store(tmp_path, sqlite_engine)
    artifact_id = library.add(photo(library.root), origin="generated")
    actual = library.get(artifact_id)
    actual.write_bytes(actual.read_bytes() + b"tampered")
    assert library.get(artifact_id) is None
    assert library.describe(artifact_id) is None
    manifest = library.root / (artifact_id + ".meta.json")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["file"] = "../../private.png"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    assert library.get(artifact_id) is None and library.list() == ()


@pytest.mark.parametrize("name,body", [
    ("broken.png", b"not pixels"), ("code.py", b"print(42)"), ("secret.bin", b"secret"),
    ("truncated.json", b'{"result":'), ("control.txt", b"private\x00bytes"),
])
def test_unsupported_and_invalid_output_is_not_registered(sqlite_engine, tmp_path, name, body):
    library = store(tmp_path, sqlite_engine)
    library.root.mkdir(parents=True, exist_ok=True)
    path = library.root / name
    path.write_bytes(body)
    with pytest.raises(ValueError):
        library.add(path, origin="generated")
    assert library.list() == ()


def test_arbitrary_host_path_and_fabricated_ids_cannot_be_used(sqlite_engine, tmp_path):
    library = store(tmp_path, sqlite_engine)
    outside = photo(tmp_path / "other")
    with pytest.raises(ValueError):
        library.add(outside, origin="generated")
    for artifact_id in ("../../other/provider.png", "file:///private", "art-missing", None):
        assert library.get(artifact_id) is None


def test_explicit_allowed_input_directory_copies_verified_result(sqlite_engine, tmp_path):
    incoming = tmp_path / "operator-input"
    path = photo(incoming)
    library = AutonomousArtifactStore(tmp_path / "artifacts", account_id=ACCOUNT, scope="qzone_publish",
        engine=sqlite_engine, source_message_ids=lambda: (), allowed_input_roots=(incoming,))
    artifact_id = library.add(path, origin="operator_generated")
    assert library.get(artifact_id).is_relative_to(library.root)


def test_list_returns_only_twelve_recent_owned_results(sqlite_engine, tmp_path):
    library = store(tmp_path, sqlite_engine)
    original = photo(library.root)
    ids = [library.add(original, origin="generated") for _ in range(14)]
    listed = library.list()
    assert len(listed) == 12
    assert {item["id"] for item in listed} == set(ids[-12:])
    assert library.get(ids[0]) is not None


def test_oversized_source_and_file_budgets_are_rejected(sqlite_engine, tmp_path):
    library = store(tmp_path, sqlite_engine, sources=tuple(range(1, 202)))
    with pytest.raises(ValueError):
        library.add(photo(library.root), origin="generated")
    library = store(tmp_path, sqlite_engine)
    oversized = library.root / "oversized.txt"
    oversized.write_bytes(b"a" * (8 * 1024 * 1024 + 1))
    with pytest.raises(ValueError):
        library.add(oversized, origin="generated")


def test_symlink_input_never_registers_even_when_target_is_in_output_root(sqlite_engine, tmp_path):
    library = store(tmp_path, sqlite_engine)
    original = photo(library.root)
    alias = library.root / "alias.png"
    try:
        alias.symlink_to(original)
    except OSError:
        pytest.skip("host cannot create file symlinks")
    with pytest.raises(ValueError):
        library.add(alias, origin="generated")


def test_pixel_budget_and_truncated_jpeg_cannot_be_persisted(sqlite_engine, tmp_path, monkeypatch):
    library = store(tmp_path, sqlite_engine)
    valid = photo(library.root)
    monkeypatch.setattr(artifacts, "MAX_PIXELS", 100)
    with pytest.raises(ValueError):
        library.add(valid, origin="generated")
    monkeypatch.setattr(artifacts, "MAX_PIXELS", 16_000_000)
    output = io.BytesIO()
    Image.new("RGB", (100, 100), "red").save(output, "JPEG")
    broken = library.root / "truncated.jpg"
    broken.write_bytes(output.getvalue()[:-20])
    with pytest.raises(ValueError):
        library.add(broken, origin="generated")


def test_symlinked_manifest_or_replaced_output_cannot_supply_pixels(sqlite_engine, tmp_path):
    library = store(tmp_path, sqlite_engine)
    artifact_id = library.add(photo(library.root), origin="generated")
    target = library.get(artifact_id)
    alternate = photo(tmp_path / "outside")
    saved = target.read_bytes()
    target.unlink()
    try:
        target.symlink_to(alternate)
    except OSError:
        target.write_bytes(saved)
        pytest.skip("host cannot create file symlinks")
    assert library.get(artifact_id) is None and library.list() == ()


def test_duplicate_or_non_integer_provenance_is_not_accepted(sqlite_engine, tmp_path):
    source_id = source(sqlite_engine)
    for bad in ((source_id, source_id), (str(source_id),), (True,)):
        library = store(tmp_path, sqlite_engine, sources=bad)
        with pytest.raises(ValueError):
            library.add(photo(library.root), origin="generated")


def test_provenance_survives_wake_and_keeps_every_canonical_source(sqlite_engine, tmp_path):
    first = source(sqlite_engine, name="first")
    second = source(sqlite_engine, name="second")
    library = store(tmp_path, sqlite_engine, sources=(second, first))
    artifact_id = library.add(photo(library.root), origin="generated")
    resumed = store(tmp_path, sqlite_engine)
    assert resumed.provenance(artifact_id) == (second, first)
    empty = resumed.add(photo(resumed.root, "fresh.png"), origin="source_free")
    assert resumed.provenance(empty) == ()


def test_provenance_revalidates_authority_and_cannot_silently_drop_sources(sqlite_engine, tmp_path):
    source_id = source(sqlite_engine)
    library = store(tmp_path, sqlite_engine, sources=(source_id,))
    artifact_id = library.add(photo(library.root), origin="generated")
    assert library.get(artifact_id) is not None
    for forbidden in (store(tmp_path, sqlite_engine, account=ACCOUNT + 1),
                      store(tmp_path, sqlite_engine, scope=f"group:{GROUP + 1}")):
        with pytest.raises(ValueError, match="artifact authority unavailable"):
            forbidden.provenance(artifact_id)
    with pytest.raises(ValueError, match="artifact authority unavailable"):
        library.provenance("art-missing")
    with session_scope(sqlite_engine) as session:
        session.get(Message, source_id).raw_json = {"delivery_state": "deleted"}
    with pytest.raises(ValueError, match="artifact authority unavailable"):
        library.provenance(artifact_id)
    assert library.get(artifact_id) is None


def test_recent_index_avoids_unbounded_logs_and_keeps_direct_reads_after_loss(sqlite_engine, tmp_path, monkeypatch):
    library = store(tmp_path, sqlite_engine)
    original = photo(library.root)
    ids = [library.add(original, origin="generated") for _ in range(14)]

    def no_directory_walk(*args, **kwargs):
        raise AssertionError("a healthy scoped catalog must not scan the shared output directory")

    monkeypatch.setattr(artifacts.os, "scandir", no_directory_walk)
    assert {item["id"] for item in store(tmp_path, sqlite_engine).list()} == set(ids[-12:])
    library._catalog.unlink()
    assert library.get(ids[0]) is not None


def test_missing_index_fallback_counts_nonmatching_logs_with_a_hard_bound(sqlite_engine, tmp_path, monkeypatch):
    library = store(tmp_path, sqlite_engine)
    artifact_id = library.add(photo(library.root), origin="generated")
    library._catalog.unlink()
    consumed = []

    class Entries:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            for index in range(100_000):
                consumed.append(index)
                yield type("Entry", (), {"name": f"turn-{index}.json"})()

    monkeypatch.setattr(artifacts.os, "scandir", lambda *_: Entries())
    assert library.list() == ()
    assert len(consumed) == artifacts.MAX_DIRECTORY_ENTRIES
    assert library.get(artifact_id) is not None


def test_corrupt_index_is_bounded_and_cannot_leak_other_scope_metadata(sqlite_engine, tmp_path):
    library = store(tmp_path, sqlite_engine)
    artifact_id = library.add(photo(library.root), origin="generated")
    library._catalog.write_text("broken", encoding="utf-8")
    assert library.list()[0]["id"] == artifact_id
    foreign = store(tmp_path, sqlite_engine, scope="qzone_publish")
    foreign._catalog.write_text(library._catalog.read_text(encoding="utf-8"), encoding="utf-8")
    assert foreign.list() == ()


def test_parallel_same_scope_adds_do_not_lose_catalog_entries(sqlite_engine, tmp_path):
    library = store(tmp_path, sqlite_engine)
    original = photo(library.root)

    def add_one(index):
        return store(tmp_path, sqlite_engine).add(original, origin=f"generated-{index}")

    with ThreadPoolExecutor(max_workers=6) as pool:
        ids = tuple(pool.map(add_one, range(24)))
    catalog = json.loads(library._catalog.read_text(encoding="utf-8"))
    assert set(catalog["ids"]) == set(ids)
    assert len(library.list()) == 12
    assert all(library.get(artifact_id) is not None for artifact_id in ids)


def test_manifest_validation_work_stays_bounded_when_sources_are_all_revoked(sqlite_engine, tmp_path, monkeypatch):
    library = store(tmp_path, sqlite_engine)
    original = photo(library.root)
    ids = [library.add(original, origin="generated") for _ in range(14)]
    checked = []

    def unavailable(artifact_id):
        checked.append(artifact_id)
        return None

    monkeypatch.setattr(library, "_read", unavailable)
    monkeypatch.setattr(artifacts, "MAX_LIST_CHECKS", 3)
    assert library.list() == ()
    assert checked == list(reversed(ids[-3:]))
