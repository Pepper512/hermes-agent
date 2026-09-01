"""Exact, caller-transaction-owned two-root session deletion tests."""

import sqlite3

import pytest

import hermes_state
from hermes_state import SessionDB


_REFUSAL = "exact session deletion refused"


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


@pytest.fixture()
def exact_delete():
    helper = getattr(hermes_state, "_delete_session_roots_exact_on", None)
    if callable(helper):
        return helper

    def missing_helper(*_args, **_kwargs):
        pytest.fail("connection-scoped exact deletion helper is missing")

    return missing_helper


def _create_root(db: SessionDB, session_id: str) -> None:
    db.create_session(session_id, "cli")
    db.append_message(session_id, "user", f"synthetic message for {session_id}")


def _create_exact_pair(db: SessionDB) -> tuple[str, str]:
    roots = ("root-a", "root-b")
    for root in roots:
        _create_root(db, root)
    return roots


def _assert_sessions_present(
    connection: sqlite3.Connection, session_ids: tuple[str, ...]
) -> None:
    placeholders = ",".join("?" for _ in session_ids)
    count = connection.execute(
        f"SELECT COUNT(*) FROM sessions WHERE id IN ({placeholders})",
        session_ids,
    ).fetchone()[0]
    assert count == len(session_ids)


def _assert_exact_pair_dependents_present(connection: sqlite3.Connection) -> None:
    assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert (
        connection.execute("SELECT COUNT(*) FROM session_model_usage").fetchone()[0]
        == 2
    )
    assert connection.execute("SELECT COUNT(*) FROM system_prompts").fetchone()[0] == 2


def _call_rejected(exact_delete, connection, roots, expected) -> None:
    connection.execute("BEGIN EXCLUSIVE")
    with pytest.raises(ValueError, match=f"^{_REFUSAL}$"):
        exact_delete(connection, roots, expected)
    assert connection.in_transaction is True
    connection.rollback()


def test_exact_batch_deletes_two_singleton_roots_with_fixed_counts(db, exact_delete):
    roots = _create_exact_pair(db)
    db.append_message("root-a", "assistant", "second synthetic message")
    for root in roots:
        db.update_system_prompt(root, f"synthetic prompt for {root}")
        db._conn.execute(
            "INSERT INTO session_model_usage (session_id, model) VALUES (?, ?)",
            (root, "synthetic-model"),
        )
    db._conn.commit()

    db._conn.execute("BEGIN EXCLUSIVE")
    counts = exact_delete(
        db._conn,
        roots,
        {"root-a": ("root-a",), "root-b": ("root-b",)},
    )
    assert counts.sessions == 2
    assert counts.messages == 3
    assert counts.session_model_usage == 2
    assert counts.system_prompts == 2
    assert db._conn.in_transaction is True
    db._conn.commit()

    assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert db.search_messages("synthetic") == []


def test_exact_batch_refuses_missing_root_without_ending_transaction(db, exact_delete):
    _create_root(db, "root-a")

    _call_rejected(
        exact_delete,
        db._conn,
        ("root-a", "missing-root"),
        {"root-a": ("root-a",), "missing-root": ("missing-root",)},
    )

    _assert_sessions_present(db._conn, ("root-a",))


def test_exact_batch_refuses_duplicate_roots(db, exact_delete):
    _create_root(db, "root-a")

    _call_rejected(
        exact_delete,
        db._conn,
        ("root-a", "root-a"),
        {"root-a": ("root-a",)},
    )

    _assert_sessions_present(db._conn, ("root-a",))


def test_exact_batch_refuses_overlapping_delegate_closures(db, exact_delete):
    roots = _create_exact_pair(db)
    db._conn.execute(
        "UPDATE sessions SET parent_session_id = ?, model_config = ? WHERE id = ?",
        ("root-a", '{"_delegate_from":"root-a"}', "root-b"),
    )
    db._conn.commit()

    _call_rejected(
        exact_delete,
        db._conn,
        roots,
        {"root-a": ("root-a",), "root-b": ("root-b",)},
    )

    _assert_sessions_present(db._conn, roots)


def test_exact_batch_refuses_root_marked_as_external_delegate(db, exact_delete):
    roots = _create_exact_pair(db)
    db._conn.execute(
        "UPDATE sessions SET model_config = ? WHERE id = ?",
        ('{"_delegate_from":"external-root"}', "root-a"),
    )
    db._conn.commit()

    _call_rejected(
        exact_delete,
        db._conn,
        roots,
        {"root-a": ("root-a",), "root-b": ("root-b",)},
    )

    _assert_sessions_present(db._conn, roots)


def test_exact_batch_refuses_delegate_added_after_expected_snapshot(db, exact_delete):
    roots = _create_exact_pair(db)
    expected = {"root-a": ("root-a",), "root-b": ("root-b",)}
    db.create_session(
        "late-delegate",
        "cli",
        parent_session_id="root-a",
        model_config={"_delegate_from": "root-a"},
    )

    _call_rejected(exact_delete, db._conn, roots, expected)

    _assert_sessions_present(db._conn, (*roots, "late-delegate"))


def test_exact_batch_refuses_delegate_root_removed_after_expected_snapshot(
    db, exact_delete
):
    roots = _create_exact_pair(db)
    expected = {
        "root-a": ("root-a",),
        "root-b": ("root-b",),
    }
    assert db.get_session_delete_targets("root-a") == ["root-a"]
    assert db.get_session_delete_targets("root-b") == ["root-b"]

    db._conn.execute(
        "UPDATE sessions SET parent_session_id = ?, model_config = ? WHERE id = ?",
        ("root-a", '{"_delegate_from":"root-a"}', "root-b"),
    )
    db._conn.execute("DELETE FROM messages WHERE session_id = ?", ("root-b",))
    db._conn.execute("DELETE FROM sessions WHERE id = ?", ("root-b",))
    db._conn.commit()

    db._conn.execute("BEGIN EXCLUSIVE")
    with pytest.raises(ValueError, match=f"^{_REFUSAL}$"):
        exact_delete(db._conn, roots, expected)
    assert db._conn.in_transaction is True
    _assert_sessions_present(db._conn, ("root-a",))
    assert (
        db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", ("root-a",)
        ).fetchone()[0]
        == 1
    )
    db._conn.rollback()


def test_exact_batch_refuses_branch_child(db, exact_delete):
    roots = _create_exact_pair(db)
    db.create_session(
        "branch",
        "cli",
        parent_session_id="root-a",
        model_config={"_branched_from": "root-a"},
    )

    _call_rejected(
        exact_delete,
        db._conn,
        roots,
        {"root-a": ("root-a",), "root-b": ("root-b",)},
    )

    _assert_sessions_present(db._conn, (*roots, "branch"))


def test_exact_batch_refuses_root_marked_as_branch(db, exact_delete):
    roots = _create_exact_pair(db)
    db._conn.execute(
        "UPDATE sessions SET model_config = ? WHERE id = ?",
        ('{"_branched_from":"former-parent"}', "root-a"),
    )
    db._conn.commit()

    _call_rejected(
        exact_delete,
        db._conn,
        roots,
        {"root-a": ("root-a",), "root-b": ("root-b",)},
    )

    _assert_sessions_present(db._conn, roots)


def test_exact_batch_refuses_root_with_lineage_parent(db, exact_delete):
    db.create_session("lineage-parent", "cli")
    db.create_session("root-a", "cli", parent_session_id="lineage-parent")
    db.append_message("root-a", "user", "synthetic root-a message")
    _create_root(db, "root-b")
    roots = ("root-a", "root-b")

    _call_rejected(
        exact_delete,
        db._conn,
        roots,
        {"root-a": ("root-a",), "root-b": ("root-b",)},
    )

    _assert_sessions_present(db._conn, ("lineage-parent", *roots))


def test_exact_batch_refuses_unexpected_third_expected_target(db, exact_delete):
    roots = _create_exact_pair(db)
    _create_root(db, "third-root")

    _call_rejected(
        exact_delete,
        db._conn,
        roots,
        {
            "root-a": ("root-a",),
            "root-b": ("root-b",),
            "third-root": ("third-root",),
        },
    )

    _assert_sessions_present(db._conn, (*roots, "third-root"))


def test_exact_batch_refuses_unequal_per_root_expected_sets(db, exact_delete):
    roots = _create_exact_pair(db)

    _call_rejected(
        exact_delete,
        db._conn,
        roots,
        {"root-a": ("root-a",), "root-b": ("root-b", "root-a")},
    )

    _assert_sessions_present(db._conn, roots)


def test_exact_batch_requires_caller_owned_transaction(db, exact_delete):
    roots = _create_exact_pair(db)

    with pytest.raises(ValueError, match=f"^{_REFUSAL}$"):
        exact_delete(
            db._conn,
            roots,
            {"root-a": ("root-a",), "root-b": ("root-b",)},
        )

    assert db._conn.in_transaction is False
    _assert_sessions_present(db._conn, roots)


def test_exact_batch_leaves_successful_transaction_for_caller(db, exact_delete):
    roots = _create_exact_pair(db)
    db._conn.execute("BEGIN EXCLUSIVE")

    exact_delete(
        db._conn,
        roots,
        {"root-a": ("root-a",), "root-b": ("root-b",)},
    )

    assert db._conn.in_transaction is True
    assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    db._conn.rollback()
    _assert_sessions_present(db._conn, roots)


def test_exact_batch_ordinary_failure_rolls_back_both_roots_and_dependents(
    db, exact_delete
):
    roots = _create_exact_pair(db)
    for root in roots:
        db.update_system_prompt(root, f"synthetic prompt for {root}")
        db._conn.execute(
            "INSERT INTO session_model_usage (session_id, model) VALUES (?, ?)",
            (root, "synthetic-model"),
        )
    db._conn.execute(
        """CREATE TRIGGER synthetic_second_root_failure
           BEFORE DELETE ON sessions WHEN OLD.id = 'root-b'
           BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END"""
    )
    db._conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="synthetic failure"):
        db._conn.execute("BEGIN EXCLUSIVE")
        try:
            exact_delete(
                db._conn,
                roots,
                {"root-a": ("root-a",), "root-b": ("root-b",)},
            )
        except Exception:
            db._conn.rollback()
            raise

    _assert_exact_pair_dependents_present(db._conn)


class _SyntheticBaseException(BaseException):
    pass


class _BaseExceptionAfterFirstRootConnection(sqlite3.Connection):
    fail_after_root: str | None = None

    def execute(self, sql, parameters=()):
        cursor = super().execute(sql, parameters)
        normalized = " ".join(sql.split()).upper()
        if normalized == "DELETE FROM SESSIONS WHERE ID = ?" and tuple(parameters) == (
            self.fail_after_root,
        ):
            raise _SyntheticBaseException
        return cursor


def test_exact_batch_base_exception_rolls_back_both_roots_and_dependents(
    tmp_path, exact_delete
):
    database = tmp_path / "base-exception.db"
    seed = SessionDB(db_path=database)
    roots = _create_exact_pair(seed)
    seed_connection = seed._conn
    assert seed_connection is not None
    for root in roots:
        seed.update_system_prompt(root, f"synthetic prompt for {root}")
        seed_connection.execute(
            "INSERT INTO session_model_usage (session_id, model) VALUES (?, ?)",
            (root, "synthetic-model"),
        )
    seed_connection.commit()
    seed.close()

    connection = sqlite3.connect(
        database,
        factory=_BaseExceptionAfterFirstRootConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.fail_after_root = "root-a"
    try:
        with pytest.raises(_SyntheticBaseException):
            connection.execute("BEGIN EXCLUSIVE")
            try:
                exact_delete(
                    connection,
                    roots,
                    {"root-a": ("root-a",), "root-b": ("root-b",)},
                )
            except BaseException:
                connection.rollback()
                raise
        _assert_exact_pair_dependents_present(connection)
    finally:
        connection.close()
