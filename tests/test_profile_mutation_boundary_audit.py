"""Static guard for the Task 12 profile-state maintenance lease inventory.

The guard deliberately reasons about *mutation coordinators*, not every SQL
leaf nested below ``SessionDB._execute_write``.  A new raw state.db writer or
transcript sidecar writer must either reuse an audited coordinator or extend
the reviewed inventory before it can land.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPO_ROOT
    / ".superpowers/sdd/2026-08-29-haos-health-connect-source"
    / "task12-cleanup-writer-audit.md"
)

CRITICAL_BOUNDARY_CONTRACTS = {
    "gateway/session.py::SessionStore._ensure_loaded_locked": {
        "profile_domains": ["fixed_launch_sessions_dir", "dynamic_session_db"],
        "lease_span": "before_mkdir_then_prove_same_or_ordered_dual_authority_through_reconcile_persistence",
    },
    "gateway/session.py::SessionStore._persist_routing_data": {
        "profile_domains": ["dynamic_session_db", "fixed_launch_sessions_dir"],
        "lease_span": "prove_same_or_ordered_dual_authority_through_db_commit_mirror_dir_fsync",
    },
    "gateway/session.py::SessionStore._save_sessions_json": {
        "profile_domains": ["fixed_launch_sessions_dir"],
        "lease_span": "caller_owned_fixed_profile_through_replace_and_parent_dir_fsync",
    },
    "gateway/shutdown_flush.py::_get_flush_dir": {
        "profile_domains": ["active_context_profile_pending_messages"],
        "lease_span": "caller_owned_before_mkdir_chmod",
    },
    "gateway/shutdown_flush.py::flush_pending_to_file": {
        "profile_domains": ["active_context_profile_pending_messages"],
        "lease_span": "before_get_flush_dir_through_all_payload_parent_dir_fsyncs",
    },
    "gateway/shutdown_flush.py::spool_dropped_transcript_message": {
        "profile_domains": ["active_context_profile_pending_messages"],
        "lease_span": "before_get_flush_dir_through_payload_parent_dir_fsync",
    },
    "gateway/shutdown_flush.py::flush_agent_history_to_file": {
        "profile_domains": ["active_context_profile_pending_messages"],
        "lease_span": "before_get_flush_dir_through_payload_parent_dir_fsync",
    },
    "gateway/shutdown_flush.py::recover_pending_to_db": {
        "profile_domains": ["active_spool_profile", "supplied_or_default_session_db_profile"],
        "lease_span": "prove_same_or_ordered_dual_authority_through_db_commit_and_spool_unlink",
    },
    "hermes_cli/update_cmd.py::_restore_state_db_from_snapshot": {
        "profile_domains": ["proven_live_state_db_profile"],
        "lease_span": "before_holder_check_through_sidecar_unlink_copy2_integrity_verification",
    },
    "hermes_state.py::_delete_session_root_on": {
        "profile_domains": ["caller_connection_profile"],
        "lease_span": "caller_owned_execute_write_or_exact_cleanup_transaction_through_root_delegate_prompt_deletion",
        "permitted_callers": [
            "hermes_state.py::SessionDB.delete_session._do",
            "hermes_state.py::_delete_session_roots_exact_on",
        ],
    },
    "hermes_state.py::_delete_unreferenced_system_prompts_on": {
        "profile_domains": ["caller_connection_profile"],
        "lease_span": "caller_owned_canonical_deletion_transaction_through_prompt_cleanup",
        "permitted_callers": [
            "hermes_state.py::SessionDB._delete_unreferenced_system_prompts",
            "hermes_state.py::_delete_session_root_on",
        ],
    },
    "hermes_state.py::SessionDB.purge_stale_tool_call_markers": {
        "profile_domains": ["session_db_profile"],
        "lease_span": "before_affected_row_admission_through_vacuum_into_update_commit_and_result",
    },
    "hermes_state_search.py::SessionSearchMixin._demote_legacy_fts_to_trash": {
        "profile_domains": ["session_db_profile"],
        "lease_span": "caller_owned_optimize_fts_storage_across_callback_and_direct_schema_commit",
    },
    "hermes_state_search.py::SessionSearchMixin._fts_cjk_reset_if_stale": {
        "profile_domains": ["session_db_profile"],
        "lease_span": "caller_owned_optimize_fts_storage_across_callback_and_direct_schema_commit",
    },
    "hermes_state_search.py::SessionSearchMixin.optimize_fts": {
        "profile_domains": ["session_db_profile"],
        "lease_span": "before_table_probe_through_commands_and_transaction_completion",
    },
    "hermes_state_search.py::SessionSearchMixin.rebuild_fts": {
        "profile_domains": ["session_db_profile"],
        "lease_span": "before_rebuild_admission_through_each_commit_or_rollback",
    },
}

MUTATING_SQL_PREFIXES = (
    "alter ",
    "begin ",
    "create ",
    "delete ",
    "drop ",
    "insert ",
    "pragma journal_mode",
    "pragma wal_checkpoint",
    "reindex",
    "replace ",
    "update ",
    "vacuum",
)

AUDITED_DIRECT_BOUNDARIES = {
    "agent/agent_init.py::init_agent",
    "agent/agent_runtime_helpers.py::dump_api_request_debug",
    "gateway/platforms/api_server.py::APIServerAdapter._handle_create_session",
    "gateway/session.py::SessionStore._persist_routing_data",
    "gateway/session.py::SessionStore._save_sessions_json",
    "gateway/session.py::SessionStore._ensure_loaded_locked",
    "gateway/shutdown_flush.py::_get_flush_dir",
    "gateway/shutdown_flush.py::_write_payload",
    "gateway/shutdown_flush.py::drain_transcript_spool",
    "gateway/shutdown_flush.py::flush_agent_history_to_file",
    "gateway/shutdown_flush.py::flush_pending_to_file",
    "gateway/shutdown_flush.py::recover_pending_to_db",
    "gateway/shutdown_flush.py::spool_dropped_transcript_message",
    "hermes_cli/backup.py::_safe_copy_db",
    "hermes_cli/backup.py::_safe_restore_db",
    "hermes_cli/doctor.py::run_doctor",
    "hermes_cli/update_cmd.py::_clear_stale_sqlite_sidecars",
    "hermes_cli/update_cmd.py::_restore_state_db_from_snapshot",
    "hermes_state.py::_db_opens_cleanly",
    "hermes_state.py::_backup_db_file",
    "hermes_state.py::_connect_repair_durable",
    "hermes_state.py::_connect_tracked_db",
    "hermes_state.py::_copy_database_snapshot",
    "hermes_state.py::_delete_session_root_on",
    "hermes_state.py::_delete_unreferenced_system_prompts_on",
    "hermes_state.py::_live_writer_holds_db",
    "hermes_state.py::_exclusive_repair_db_guard",
    "hermes_state.py::_prune_malformed_backups",
    "hermes_state.py::_record_repair_outcome",
    "hermes_state.py::_repair_state_db_schema_locked",
    "hermes_state.py::_restore_journal_mode_after_repair",
    "hermes_state.py::_run_repair_strategies",
    "hermes_state.py::_set_journal_mode_no_wait",
    "hermes_state.py::_unlink_db_triple",
    "hermes_state.py::apply_wal_with_fallback",
    "hermes_state.py::preflight_db_writability",
    "hermes_state.py::quarantine_zeroed_state_db",
    "hermes_state.py::repair_state_db_schema",
    "hermes_state.py::SessionDB.__init__",
    "hermes_state.py::SessionDB._enter_fts_fail_open",
    "hermes_state.py::SessionDB._execute_write",
    "hermes_state.py::SessionDB._try_wal_checkpoint",
    "hermes_state.py::SessionDB.close",
    "hermes_state.py::SessionDB._remove_session_files",
    "hermes_state.py::SessionDB.delete_empty_sessions",
    "hermes_state.py::SessionDB.delete_session",
    "hermes_state.py::SessionDB.delete_session_if_empty",
    "hermes_state.py::SessionDB.delete_sessions",
    "hermes_state.py::SessionDB.maybe_auto_prune_and_vacuum",
    "hermes_state.py::SessionDB.prune_sessions",
    "hermes_state.py::SessionDB.prune_empty_ghost_sessions",
    "hermes_state.py::SessionDB.purge_stale_tool_call_markers",
    "hermes_state.py::SessionDB.vacuum",
    "hermes_state_search.py::SessionSearchMixin._demote_legacy_fts_to_trash",
    "hermes_state_search.py::SessionSearchMixin._fts_cjk_reset_if_stale",
    "hermes_state_search.py::SessionSearchMixin.optimize_fts_storage",
    "hermes_state_search.py::SessionSearchMixin._merge_fts_incrementally",
    "hermes_state_search.py::SessionSearchMixin.optimize_fts",
    "hermes_state_search.py::SessionSearchMixin.rebuild_fts",
    "run_agent.py::AIAgent._save_session_log",
}

SESSION_WRITE_LEAVES = {
    "SessionDB._check_transcript_write_guards",
    "SessionDB._delete_unreferenced_system_prompts",
    "SessionDB._drop_fts_triggers",
    "SessionDB._ensure_fts_cjk_schema",
    "SessionDB._ensure_fts_schema",
    "SessionDB._insert_message_rows",
    "SessionDB._record_model_usage",
    "SessionDB._store_system_prompt",
    "_delete_delegate_children",
    "_repair_state_db_schema_locked",
    "_restore_journal_mode_after_repair",
    "_run_repair_strategies",
}

CANONICAL_FILE_WRITERS = {
    "agent/agent_runtime_helpers.py": {"dump_api_request_debug"},
    "gateway/session.py": {
        "SessionStore._persist_routing_data",
        "SessionStore._save_sessions_json",
    },
    "gateway/shutdown_flush.py": {
        "_write_payload",
        "drain_transcript_spool",
        "recover_pending_to_db",
    },
    "hermes_state.py": {
        "SessionDB._remove_session_files",
        "SessionDB.delete_empty_sessions",
        "SessionDB.delete_session",
        "SessionDB.delete_session_if_empty",
        "SessionDB.delete_sessions",
        "SessionDB.prune_sessions",
        "SessionDB.prune_empty_ghost_sessions",
    },
    "hermes_cli/update_cmd.py": {"_clear_stale_sqlite_sidecars"},
    "run_agent.py": {"AIAgent._save_session_log"},
}

OPTIMIZER_OWNED_MULTIPHASE_LEAVES = {
    "_demote_legacy_fts_to_trash",
    "_fts_cjk_reset_if_stale",
}

CANONICAL_DELETION_LEAF_CALLERS = {
    "_delete_session_root_on": {
        ("hermes_state.py", "SessionDB.delete_session._do"),
        ("hermes_state.py", "_delete_session_roots_exact_on"),
    },
    "_delete_unreferenced_system_prompts_on": {
        ("hermes_state.py", "SessionDB._delete_unreferenced_system_prompts"),
        ("hermes_state.py", "_delete_session_root_on"),
    },
}


def _qualified_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    names: list[str] = []
    cursor: ast.AST | None = node
    while cursor is not None:
        if isinstance(cursor, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(cursor.name)
        cursor = parents.get(cursor)
    return ".".join(reversed(names)) or "<module>"


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _literal_text(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    return ""


def _has_mutating_sql(function: ast.AST) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or _call_name(node) not in {
            "execute", "executemany", "executescript"
        }:
            continue
        if _call_name(node) == "executescript":
            return True
        if not node.args:
            continue
        sql = " ".join(_literal_text(node.args[0]).lower().split())
        if any(sql.startswith(prefix) for prefix in MUTATING_SQL_PREFIXES):
            return True
    return False


def _opens_writable_sqlite(function: ast.AST) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or _call_name(node) != "connect":
            continue
        if not node.args:
            continue
        target = ast.unparse(node.args[0]).lower()
        if ":memory:" in target or "mode=ro" in target:
            continue
        return True
    return False


def _has_live_state_db_replacement(function: ast.AST) -> bool:
    """Detect copy-over-live-state operations hidden from the SQL scanner."""
    source = ast.unparse(function)
    has_state_destination = any(
        marker in source for marker in ("state_path", "state_db_path", "db_path")
    )
    has_copy_over = "shutil.copy2(" in source or "shutil.copyfile(" in source
    clears_sidecars = (
        "_clear_stale_sqlite_sidecars(" in source
        or (".unlink(" in source and any(suffix in source for suffix in ("-wal", "-shm", "-journal")))
    )
    return has_state_destination and has_copy_over and clears_sidecars


def _profile_path_mutation_families(function: ast.AST) -> set[str]:
    """Classify canonical profile filesystem mutations by semantic sink."""
    source = ast.unparse(function)
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    call_names = {_call_name(node) for node in calls}
    families: set[str] = set()

    def mutating_call_touches(tokens: tuple[str, ...]) -> bool:
        for call in calls:
            if _call_name(call) not in {
                "atomic_json_write", "atomic_replace", "chmod", "copy2", "copyfile",
                "dump", "mkdir", "rename", "replace", "unlink", "write_text",
            }:
                continue
            rendered = ast.unparse(call)
            if any(token in rendered for token in tokens):
                return True
        return False

    if _has_live_state_db_replacement(function) or (
        "state.db" in source
        and mutating_call_touches(("state_path", "state_db_path", "db_path"))
    ):
        families.add("state_db_or_journal")

    if (
        call_names.intersection({"_get_flush_dir", "_write_payload"})
        or (
            "flush_dir" in source
            and mutating_call_touches(("flush_dir", "final_path", "path"))
        )
    ):
        families.add("pending_transcript_spool")

    canonical_session_sink = (
        ("logs_dir" in source and any(marker in source for marker in ("session_", "request_dump_")))
        or (
            "sessions_dir" in source
            and any(
                marker in source
                for marker in ("sessions_file", "request_dump_", ".jsonl", "{session_id}.json")
            )
        )
    )
    if (
        canonical_session_sink
        and mutating_call_touches(
            ("logs_dir", "sessions_dir", "log_file", "dump_file", "sessions_file")
        )
    ):
        families.add("session_or_request_sidecar")
    return families


def _has_direct_self_connection_mutation(function: ast.AST) -> bool:
    """Find mutating ``self._conn`` calls, excluding nested callbacks."""

    class DirectCallVisitor(ast.NodeVisitor):
        found = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is function:
                for statement in node.body:
                    self.visit(statement)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"execute", "executemany", "executescript"}
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "self"
                and node.func.value.attr == "_conn"
                and node.args
            ):
                sql = " ".join(_literal_text(node.args[0]).lower().split())
                if node.func.attr == "executescript" or any(
                    sql.startswith(prefix) for prefix in MUTATING_SQL_PREFIXES
                ):
                    self.found = True
            self.generic_visit(node)

    visitor = DirectCallVisitor()
    visitor.visit(function)
    return visitor.found


def _top_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    cursor = node
    while isinstance(parents.get(cursor), (ast.FunctionDef, ast.AsyncFunctionDef)):
        cursor = parents[cursor]
    return cursor


def _coordinator_for(
    relative: str,
    function: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    qualified = _qualified_name(function, parents)
    source = ast.unparse(function)

    if relative == "hermes_state.py":
        if qualified == "_on_disk_journal_mode":
            # Bare PRAGMA journal_mode is an inspection; only an assignment
            # form changes journal mode and is frozen at its setter/caller.
            return None
        if qualified.startswith("SessionDB.") and _has_direct_self_connection_mutation(function):
            # A method can mutate self._conn directly and later delegate a
            # second phase to _execute_write.  The direct phase makes the
            # entire public method a boundary; delegation must not mask it.
            return f"{relative}::{qualified}"
        if qualified in SESSION_WRITE_LEAVES:
            if qualified.startswith("SessionDB.") or qualified == "_delete_delegate_children":
                return "hermes_state.py::SessionDB._execute_write"
            return f"{relative}::{qualified}"
        if (
            qualified.startswith("SessionDB.")
            and "self._execute_write(" in source
            and "self._remove_session_files(" in source
        ):
            # A delegated SQL phase followed by a filesystem mutation needs
            # one outer span.  Collapsing it to _execute_write would hide the
            # post-callback lease gap this gate is intended to prevent.
            return f"{relative}::{qualified}"
        if qualified.startswith("SessionDB.") and "self._execute_write(" in source:
            return "hermes_state.py::SessionDB._execute_write"
        if qualified.startswith("SessionDB.") and qualified.endswith("._do"):
            return "hermes_state.py::SessionDB._execute_write"
        return f"{relative}::{qualified}"

    if relative == "hermes_state_schema.py":
        # Schema helpers run either during the writable constructor or from a
        # callback already admitted through SessionDB._execute_write.
        return "hermes_state.py::SessionDB.__init__"

    if relative.startswith("hermes_state_search"):
        if _has_direct_self_connection_mutation(function):
            return f"{relative}::{qualified}"
        if qualified.startswith("SessionSearchMixin.optimize_fts_storage"):
            return "hermes_state_search.py::SessionSearchMixin.optimize_fts_storage"
        return "hermes_state.py::SessionDB._execute_write"

    # Outside the state modules, only a raw writable SQLite open paired with
    # mutation SQL is a new coordinator. SessionDB callers inherit the lease
    # inside SessionDB and read-only ``mode=ro`` opens are intentionally out.
    explicitly_scoped = {
        "gateway/platforms/api_server.py": {
            "APIServerAdapter._handle_create_session",
        },
        "hermes_cli/backup.py": {"_safe_copy_db", "_safe_restore_db"},
        "hermes_cli/doctor.py": {"run_doctor"},
    }
    if qualified in explicitly_scoped.get(relative, set()):
        return f"{relative}::{qualified}"
    if _profile_path_mutation_families(function):
        return f"{relative}::{qualified}"
    if _has_live_state_db_replacement(function):
        return f"{relative}::{qualified}"
    if (
        "state.db" in source
        and _opens_writable_sqlite(function)
        and _has_mutating_sql(function)
    ):
        return f"{relative}::{qualified}"
    if "state.db" in source and "._conn.execute(" in source and _has_mutating_sql(function):
        return f"{relative}::{qualified}"
    return None


def _audited_python_sources(root: Path) -> list[Path]:
    return [
        root / "hermes_state.py",
        root / "hermes_state_schema.py",
        *sorted(root.glob("hermes_state_search*.py")),
        root / "run_agent.py",
        *sorted((root / "agent").rglob("*.py")),
        *sorted((root / "hermes_cli").rglob("*.py")),
        *sorted((root / "gateway").rglob("*.py")),
        *sorted((root / "tui_gateway").rglob("*.py")),
    ]


def _audited_source_texts(root: Path) -> list[tuple[str, str]]:
    return [
        (path.relative_to(root).as_posix(), path.read_text(encoding="utf-8"))
        for path in _audited_python_sources(root)
        if path.exists()
    ]


def _discover_profile_mutation_boundaries(root: Path) -> set[str]:
    sources = _audited_python_sources(root)
    discovered: set[str] = set()
    for path in sources:
        if not path.exists():
            continue
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        functions = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _top_function(node, parents) is node
        ]
        for function in functions:
            if (
                _has_mutating_sql(function)
                or _has_live_state_db_replacement(function)
                or _profile_path_mutation_families(function)
            ):
                coordinator = _coordinator_for(relative, function, parents)
                if coordinator:
                    discovered.add(coordinator)

        for writer in CANONICAL_FILE_WRITERS.get(relative, set()):
            discovered.add(f"{relative}::{writer}")
    return discovered


def _owned_leaf_callers(source: str, leaf_names: set[str]) -> set[tuple[str, str]]:
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    callers: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in leaf_names:
            continue
        cursor: ast.AST | None = parents.get(node)
        while cursor is not None and not isinstance(
            cursor, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            cursor = parents.get(cursor)
        caller = _qualified_name(cursor, parents) if cursor is not None else "<module>"
        callers.add((_call_name(node), caller))
    return callers


def _optimizer_owned_leaf_callers(source: str) -> set[tuple[str, str]]:
    return _owned_leaf_callers(source, OPTIMIZER_OWNED_MULTIPHASE_LEAVES)


def _nearest_import_scope(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    module: ast.Module,
) -> ast.AST:
    cursor: ast.AST | None = parents.get(node)
    while cursor is not None:
        if isinstance(
            cursor, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            return cursor
        cursor = parents.get(cursor)
    return module


def _visible_import_scopes(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    module: ast.Module,
) -> list[ast.AST]:
    scopes: list[ast.AST] = []
    cursor: ast.AST | None = parents.get(node)
    has_function_scope = False
    while cursor is not None:
        if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(cursor)
            has_function_scope = True
        elif isinstance(cursor, ast.ClassDef) and not has_function_scope:
            scopes.append(cursor)
        elif isinstance(cursor, ast.Module):
            scopes.append(cursor)
            break
        cursor = parents.get(cursor)
    if not scopes:
        scopes.append(module)
    return scopes


def _canonical_deletion_leaf_callers(
    sources: list[tuple[str, str]],
) -> set[tuple[str, str, str]]:
    leaf_names = set(CANONICAL_DELETION_LEAF_CALLERS)
    module_binding = "<hermes_state_module>"
    callers: set[tuple[str, str, str]] = set()
    for relative, source in sources:
        if not any(leaf_name in source for leaf_name in leaf_names):
            continue
        tree = ast.parse(source, filename=relative)
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        binding_events: dict[
            ast.AST, dict[str, list[tuple[tuple[int, int], str | None]]]
        ] = {}
        assignment_events: list[tuple[ast.AST, str, int, ast.AST]] = []

        def add_binding(
            scope: ast.AST,
            name: str,
            node: ast.AST,
            binding: str | None,
        ) -> int:
            position = (
                getattr(node, "end_lineno", node.lineno),
                getattr(node, "end_col_offset", node.col_offset),
            )
            events = binding_events.setdefault(scope, {}).setdefault(name, [])
            events.append((position, binding))
            return len(events) - 1

        for node in ast.walk(tree):
            scope = _nearest_import_scope(node, parents, tree)
            if isinstance(node, ast.ImportFrom) and node.module == "hermes_state":
                for alias in node.names:
                    add_binding(
                        scope,
                        alias.asname or alias.name,
                        node,
                        alias.name if alias.name in leaf_names else None,
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    add_binding(
                        scope,
                        alias.asname or alias.name.split(".", maxsplit=1)[0],
                        node,
                        module_binding if alias.name == "hermes_state" else None,
                    )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        add_binding(scope, alias.asname or alias.name, node, None)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definition_binding = (
                    node.name
                    if relative == "hermes_state.py"
                    and scope is tree
                    and node.name in leaf_names
                    else None
                )
                add_binding(scope, node.name, node, definition_binding)
                arguments = [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                if node.args.vararg is not None:
                    arguments.append(node.args.vararg)
                if node.args.kwarg is not None:
                    arguments.append(node.args.kwarg)
                for argument in arguments:
                    add_binding(node, argument.arg, node, None)
            elif isinstance(node, ast.ClassDef):
                add_binding(scope, node.name, node, None)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        index = add_binding(scope, target.id, node, None)
                        assignment_events.append((scope, target.id, index, node.value))
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and isinstance(
                node.target, ast.Name
            ):
                index = add_binding(scope, node.target.id, node, None)
                if node.value is not None:
                    assignment_events.append(
                        (scope, node.target.id, index, node.value)
                    )
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                add_binding(scope, node.target.id, node, None)

        def resolve_name(name: str, node: ast.AST) -> str | None:
            reference_scope = _nearest_import_scope(node, parents, tree)
            position = (node.lineno, node.col_offset)
            for scope in _visible_import_scopes(node, parents, tree):
                events = binding_events.get(scope, {}).get(name)
                if not events:
                    continue
                if scope is reference_scope:
                    preceding = [event for event in events if event[0] < position]
                    if not preceding:
                        return None
                    return max(preceding, key=lambda event: event[0])[1]
                return max(events, key=lambda event: event[0])[1]
            return None

        def resolve_reference(node: ast.AST) -> str | None:
            if isinstance(node, ast.Name):
                return resolve_name(node.id, node)
            if (
                isinstance(node, ast.Attribute)
                and node.attr in leaf_names
                and isinstance(node.value, ast.Name)
                and resolve_name(node.value.id, node) == module_binding
            ):
                return node.attr
            return None

        while True:
            changed = False
            for scope, name, index, value in assignment_events:
                binding = resolve_reference(value)
                if binding is None:
                    continue
                position, previous = binding_events[scope][name][index]
                if previous != binding:
                    binding_events[scope][name][index] = (position, binding)
                    changed = True
            if not changed:
                break

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            leaf = resolve_reference(node.func)
            if leaf not in leaf_names:
                continue

            cursor: ast.AST | None = parents.get(node)
            while cursor is not None and not isinstance(
                cursor, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                cursor = parents.get(cursor)
            caller = (
                _qualified_name(cursor, parents) if cursor is not None else "<module>"
            )
            callers.add((leaf, relative, caller))
    return callers


def _assert_canonical_deletion_leaf_callers(
    sources: list[tuple[str, str]],
) -> None:
    expected = {
        (leaf, relative, caller)
        for leaf, permitted_callers in CANONICAL_DELETION_LEAF_CALLERS.items()
        for relative, caller in permitted_callers
    }
    actual = _canonical_deletion_leaf_callers(sources)
    assert actual == expected, (
        "canonical_deletion_leaf_callers_changed: "
        f"unexpected={sorted(actual - expected)!r}, missing={sorted(expected - actual)!r}"
    )


def test_static_gate_rejects_unleased_profile_writer() -> None:
    discovered = _discover_profile_mutation_boundaries(REPO_ROOT)
    unaudited = sorted(discovered - AUDITED_DIRECT_BOUNDARIES)
    assert not unaudited, "unaudited_profile_mutation_boundary: " + ", ".join(unaudited)


def test_read_only_sqlite_path_is_not_a_mutation_boundary(tmp_path: Path) -> None:
    source = tmp_path / "readonly.py"
    source.write_text(
        "import sqlite3\n"
        "def inspect(profile_root):\n"
        "    conn = sqlite3.connect(f'file:{profile_root}/state.db?mode=ro', uri=True)\n"
        "    return conn.execute('SELECT COUNT(*) FROM sessions').fetchone()\n",
        encoding="utf-8",
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    assert not _opens_writable_sqlite(function)
    assert not _has_mutating_sql(function)


def test_mixed_direct_and_delegated_mutation_keeps_direct_boundary() -> None:
    tree = ast.parse(
        """
class SessionDB:
    def mixed_writer(self):
        self._conn.execute("VACUUM INTO ?", ("snapshot.db",))
        self._execute_write(lambda conn: conn.execute("DELETE FROM sessions"))
"""
    )
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "mixed_writer"
    )
    assert _coordinator_for("hermes_state.py", function, parents) == (
        "hermes_state.py::SessionDB.mixed_writer"
    )


def test_delegated_sql_followed_by_sidecar_mutation_keeps_outer_boundary() -> None:
    tree = ast.parse(
        """
class SessionDB:
    def mixed_writer(self, sessions_dir):
        removed = self._execute_write(lambda conn: conn.execute("DELETE FROM sessions"))
        self._remove_session_files(sessions_dir, removed)
"""
    )
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "mixed_writer"
    )
    assert _coordinator_for("hermes_state.py", function, parents) == (
        "hermes_state.py::SessionDB.mixed_writer"
    )


def test_optimizer_owned_multiphase_leaves_are_frozen() -> None:
    required = {
        "hermes_state_search.py::SessionSearchMixin._fts_cjk_reset_if_stale",
        "hermes_state_search.py::SessionSearchMixin._demote_legacy_fts_to_trash",
    }
    assert required <= AUDITED_DIRECT_BOUNDARIES

    callers = _optimizer_owned_leaf_callers(
        (REPO_ROOT / "hermes_state_search.py").read_text(encoding="utf-8")
    )
    assert callers == {
        ("_demote_legacy_fts_to_trash", "SessionSearchMixin.optimize_fts_storage"),
        ("_fts_cjk_reset_if_stale", "SessionSearchMixin.optimize_fts_storage"),
    }

    synthetic_outside_call = """
class SessionSearchMixin:
    def new_public_writer(self):
        self._fts_cjk_reset_if_stale()
"""
    assert _optimizer_owned_leaf_callers(synthetic_outside_call) == {
        ("_fts_cjk_reset_if_stale", "SessionSearchMixin.new_public_writer")
    }


def test_canonical_deletion_leaves_and_exact_callers_are_frozen() -> None:
    required = {
        "hermes_state.py::_delete_session_root_on",
        "hermes_state.py::_delete_unreferenced_system_prompts_on",
    }
    assert required <= AUDITED_DIRECT_BOUNDARIES

    _assert_canonical_deletion_leaf_callers(_audited_source_texts(REPO_ROOT))


def test_imported_outside_module_deletion_leaf_caller_fails_gate() -> None:
    synthetic_outside_calls = """
from hermes_state import _delete_session_root_on as remove_root
import hermes_state as state

def new_root_delete(connection):
    remove_root(connection, "root")

def new_prompt_cleanup(connection):
    state._delete_unreferenced_system_prompts_on(connection)

def unrelated_terminal_name(service, connection):
    service._delete_session_root_on(connection, "unrelated")
"""
    injected = [("gateway/injected_delete.py", synthetic_outside_calls)]
    assert _canonical_deletion_leaf_callers(injected) == {
        (
            "_delete_session_root_on",
            "gateway/injected_delete.py",
            "new_root_delete",
        ),
        (
            "_delete_unreferenced_system_prompts_on",
            "gateway/injected_delete.py",
            "new_prompt_cleanup",
        ),
    }
    with pytest.raises(
        AssertionError, match="^canonical_deletion_leaf_callers_changed:"
    ):
        _assert_canonical_deletion_leaf_callers([
            (
                "hermes_state.py",
                (REPO_ROOT / "hermes_state.py").read_text(encoding="utf-8"),
            ),
            *injected,
        ])


def test_assignment_aliased_outside_module_deletion_leaf_callers_fail_gate() -> None:
    synthetic_outside_calls = """
from hermes_state import _delete_session_root_on as imported_root_delete
import hermes_state as state

assigned_root_delete = imported_root_delete

def new_root_delete(connection):
    assigned_root_delete(connection, "root")

def new_prompt_cleanup(connection):
    assigned_prompt_cleanup = state._delete_unreferenced_system_prompts_on
    assigned_prompt_cleanup(connection)

def unrelated_root_assignment(service, connection):
    assigned_root_delete = service._delete_session_root_on
    assigned_root_delete(connection, "unrelated")

def unrelated_prompt_assignment(service, connection):
    assigned_prompt_cleanup = service._delete_unreferenced_system_prompts_on
    assigned_prompt_cleanup(connection)
"""
    injected = [("gateway/injected_assignment_delete.py", synthetic_outside_calls)]
    assert _canonical_deletion_leaf_callers(injected) == {
        (
            "_delete_session_root_on",
            "gateway/injected_assignment_delete.py",
            "new_root_delete",
        ),
        (
            "_delete_unreferenced_system_prompts_on",
            "gateway/injected_assignment_delete.py",
            "new_prompt_cleanup",
        ),
    }
    with pytest.raises(
        AssertionError, match="^canonical_deletion_leaf_callers_changed:"
    ):
        _assert_canonical_deletion_leaf_callers([
            (
                "hermes_state.py",
                (REPO_ROOT / "hermes_state.py").read_text(encoding="utf-8"),
            ),
            *injected,
        ])


def test_audit_artifact_freezes_every_boundary() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    documented = set(re.findall(r"^\| `([^`]+)` \|", audit, flags=re.MULTILINE))
    missing = sorted(AUDITED_DIRECT_BOUNDARIES - documented)
    assert not missing, "audit_missing_profile_mutation_boundary: " + ", ".join(missing)

    contract_match = re.search(
        r"<!-- TASK12_BOUNDARY_CONTRACTS_START -->\n(.*?)\n<!-- TASK12_BOUNDARY_CONTRACTS_END -->",
        audit,
        flags=re.DOTALL,
    )
    assert contract_match is not None, "audit_missing_machine_boundary_contracts"
    assert json.loads(contract_match.group(1)) == CRITICAL_BOUNDARY_CONTRACTS
