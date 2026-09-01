# Task 12 Hermes profile-state mutation-boundary audit

**Source frozen:** `ef05af5c7824a875693bc13e472195323d40e257`
**Scope:** Task 1 of the approved exact-cleanup maintenance plan (`ace1017`)
**Result:** inventory only; no production, live profile, service, database, or
network mutation was performed.

## Audit rule

The future profile shared lease must begin before a mutation can observe or
create profile state, the recovery barrier must be checked after that lease is
held, and the lease must remain held through commit plus every related
filesystem publication/removal and durability operation. A leaf that receives
an already-open connection does not acquire a second lease; its enclosing
coordinator owns the complete span.

The companion static test treats read-only `mode=ro` SQLite opens and `SELECT` /
read-only pragmas as non-mutations. It rejects a new raw writable `state.db`
connection paired with mutation SQL and freezes all canonical transcript-file
writers/deleters below.

## Authoritative inventory

| File and function | Profile resolution | Required future lease span |
| --- | --- | --- |
| `hermes_state.py::SessionDB.__init__` | Explicit `db_path`, otherwise `_default_db_path()` → active `get_hermes_home()/state.db`; read-only construction is excluded. | Writable connection creation through pragma selection, complete schema initialization/migration/rebuild, commit, and startup checkpoint. The barrier check belongs immediately after lease acquisition and before the writable open. |
| `hermes_state.py::_connect_tracked_db` | Explicit path plus optional tracking path; writable SessionDB calls resolve to `self.db_path`. | Connection factory only: a writable caller must already hold the profile lease before entry and keep it after return. URI `mode=ro` callers are excluded. |
| `hermes_state.py::_connect_repair_durable` | Explicit repair/probe `db_path`; live profile is `db_path.parent`. | Connection factory only: live-profile callers must already hold the shared lease and retain it through all pragma application and close. Scratch/backup databases remain covered by the enclosing repair lease so recovery cannot interleave. |
| `hermes_state.py::apply_wal_with_fallback` | Connection supplied by the owning `SessionDB` construction or repair path; owning database path supplies profile identity. | Caller-owned span covering journal-mode probing/change, fallback, durability pragmas, and commit/checkpoint consequences. Never acquire from a connection alone. |
| `hermes_state.py::_set_journal_mode_no_wait` | Same owning path as the supplied connection. | Caller-owned span around the journal-mode change and validation. |
| `hermes_state.py::SessionDB._execute_write` | `self.db_path.parent` is the profile root. | Acquire before `BEGIN IMMEDIATE`; post-lock barrier check; hold through callback, commit or rollback, corruption recovery decision, and retry completion. This is the canonical coordinator for ordinary SessionDB, schema, and FTS writes. |
| `hermes_state.py::SessionDB._enter_fts_fail_open` | `self.db_path.parent`. | One shared span across `BEGIN IMMEDIATE`, stale-marker/trigger mutation, commit/rollback, and in-memory state publication. |
| `hermes_state.py::SessionDB._try_wal_checkpoint` | `self.db_path.parent`. | Shared span around checkpoint issue and result validation. |
| `hermes_state.py::SessionDB.close` | `self.db_path.parent`. | Shared span from final checkpoint decision through checkpoint completion and connection close. |
| `hermes_state.py::SessionDB.vacuum` | `self.db_path.parent`. | Shared span covering writer exclusion check, checkpoint, `VACUUM`, post-vacuum checkpoint, and close/error classification. |
| `hermes_state.py::SessionDB.purge_stale_tool_call_markers` | `self.db_path.parent`. | Mixed direct/delegated boundary: acquire before affected-row admission and retain one span across adjacent snapshot selection and `VACUUM INTO`, snapshot completion, the later `_execute_write` update commit/rollback, and result publication. The direct snapshot phase means this method cannot collapse to the `_execute_write` boundary. |
| `hermes_state.py::SessionDB.maybe_auto_prune_and_vacuum` | `self.db_path.parent`; sidecars use its resolved sessions directory argument. | A single shared span across prune transaction, canonical sidecar removals, checkpoint, vacuum, and final checkpoint; no gap between database deletion and file deletion. |
| `hermes_state_search.py::SessionSearchMixin.optimize_fts_storage` | Inherited `SessionDB.self.db_path.parent`. | Shared span across marker seeding, FTS rebuild/demote/optimize chunks, schema changes, checkpoint, vacuum, settle transaction, and final metadata publication. Nested `_execute_write` calls must not release the outer lease between phases. |
| `hermes_state_search.py::SessionSearchMixin._demote_legacy_fts_to_trash` | Inherited `SessionDB.self.db_path.parent`; caller-owned by `optimize_fts_storage`. | Multi-phase leaf: the outer optimizer must already hold the lease before entry and retain it continuously across the `_execute_write` staging commit/rollback, direct `self._conn` FTS schema recreation, direct commit, and result. Static call ownership forbids use outside `optimize_fts_storage`. |
| `hermes_state_search.py::SessionSearchMixin._fts_cjk_reset_if_stale` | Inherited `SessionDB.self.db_path.parent`; caller-owned by `optimize_fts_storage`. | Multi-phase leaf: the outer optimizer must already hold the lease before entry and retain it continuously across stale-marker/drop work in `_execute_write`, direct `self._conn` CJK schema recreation, direct commit, and result. Static call ownership forbids use outside `optimize_fts_storage`. |
| `hermes_state_search.py::SessionSearchMixin.optimize_fts` | Inherited `SessionDB.self.db_path.parent`. | Direct public writer: acquire before the first table-existence decision and hold across every FTS `optimize` command and the enclosing caller's commit/VACUUM boundary. Its current `SessionDB.vacuum` caller does not make the method itself safe for future direct callers. |
| `hermes_state_search.py::SessionSearchMixin.rebuild_fts` | Inherited `SessionDB.self.db_path.parent`. | Direct public writer: acquire before `fts_rebuild_admission`, then hold across table probes, every FTS `rebuild`, each commit/rollback, and release of both the SQLite and rebuild locks. This is called directly from gateway transcript-corruption recovery and SessionDB runtime corruption recovery, not from `_execute_write`. |
| `hermes_state_search.py::SessionSearchMixin._merge_fts_incrementally` | Inherited `SessionDB.self.db_path.parent`. | Direct writer: shared span across table probes, persisted `usermerge` configuration, every bounded `merge` command, implicit transaction completion, and in-memory floor publication. Callers may hold a wider `optimize_fts_storage` span but the helper cannot assume one. |
| `hermes_state.py::_exclusive_repair_db_guard` | Explicit fixed `db_path`; profile is `db_path.parent`. | Shared span from before the writable guard open through `BEGIN EXCLUSIVE`, the complete guarded repair/probe, rollback, and close. |
| `hermes_state.py::preflight_db_writability` | Explicit `db_path`; profile is `db_path.parent`. | Shared span across owner/mode/free-space checks and every probe-file create/fsync/unlink operation. The Task 12 cleanup's own admission remains separately read-only and must not call this mutating probe. |
| `hermes_state.py::_db_opens_cleanly` | Explicit `db_path`; profile is `db_path.parent`. | Shared span across the rolled-back write-health probe and any nested schema/FTS effects. Although logical rows roll back, the write transaction and journal effects make this a mutation boundary. |
| `hermes_state.py::_live_writer_holds_db` | Explicit `db_path`; profile is `db_path.parent`. | Shared span around the write-lock probe (`BEGIN IMMEDIATE`/rollback); this cannot run unleased merely because it normally rolls back. |
| `hermes_state.py::repair_state_db_schema` | Explicit `db_path`; profile is `db_path.parent`. | Public outer coordinator: one shared span from admission and repair-attempt claiming through backup, guarded repair, verification, ledger/backup retention updates, and final close. |
| `hermes_state.py::_repair_state_db_schema_locked` | Explicit `db_path`; profile is `db_path.parent`. | One shared span across backup, scratch database creation, strategy selection, live-database mutation/replacement, journal restoration, verification, ledger publication, and cleanup. |
| `hermes_state.py::_backup_db_file` | Explicit `db_path`; malformed backup is adjacent to the live database. | Caller-owned repair span across identity/holder checks, SQLite snapshot creation, verification, permissions, publication, and close. |
| `hermes_state.py::_copy_database_snapshot` | Explicit source/destination paths supplied by repair. | Caller-owned repair span across both opens, backup completion, deadline checks, commits/closes, and destination verification. |
| `hermes_state.py::_record_repair_outcome` | Ledger is derived from the fixed `db_path`. | Caller-owned repair span through categorical ledger publication and durability. |
| `hermes_state.py::_prune_malformed_backups` | Adjacent backups derived from fixed `db_path`. | Caller-owned repair span across complete evidence enumeration and exact stale-backup removal. |
| `hermes_state.py::_unlink_db_triple` | Fixed database path plus its WAL/SHM names. | Caller-owned repair span across identity proof and exact database/WAL/SHM unlink operations. |
| `hermes_state.py::quarantine_zeroed_state_db` | Fixed `state.db` path and adjacent quarantine name. | Shared span across zeroed-file revalidation, no-replace quarantine publication, directory durability, and error cleanup. |
| `hermes_state.py::_run_repair_strategies` | Explicit `db_path`; profile is `db_path.parent`. | Caller-owned repair span across FTS rebuild, `REINDEX`, writable-schema repair, `VACUUM`, and closes. |
| `hermes_state.py::_restore_journal_mode_after_repair` | Explicit `db_path`; profile is `db_path.parent`. | Caller-owned repair span across journal-mode restoration and validation. |
| `hermes_state.py::SessionDB._remove_session_files` | `sessions_dir` is supplied by the outer deletion API and must resolve to the same profile as `self.db_path.parent`. | Never lease independently. It must execute inside the outer delete/prune span and cover no-follow inventory/revalidation plus every JSON, legacy JSONL, and request-dump removal. |
| `hermes_state.py::SessionDB.delete_session` | `self.db_path.parent`; optional `sessions_dir` must be proven same-profile. | One shared span across canonical delegate discovery, database transaction, commit, and all root/delegate sidecar removals. |
| `hermes_state.py::SessionDB.delete_session_if_empty` | Same as `delete_session`. | One shared span across empty revalidation, transaction, commit, and sidecar removal. |
| `hermes_state.py::SessionDB.delete_sessions` | Same as `delete_session`. | One shared span across complete batch transaction and all sidecar removals; no per-root lease gaps. |
| `hermes_state.py::SessionDB.delete_empty_sessions` | Same as `delete_session`. | One shared span across selection revalidation, database transaction, and every sidecar removal. |
| `hermes_state.py::SessionDB.prune_sessions` | Same as `delete_session`. | One shared span across selector revalidation, transaction, commit, and every sidecar removal. |
| `gateway/platforms/api_server.py::APIServerAdapter._handle_create_session` | Cached SessionDB opened for the adapter's profile-aware active scope. | The `_do_create` mutation is already one `_execute_write` callback; its future lease is the exact `_execute_write` span. Handler auth and input parsing remain outside the lease. |
| `gateway/session.py::SessionStore._ensure_loaded_locked` | Two domains can differ: fixed launch-time `self.sessions_dir` and dynamically context-scoped `self._db.db_path.parent`. | Authority must be acquired before `self.sessions_dir.mkdir`. Prove both domains are the same, disable the cross-profile mirror/import, or acquire both profile authorities in canonical resolved-path order and hold them through legacy import, stale-route repair, any DB commit, JSON publication, and parent-directory fsync. |
| `gateway/session.py::SessionStore._persist_routing_data` | Two domains can differ in multiplexed production: `_db` resolves dynamically from the current context-local profile, while `self.sessions_dir` remains fixed to the launch configuration. | Prove equality or acquire both profile authorities in canonical resolved-path order before either mutation. Hold through generation ordering, DB commit, optional mirror temp-file fsync/replace, explicit mirror parent-directory fsync, and result publication. Alternatively suppress the mirror whenever domains differ. |
| `gateway/session.py::SessionStore._save_sessions_json` | Fixed launch-time `self.sessions_dir`, not necessarily the dynamically active DB profile. | Caller must already hold authority for that fixed filesystem profile; retain it from temp creation through JSON/file fsync, atomic replace, cleanup, and explicit parent-directory fsync. |
| `gateway/shutdown_flush.py::_get_flush_dir` | Active context-local `get_hermes_home()/pending_messages`. | This helper already mutates with `mkdir` and `chmod`; its caller must acquire/check the active profile before entry and retain authority afterward. It cannot be used as a supposedly read-only profile resolver. |
| `gateway/shutdown_flush.py::flush_pending_to_file` | Active context-local pending-message profile. | Acquire/check before `_get_flush_dir`; hold across complete iteration, every `_write_payload` publication, every parent-directory fsync, and categorical completion. |
| `gateway/shutdown_flush.py::spool_dropped_transcript_message` | Active context-local pending-message profile. | Acquire/check before `_get_flush_dir`; hold through payload publication and parent-directory fsync. |
| `gateway/shutdown_flush.py::flush_agent_history_to_file` | Active context-local pending-message profile. | Acquire/check before `_get_flush_dir`; hold through transcript normalization, payload publication, and parent-directory fsync. |
| `gateway/shutdown_flush.py::_write_payload` | Already-resolved `flush_dir`; owning profile authority comes from the outer writer. | Caller-owned span across atomic JSON publication and the existing explicit directory fsync. This leaf is too late to authorize `_get_flush_dir`'s preceding mutations. |
| `gateway/shutdown_flush.py::drain_transcript_spool` | Active spool profile plus the profile behind the replay callback, which is not encoded in the callback signature. | The caller must prove same-profile replay or supply/acquire both authorities in canonical resolved-path order before `_get_flush_dir`; hold through replay commit and exact spool unlink. |
| `gateway/shutdown_flush.py::recover_pending_to_db` | Active spool profile plus supplied/default `session_db.db_path.parent`; an arbitrary supplied handle can differ. | Prove equality or acquire both authorities in canonical resolved-path order before `_get_flush_dir`; hold per recovery unit through append commit and spool unlink, and include owned SessionDB open/close. |
| `run_agent.py::AIAgent._save_session_log` | `agent.logs_dir`, initialized in `agent/agent_init.py` from active context-local `get_hermes_home()/sessions`. | Shared span from target derivation and existing-file guard through redaction, atomic JSON publication, and durability. Barrier check must precede reading an existing snapshot used to authorize overwrite. |
| `agent/agent_init.py::init_agent` | Active context-local `get_hermes_home()/sessions`; ephemeral sessions skip directory creation. | For durable sessions, acquire/check the profile before `logs_dir.mkdir` and hold through directory creation/mode validation and publication of the resolved sink on the agent. Later snapshot/request-dump writes acquire their own complete spans. |
| `agent/agent_runtime_helpers.py::dump_api_request_debug` | `agent.logs_dir`, same active profile sessions directory. | Shared span from persistence-policy/profile resolution and barrier check through redaction, atomic request-dump publication, and durability. |
| `hermes_cli/backup.py::_safe_copy_db` | Caller supplies source/destination; when source is a profile `state.db`, source profile is `src.parent`. | Shared source lease from before the read connection/identity proof through SQLite backup completion and source close. Destination backup state is not a live-profile writer unless the destination itself is a profile database. |
| `hermes_cli/backup.py::_safe_restore_db` | Caller supplies live destination; profile is `dst.parent` when destination is `state.db`. | Shared lease (or the cleanup's already-held exclusive lease) across checkpoint, backup into live DB, permissions, fallback decision, sidecar handling, replacement, verification, and close. No fallback replacement may occur after lease release. |
| `hermes_cli/doctor.py::run_doctor` | Local `hermes_home/state.db`. | Read-only reporting stays outside. The rolled-back FTS write-health probe, `--fix` repair, and WAL checkpoint each require a shared lease covering their complete mutation/repair span. |
| `hermes_cli/update_cmd.py::_clear_stale_sqlite_sidecars` | Sidecar names are derived only from the supplied live `state.db` path. | Never acquire independently. It must remain inside `_restore_state_db_from_snapshot`'s proven live-profile span across exact WAL/SHM/journal identity validation and removal. |
| `hermes_cli/update_cmd.py::_restore_state_db_from_snapshot` | Both update paths supply the active profile's `state.db` plus the updater-created snapshot; profile root is the proven `state_path.parent`. | One shared span from holder/admission checks through sidecar removal, `shutil.copy2` overwrite of live `state.db`, integrity verification, failure classification, and return. The lease must begin before the point-in-time holder check so no cooperating writer can start afterward. |

### Leaves owned by the canonical coordinators

The following SQL-bearing leaves do not form independent acquisition points.
They must stay dynamically enclosed by either `SessionDB.__init__`,
`SessionDB._execute_write`, or the repair coordinator above:

- `SessionDB._store_system_prompt`, `_delete_unreferenced_system_prompts`,
  `_ensure_fts_cjk_schema`, `_drop_fts_triggers`, `_ensure_fts_schema`,
  `_check_transcript_write_guards`, `_insert_message_rows`, and
  `_record_model_usage`;
- `_delete_delegate_children`;
- every mutation helper in `hermes_state_schema.py`; and
- every callback-scoped FTS mutation helper in `hermes_state_search.py` that
  is dynamically passed to `_execute_write`. The six direct/multi-phase
  exceptions are separately frozen above: `optimize_fts_storage`,
  `optimize_fts`, `rebuild_fts`, `_merge_fts_incrementally`,
  `_demote_legacy_fts_to_trash`, and `_fts_cjk_reset_if_stale`. The final two
  are caller-owned leaves whose only permitted caller is
  `optimize_fts_storage`; their direct post-callback phases make release at
  the nested `_execute_write` return unsafe.

## Call-site inspection results

- `hermes_cli/`, `gateway/`, and `tui_gateway/` ordinary session mutations use
  `SessionDB` public methods and therefore converge on `SessionDB._execute_write`.
  The API-server create endpoint is the one reviewed direct SQL callback and it
  still executes through `_execute_write`.
- SessionDB connection creation is centralized by `_connect_tracked_db` for
  ordinary state; `_connect_repair_durable` is limited to repair/probe paths.
  Read pools and statistics use URI `mode=ro` and are excluded.
- Schema mutation is concentrated in `hermes_state_schema.py` during writable
  construction or an already-admitted write callback. FTS background mutation
  is mostly routed through `_execute_write`; the static guard now identifies
  direct `self._conn` mutation separately so `optimize_fts`, `rebuild_fts`,
  `_merge_fts_incrementally`, the two optimizer-owned multi-phase leaves, and
  future direct writers cannot be masked by the callback mapping. The guard
  also freezes the two leaves' exact caller so a new public call site is a
  review STOP.
- The active sessions-directory writers found by search are the opt-in JSON
  snapshot, request-debug dump, and gateway `sessions.json` mirror. No active
  canonical writer for a per-session `.jsonl` file under the sessions directory
  exists at this source revision; `.jsonl` remains a legacy deletion grammar.
- Pending-message/transcript-spool JSON is also transcript-bearing profile
  state and is included even though it lives under `pending_messages/`, not
  `sessions/`.

## Read-only and unrelated exclusions

- `sqlite_source_id()` and `_parse_schema_columns()` use `:memory:` databases.
- `_get_read_conn`, stats/readiness/status/backup-integrity probes, and other URI
  `mode=ro` connections are read-only and intentionally excluded by the guard.
- `gateway.platforms.api_server.ResponseStore`, `projects.db`, `kanban.db`,
  delivery ledgers, shared metrics, browser credential copies, exports, and
  user-selected recovery outputs are separate stores or outputs. They are not
  canonical `state.db`/session-sidecar mutation paths and the cleanup lease must
  not serialize unrelated stores. A restore whose destination is a live
  profile `state.db` is explicitly included above.
- `tui_gateway` spawn-tree JSONL, MOA traces, trajectories, crash logs, caches,
  and configuration files are not canonical session transcript sidecars used
  by exact cleanup. They remain out of this lease to preserve REQ-M2 profile and
  service isolation.

## Resolved implementation rulings

1. **Session-sidecar grammar:** the current remover's `{session_id}.json` /
   `{session_id}.jsonl` grammar and the current writer's
   `session_{session_id}.json` grammar are both authoritative existing shapes.
   The mismatch is a confirmed cleanup defect, not permission to choose one.
   Tasks 2–5 must lease and inventory both forms plus
   `request_dump_{session_id}_*.json`; live cleanup remains STOP until the exact
   plan proves absence/quarantine for every accepted form.
2. **Atomic JSON durability:** `utils.atomic_json_write` fsyncs the staged file
   before `atomic_replace` but does **not** fsync the parent directory after
   publication. Therefore it is not, by itself, the end of the required lease
   span. Each canonical profile writer must retain the shared lease through an
   explicit post-publication parent-directory fsync (or a future reviewed
   helper that provides the same guarantee). `shutdown_flush._write_payload`
   already performs that extra fsync; the other writers do not yet.
3. **Generic repair/backup paths:** a generic path receives profile-maintenance
   semantics only after code-owned resolution proves the basename is exactly
   `state.db` and its parent is the canonical active/explicit Hermes profile
   root. Scratch databases, snapshots, exports, and unrelated SQLite stores do
   not acquire a profile lease independently, but remain inside an enclosing
   live-profile repair span when their mutation participates in that repair.
   Caller, environment, model, plugin, or database content cannot select the
   profile root or cleanup authority.

Any newly discovered canonical writer, direct `self._conn` FTS mutation, or
copy/unlink replacement of a live profile database is a STOP until this audit
and guard are updated and independently reviewed.

## Machine-checked critical boundary contracts

The static test parses this block as JSON and requires exact equality; prose
mentions elsewhere do not satisfy the contract gate.

<!-- TASK12_BOUNDARY_CONTRACTS_START -->
{
  "gateway/session.py::SessionStore._ensure_loaded_locked": {
    "profile_domains": ["fixed_launch_sessions_dir", "dynamic_session_db"],
    "lease_span": "before_mkdir_then_prove_same_or_ordered_dual_authority_through_reconcile_persistence"
  },
  "gateway/session.py::SessionStore._persist_routing_data": {
    "profile_domains": ["dynamic_session_db", "fixed_launch_sessions_dir"],
    "lease_span": "prove_same_or_ordered_dual_authority_through_db_commit_mirror_dir_fsync"
  },
  "gateway/session.py::SessionStore._save_sessions_json": {
    "profile_domains": ["fixed_launch_sessions_dir"],
    "lease_span": "caller_owned_fixed_profile_through_replace_and_parent_dir_fsync"
  },
  "gateway/shutdown_flush.py::_get_flush_dir": {
    "profile_domains": ["active_context_profile_pending_messages"],
    "lease_span": "caller_owned_before_mkdir_chmod"
  },
  "gateway/shutdown_flush.py::flush_pending_to_file": {
    "profile_domains": ["active_context_profile_pending_messages"],
    "lease_span": "before_get_flush_dir_through_all_payload_parent_dir_fsyncs"
  },
  "gateway/shutdown_flush.py::spool_dropped_transcript_message": {
    "profile_domains": ["active_context_profile_pending_messages"],
    "lease_span": "before_get_flush_dir_through_payload_parent_dir_fsync"
  },
  "gateway/shutdown_flush.py::flush_agent_history_to_file": {
    "profile_domains": ["active_context_profile_pending_messages"],
    "lease_span": "before_get_flush_dir_through_payload_parent_dir_fsync"
  },
  "gateway/shutdown_flush.py::recover_pending_to_db": {
    "profile_domains": ["active_spool_profile", "supplied_or_default_session_db_profile"],
    "lease_span": "prove_same_or_ordered_dual_authority_through_db_commit_and_spool_unlink"
  },
  "hermes_cli/update_cmd.py::_restore_state_db_from_snapshot": {
    "profile_domains": ["proven_live_state_db_profile"],
    "lease_span": "before_holder_check_through_sidecar_unlink_copy2_integrity_verification"
  },
  "hermes_state.py::SessionDB.purge_stale_tool_call_markers": {
    "profile_domains": ["session_db_profile"],
    "lease_span": "before_affected_row_admission_through_vacuum_into_update_commit_and_result"
  },
  "hermes_state_search.py::SessionSearchMixin._demote_legacy_fts_to_trash": {
    "profile_domains": ["session_db_profile"],
    "lease_span": "caller_owned_optimize_fts_storage_across_callback_and_direct_schema_commit"
  },
  "hermes_state_search.py::SessionSearchMixin._fts_cjk_reset_if_stale": {
    "profile_domains": ["session_db_profile"],
    "lease_span": "caller_owned_optimize_fts_storage_across_callback_and_direct_schema_commit"
  },
  "hermes_state_search.py::SessionSearchMixin.optimize_fts": {
    "profile_domains": ["session_db_profile"],
    "lease_span": "before_table_probe_through_commands_and_transaction_completion"
  },
  "hermes_state_search.py::SessionSearchMixin.rebuild_fts": {
    "profile_domains": ["session_db_profile"],
    "lease_span": "before_rebuild_admission_through_each_commit_or_rollback"
  }
}
<!-- TASK12_BOUNDARY_CONTRACTS_END -->
