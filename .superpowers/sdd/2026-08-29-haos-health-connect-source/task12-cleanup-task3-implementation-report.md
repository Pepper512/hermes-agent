# Task 12 cleanup — Task 3 implementation report

## Status

`DONE_WITH_CONCERNS` on Darwin. Task 3's crash-persistent recovery barrier and
the complete frozen writer instrumentation are implemented and locally
verified. Native Linux is not available on this host, so cross-platform
completion is intentionally not claimed.

The worktree remained pinned to the required starting commit
`deb7550e6e1b6b1c49e80b573736cfe01ec8b058` until the Task 3 commit. No live
profile, external service, provider, network, deployment, cleanup, deletion,
migration, push, merge, or dependency action was performed. All behavioral
tests used synthetic temporary profiles.

## Authority and resolved rulings

This is high-risk work because it changes crash recovery and every canonical
profile mutation boundary. The approved amendment, implementation plan,
STRIDE analysis, rollback guidance, frozen Task 1 audit, accepted Task 2
implementation, and Task 3 brief were read before implementation.

Two scope questions were raised before the affected edits and resolved by the
controller:

1. Task/spec commit `e9b42d50038e1eba2da4589842ab0eb3b0b53af3`
   confirms that the approved amendment controls over the narrower drafting
   phrase in the original plan. Every canonical database, schema, FTS,
   checkpoint, backup, restore, repair, sidecar, and request-dump mutation
   frozen by Task 1 must participate.
2. Task/spec commit `26d04dab734945dc473227c06f5e5c0c71b7c75a`
   permits `tests/conftest.py` to create/chmod only its hermetic synthetic
   `HERMES_HOME` as exact owner-only `0700`, and permits exact existing writer
   test doubles to expose an explicit synthetic `db_path`. It expressly
   forbids a production fallback, current-directory authority,
   environment-selected test-double authority, or weakened `Path`/`0700`
   validation. The implementation now requires an explicit `db_path` for a
   non-`None` SessionStore database double; an intentional `None` database
   leases only the fixed sessions profile.

No later spec/audit contradiction was found. No ADR was added: this task makes
no architecture choice beyond the already-approved barrier and lease design.

## Frozen audit membership

The exact Task 1 direct-writer files governed by the ruling are:

- `hermes_state.py`
- `hermes_state_schema.py`
- `hermes_state_search.py`
- `agent/agent_init.py`
- `agent/agent_runtime_helpers.py`
- `gateway/platforms/api_server.py`
- `gateway/session.py`
- `gateway/shutdown_flush.py`
- `run_agent.py`
- `hermes_cli/backup.py`
- `hermes_cli/doctor.py`
- `hermes_cli/update_cmd.py`

`hermes_state_schema.py` needs no direct edit because all audited schema leaves
remain dynamically enclosed by the newly leased writable `SessionDB`
constructor or `_execute_write`. `gateway/platforms/api_server.py` needs no
direct edit because its audited create callback remains enclosed by
`SessionDB._execute_write`. No writer absent from the frozen audit was changed.

## Implementation

### Recovery barrier

`hermes_state_maintenance.py` now provides exactly:

- `publish_recovery_barrier(exclusive_lease, operation_nonce)`
- `require_no_recovery_barrier(shared_lease)`
- `retire_recovery_barrier(exclusive_lease, operation_nonce)`

The fixed marker uses a closed V1 record with one exact 64-character lowercase
hex nonce and no target, transcript, SQL, identifier, or path data. Publication
uses an owner-only no-follow staged file, complete write, file `fsync`,
descriptor-relative hard-link no-replace publication, stage removal, profile
directory `fsync`, and exact reread. Reads are descriptor-relative/no-follow
and prove owner, regular-file type, `0600`, link count, size, stable inode
identity, schema, and nonce syntax before returning a categorical result.
Retirement accepts only the exact nonce and identity, unlinks under the held
exclusive lease, and fsyncs the directory. A post-unlink durability failure
attempts conservative republishing and returns only a fixed categorical error.

Barrier-present, malformed, substituted, disappeared-during-check, unsupported,
and unsafe-profile results are path-free categorical errors. Windows or hosts
without the reviewed POSIX primitives fail closed.

### Writer protocol

`_profile_state_mutation_scope` acquires unique profile shared leases in
deterministic canonical resolved-path order under one 60-second total budget,
checks every barrier only after every lease is held, spans the complete
mutation/durability operation, and releases in reverse order. Canonical
ordering does not replace the caller path used for authority, so a final
symlink remains invalid. A narrow same-inode/zero-size first-lock
initialization transition is retried within the original deadline; all owner,
type, mode, inode, and link-count checks remain fail-closed.

Every frozen coordinator now participates, including writable construction,
ordinary transactions, FTS direct/multiphase writers, checkpoints, close,
vacuum, prune/delete families, repair/probes/quarantine, gateway routing and
pending spools, optional snapshots and request dumps, durable agent log-root
creation, SQLite backup/restore, doctor checkpoint, and updater restoration.
Multi-profile gateway and recovery operations supply both explicit roots.
Generic database paths participate only when the code-owned live basename is
exactly `state.db`; unrelated scratch/output stores remain excluded. Required
sidecar publications/removals now include parent-directory fsync inside the
lease span.

No Task 4 selector, delete-plan, exact-batch deletion, cleanup transaction, or
live recovery logic was added.

### Test support

The shared hermetic `HERMES_HOME` and exact profile fixtures are owner-only
`0700`. Exact gateway database doubles now expose explicit synthetic
`db_path` values. These changes are confined to `tests/conftest.py` and the
existing tests for audited writers.

## TDD record

Strict RED-to-GREEN cycles were recorded as follows:

1. Barrier interface RED: the three required APIs were absent. GREEN after the
   closed-schema barrier implementation.
2. Already-open/writable-constructor RED: later writes and writable open did
   not refuse a durable barrier. GREEN after constructor and per-mutation
   instrumentation.
3. Direct state/FTS family RED: 12 failures and one pass. GREEN: 13/13.
4. Sidecar/request/snapshot family RED: eight failures. GREEN after complete
   lease spans and directory durability.
5. Dual-profile/update family RED: two genuine failures after correcting
   invalid synthetic fixtures. GREEN: 5/5.
6. Durable initialization RED: one failure. GREEN after the log-root span.
7. Shared existing-writer run RED: 57 `unsafe_profile_state` failures proved
   the suite-wide fixture was `0755`. The controller approved the exact
   `0700` fixture correction; production validation was not weakened.
8. Self-review RED: raw spelling acquired the dual roots in the wrong order
   through a symlinked parent. GREEN after canonical resolved-path ordering.
9. Self-review RED: a missing gateway fake `db_path` selected an environment
   fallback. GREEN after removing the fallback and requiring explicit
   authority; intentional `_db=None` remains fixed-profile-only.
10. Self-review RED: barrier disappearance after descriptor open leaked a raw
    filesystem exception. GREEN after categorical `unsafe_recovery_barrier`
    conversion.
11. Expanded-writer RED: concurrent first lock initialization could observe
    the same safe inode at sizes zero/nonzero and fail spuriously. A
    deterministic RED reproduced it. GREEN after a deadline-bounded retry of
    only that exact same-inode zero-size transition; ten repeated concurrent
    zeroed-DB runs and the final exact suite passed.

One process mistake occurred: a broad writer-family instrumentation draft was
briefly applied before its RED was captured. Those exploratory edits were
removed, the intended RED was run, and the changes were then reapplied to
GREEN before further work.

## Verification evidence

### Passing behavioral gates

- Final focused barrier plus frozen static audit:
  `91 passed, 0 failed` (`86` maintenance tests + `5` audit tests).
- The focused suite includes the eight-child cross-process stress proving a
  durable barrier blocks the same profile while every unrelated-profile child
  writes successfully.
- Final corrected existing-writer subset:
  `24 passed, 0 failed` across delete isolation, concurrent zeroed recovery,
  SessionStore prune, and stale-prune suites.
- Gateway writer matrix: initial `192 passed, 3 failed`; all three were exact
  missing-`db_path`/mode fixture mismatches, followed by `19 passed, 0 failed`
  in the corrected files. Earlier exact pending/session rerun: `71 passed`.
- Backup/doctor/update matrix: `211 passed, 0 failed, 1 skipped` across ten
  exact files.
- State/repair/FTS matrix: `460 passed, 3 failed` across twenty files. Two
  in-scope failures were corrected and the affected files then passed 5/5.
  The sole remaining failure is the unrelated read-only projection baseline
  named below.
- Agent/request-dump/snapshot matrix: `393 passed, 1 failed`; the sole failure
  is the separately identified missing optional dependency baseline below.
- Cross-process stress selected separately earlier: `1 passed`.

No failed test was deleted, weakened, skipped, xfailed, or retried.
`HERMES_TEST_FILE_RETRIES=0` was set on the parallel suite runs.

### Passing static/security gates

- Ruff lint across all 24 modified code/test files: pass.
- `compileall` across all 24 modified code/test files: pass.
- Ruff format check for the new/owned Task 3 implementation and focused test:
  pass.
- `ty check hermes_state_maintenance.py`: pass.
- `git diff --check`: pass.
- Dependency-delta check for Python and JavaScript manifests/locks: empty.
- Frozen static mutation-boundary audit: 5/5 pass; no unleased canonical
  writer discovered.
- Added-line credential, PHI, prohibited-output, deleted-test, skip, xfail,
  and retry scans: no finding (the word `FlakyDb` is a test class name, not a
  retry marker).
- No secret, token, PHI, raw identifier, transcript, prompt, SQL, response
  body, or live path was added to the barrier schema or its categorical
  errors.

## Known baseline/tooling concerns

1. `tests/test_hermes_state.py::TestFTS5Search::test_search_projection_skips_context_enrichment_queries`
   expects one traced read-only context query and observes zero. Task 3 does
   not change `search_messages` or the read projection path. This isolated
   baseline remains unfixed and unhidden.
2. `tests/run_agent/test_run_agent.py::TestAnthropicInterruptHandler::test_interruptible_anthropic_interrupt_never_closes_shared_client`
   fails because the optional `anthropic` package is absent. Per instruction,
   no dependency was installed and the failure was not hidden. The same file
   emits a pre-existing fake `_BarrierDB` thread warning because that fake
   lacks `flush_token_counts`; Task 3 does not change that test double.
3. Whole-file Ruff formatting would mechanically rewrite 22 legacy modified
   files with extensive pre-existing formatting debt. The new Task 3 module
   and focused test are format-clean; unrelated legacy code was not rewritten.
4. The full modified-production `ty` run reports 643 pre-existing diagnostics
   in legacy modules. The new maintenance module is type-clean. These baseline
   diagnostics were not expanded into out-of-scope refactoring.
5. `semgrep` is installed but there is no repository-local rule set. Registry
   rules would require prohibited network access, so no Semgrep registry scan
   was run. Exact local diff scans and the frozen AST writer audit passed.

## Native Linux gate not run

This host is Darwin. The controller must run this exact command on native
Linux before cross-platform completion is claimed:

```sh
HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh tests/test_session_maintenance_lock.py tests/test_profile_mutation_boundary_audit.py -q
```

## Security/control review

- **Spoofing/tampering:** held no-follow profile/lock/barrier descriptors,
  stable device/inode evidence, exact owner/mode/type/link/size checks,
  no-replace publication, exact nonce retirement, post-lock barrier checks,
  canonical multi-root ordering, and the frozen writer audit are satisfied.
- **Repudiation/information disclosure:** only fixed categorical errors are
  emitted by the barrier protocol; the marker carries no target or transcript
  data. No secret or PHI was added. Live operations were not performed.
- **Denial of service/elevation:** acquisition remains deadline-bounded,
  profile-scoped, and fail-closed; unrelated profiles are proven live under
  barrier stress; opaque lease types cannot be constructed by callers.
- **Authorization/input validation:** no request authorization boundary is
  changed. Operation nonce, lease type/liveness, exact `Path`, profile mode,
  barrier schema, and filesystem evidence are validated at their boundaries.
- **Dependencies/secrets:** no dependency was added and no secret was
  introduced or hardcoded. Authz and RLS are not applicable to this local
  profile coordination task.
- **Rollback:** do not roll back to a writer version that ignores a present
  barrier. First reconcile to accepted cleanup or verified restoration and
  retire the exact barrier under exclusive authority, per the approved
  amendment.

Independent review remains required, and native Linux verification is the one
platform gate still outstanding.
