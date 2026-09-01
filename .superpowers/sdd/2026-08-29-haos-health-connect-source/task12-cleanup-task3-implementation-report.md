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

## Fix round 1 — 2026-09-01

### Status and authority

`DONE_WITH_CONCERNS` on Darwin after addressing every P2 and P3 in
`task12-cleanup-task3-review.md`. This appendix supersedes the original
retirement description above: successful retirement no longer performs a
path-only unlink. Native Linux remains unavailable, so cross-platform
completion is still intentionally not claimed.

The controller resolved P2-1's frozen-audit mismatch by authorizing
`SessionDB.prune_empty_ghost_sessions` as an audited Task 3 writer and
authorizing the exact audit artifact/static test changes needed to freeze it.
The complete frozen direct-writer file list remains:

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

This round changed only the newly authorized ghost boundary/audit, the
previously approved Task 3 barrier and writer files, and their exact existing
tests/report. No dependency, live profile, network, external service,
provider, push, merge, deployment, production cleanup, or Task 4 action was
performed. Every behavioral test used synthetic temporary profiles.

### Review-finding disposition

1. **P2-1 — fixed.** `prune_empty_ghost_sessions` now owns one outer shared
   profile lease from before selection/SQL through commit, every sidecar
   removal, and sidecar-directory durability. The audit lists it as a direct
   canonical file writer. Its AST gate now preserves any SessionDB method that
   delegates SQL to `_execute_write` and then calls `_remove_session_files` as
   an outer coordinator, preventing the same post-callback lease gap from
   collapsing into `_execute_write` again. A deterministic two-thread test
   proves an exclusive maintainer cannot publish during the sidecar phase.
2. **P2-2 — fixed.** Retirement atomically moves the fixed descriptor-relative
   barrier name to a random retired name with Darwin `renameatx_np(...,
   RENAME_EXCL)` or Linux `renameat2(..., RENAME_NOREPLACE)`. It proves the
   moved name is the exact held inode, proves the fixed name absent, fsyncs the
   held profile directory, and repeats both proofs. It never path-unlinks the
   retired object. Ambiguity, ordinary durability failure, or cancellation
   after the move conservatively republishes the fixed nonce-bound blocker.
   The small closed-schema retired record intentionally remains as quarantine;
   deleting it safely is not smuggled-in Task 4 cleanup logic.
3. **P2-3 — fixed.** Ordinary `Exception` handling once again logs,
   preserves the spool file, and continues. The later `BaseException` branch
   closes an owned database and re-raises cancellation/interrupt. A focused
   transient-append test locks the REQ-M13 behavior while the existing
   `KeyboardInterrupt` test locks cancellation custody.
4. **P2-4 — fixed.** Session deletion APIs authorize only
   `self.db_path.parent`. A supplied sidecar sink must be the exact lexical
   `profile/sessions` Path and, when present, a no-follow directory rather than
   an alias or replacement symlink. Unrelated-profile, alias-spelling, and
   replaced-canonical-sink tests prove the row and sidecar both remain on
   refusal. No second profile, environment fallback, current-directory
   fallback, or weakened profile validation was introduced.
5. **P2-5 — fixed.** Zeroed-database quarantine fsyncs the database parent
   after the complete database/WAL/SHM rename set and before releasing the
   shared lease. Directory-fsync failure is the fixed `unsafe_profile_state`
   category and preserves the moved evidence.
6. **P3-1 — fixed.** Barrier descriptor custody is centralized in a one-shot
   close helper. `EINTR` is never retried, so a reused descriptor cannot be
   closed, and close failure cannot replace the already selected categorical
   result.
7. **P3-2 — fixed.** Focused tests now cover read-only construction/read/close
   without barrier authority, retirement directory-fsync failure and
   republish, publication and retirement `BaseException`, multi-root
   acquisition cancellation/reverse release, held-inode quarantine identity,
   and close-time descriptor reuse.

### Calibrated RED-to-GREEN record

- Ghost audit/span RED: `2 failed` (outer coordinator collapsed to
  `_execute_write`; exclusive publication entered the sidecar gap). GREEN:
  `2 passed`.
- Barrier review RED: seven new tests produced the expected `3 failed, 4
  passed`; the failures were the final replacement window, cancellation after
  retirement, and raw close error. GREEN with the existing exact-nonce test:
  `8 passed`.
- Spool compatibility RED: ordinary append failure escaped while the existing
  cancellation test passed (`1 failed, 1 passed`). GREEN: `2 passed`.
- Same-profile authority RED: all three unrelated/alias/replacement cases
  deleted without refusal (`3 failed`). GREEN plus the adjusted canonical
  writer tests and ghost span: `6 passed`.
- Quarantine durability RED: success made no directory-fsync call and injected
  failure did not raise (`2 failed`). GREEN: `2 passed`.

No failed test was deleted, weakened, skipped, xfailed, or given a retry.
`HERMES_TEST_FILE_RETRIES=0` was used for the parallel matrices.

### Final verification evidence

- Final integrated focused/audit gate: `112 passed` across
  `tests/test_session_maintenance_lock.py`,
  `tests/test_profile_mutation_boundary_audit.py`,
  `tests/gateway/test_shutdown_flush.py`, and
  `tests/test_zeroed_state_db.py`.
- Exact maintenance/static parallel gate: `100 passed` (`94` maintenance,
  including cross-process same/unrelated-profile stress; `6` frozen audit).
- Gateway writer matrix: `113 passed` across seven exact files.
- State/repair/FTS matrix: `458 passed, 1 failed` across twenty files. The sole
  failure is the unchanged read-only FTS context trace baseline already named
  in the original report.
- Agent/request/snapshot matrix: `434 passed, 1 failed`; the sole failure is
  the unchanged missing optional `anthropic` dependency baseline, with the
  separately known `_BarrierDB.flush_token_counts` warning. No dependency was
  installed.
- Backup/doctor/focused-update files: `214 passed, 1 skipped`. An additional
  broad `tests/hermes_cli/test_cmd_update.py` characterization produced `31
  passed, 8 failed`; those unrelated updater fixture/fleet failures do not
  exercise this diff and were neither fixed nor hidden.
- Canonical deletion compatibility: `24 passed` across empty-session hygiene
  and lifecycle status. A sequential mixed-file characterization exposed one
  pre-existing HERMES_HOME test-order leak; the exact per-file parallel gateway
  matrix above passed `113/113` under clean environments.
- Ruff lint, `compileall`, `ty check hermes_state_maintenance.py`, focused Ruff
  format, and `git diff --check`: pass.
- Dependency-manifest/lock delta: empty. Added-line credential, PHI,
  prohibited-output, deleted-test, skip, xfail, and retry scans: no finding.
- No repository-local Semgrep rule set exists; registry rules would require
  prohibited network access, so no registry scan was run. The frozen AST audit
  and local diff scans passed.

### Native Linux gate still unrun

`uname -s` is Darwin. Run exactly this command on native Linux with zero
retries before claiming cross-platform completion:

```sh
HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh tests/test_session_maintenance_lock.py tests/test_profile_mutation_boundary_audit.py -q
```

### Final security/control review

This remains high-risk crash-recovery/data-integrity work under the approved
spec, STRIDE pass, and rollback plan; no new ADR decision was made. Exact
lease typing/liveness, fixed-schema nonce validation, no-follow descriptor
authority, atomic no-replace publication/retirement, before/after inode proof,
directory durability, fixed path-free categories, same-profile sidecar
binding, reverse lease release, synthetic owner-only profiles, and static
writer coverage are satisfied. Authz/RLS are not request boundaries in this
local coordination layer; input/path validation and secret handling are
satisfied. No dependency or secret was added. Native Linux evidence and
independent re-review remain the only completion concerns.

---

## Fix round 2 — 2026-09-01

### Status and scope

`DONE_WITH_CONCERNS` on Darwin after addressing P2-A, P2-B, and P3-A from
`task12-cleanup-task3-re-review-1.md`. The only completion concern is the
unchanged native-Linux gate below; independent re-review is still required.

This round changed only `hermes_state_maintenance.py`, `hermes_state.py`,
`tests/test_session_maintenance_lock.py`, `tests/test_hermes_state.py`, and
this report. Those are the already-approved Task 3 barrier/writer files and
their exact tests/report. The frozen direct-writer file list and both approved
scope rulings recorded above are unchanged. No dependency, live profile,
network, external service, provider, process, push, merge, deployment,
production cleanup/deletion, or Task 4 action was performed. Every behavioral
test used synthetic temporary profiles.

### Finding disposition

1. **P2-A — fixed.** Retirement now records conservative publication
   uncertainty before entering the no-replace rename syscall. Any exception
   after that point attempts durable no-replace republication of the fixed,
   exact-nonce blocker. This covers a catchable signal delivered after the
   kernel move but before the helper returns, without relying on a Python
   assignment after the mutating call. The retired exact held inode remains
   quarantined; no path-only unlink or Task 4 cleanup was added.
2. **P2-B — fixed.** Every audited SessionDB deletion/prune API that can remove
   transcript sidecars now acquires the profile shared lease first, opens the
   fixed `sessions` name no-follow through the held profile descriptor, and
   retains that exact directory descriptor/identity for the complete DB and
   filesystem mutation span. The held and fixed-name identities are checked
   after admission and again immediately before database mutation. Sidecar
   inventory, unlink, and directory fsync are descriptor-relative. The absent
   case records absence through the held profile descriptor and refuses a
   later object instead of following it. Descriptor close failure is
   best-effort and cannot replace an already selected categorical outcome.
3. **P3-A — fixed.** New focused tests inject cancellation immediately after
   the real kernel rename, replace the canonical sessions sink both immediately
   after lease admission and after its descriptor was captured, and record
   reverse release of two held leases when the second barrier check is
   interrupted. The replacement tests prove the database row, original
   sidecar, and unrelated-profile sidecar all remain unchanged on refusal.

The descriptor identity deliberately excludes mutable directory link count:
an existing two-sidecar test exposed that Darwin changes directory link
metadata during ordinary file removal. Type/owner/link safety is still checked
at every validation, while stable device/inode/owner/group/mode evidence binds
the retained authority. This preserves ordinary multi-file cleanup semantics
without weakening the replacement proof.

### Calibrated RED-to-GREEN record

- Immediate post-rename cancellation RED: the real no-replace rename completed,
  the shim raised `KeyboardInterrupt` before returning, and the fixed barrier
  was absent (`1 failed`). GREEN: cancellation propagated, the fixed blocker
  was republished, the retired name retained the original inode, and a later
  shared writer observed recovery-required (`1 passed`; the neighboring
  retirement/cancellation group also passed).
- Post-admission sidecar replacement RED: deletion followed the replacement
  symlink, committed the row removal, and removed the unrelated same-named
  sidecar (`1 failed`). GREEN: exact post-lock descriptor admission rejected
  the replacement before database mutation.
- Post-descriptor-validation replacement RED: a deterministic `_execute_write`
  seam replaced the fixed name after descriptor capture; the old code deleted
  the row (`1 failed`). GREEN: immediate held/fixed identity revalidation
  rejected the replacement and preserved both profiles (`1 passed`).
- Existing-writer compatibility RED: the first full state run produced two
  in-scope multi-sidecar/auto-prune failures because mutable directory link
  metadata was treated as immutable identity (`247 passed, 3 failed`, including
  the unchanged FTS baseline). GREEN: both focused failures passed, followed by
  `249 passed, 1 failed`; the sole final failure is the unchanged read-only FTS
  trace baseline already recorded above.
- Reverse-release coverage passed on first execution because the underlying
  `finally` already released all held leases in reverse; the new test makes the
  two-held-lease order directly observable.

No failed test was deleted, weakened, skipped, xfailed, or given a retry.

### Final verification evidence

- Repository-runner maintenance/static gate with retries disabled: `102
  passed` (`96` maintenance, including cross-process same/unrelated-profile
  stress, plus `6` frozen audit).
- Explicit cross-process profile-scoping stress rerun: `1 passed`.
- Final focused race and reverse-release selection: `5 passed`.
- Full `tests/test_hermes_state.py`: `249 passed, 1 failed`; only the unchanged
  `TestFTS5Search.test_search_projection_skips_context_enrichment_queries`
  trace baseline failed.
- Exact changed-writer matrix: `183 passed, 1 failed`. The failure was the
  previously recorded combined-file HERMES_HOME order leak in
  `test_session_store_default_db_uses_runtime_hermes_home`; its exact file
  passed `10/10` in a clean isolated rerun.
- Canonical deletion/lifecycle compatibility subset: `42 passed`.
- An additional broad deletion-caller characterization returned `660 passed,
  4 failed`. The four failures were unrelated TUI agent-build timing (two), an
  existing non-`0700` synthetic profile fixture, and model-options state. None
  exercises this round's barrier or sidecar authority, and none was fixed or
  hidden.
- Ruff lint for all four code/test files, Ruff format for the owned maintenance
  module/focused test, `compileall`, `ty check hermes_state_maintenance.py`,
  focused type review of the new `hermes_state.py` authority block, and `git
  diff --check`: pass. The full legacy `hermes_state.py` type run retains 85
  diagnostics outside the new authority block.
- Dependency-manifest/lock delta: empty. Added-line credential, PHI,
  prohibited-output, deleted-test, skip, xfail, and retry scans: no finding.
  No repository-local Semgrep rules exist, and prohibited network access
  prevents registry-rule retrieval; the frozen AST/security scans passed.

### Native Linux gate still unrun

`uname -s` is Darwin. Run exactly this command on native Linux with zero
retries before claiming cross-platform completion:

```sh
HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh tests/test_session_maintenance_lock.py tests/test_profile_mutation_boundary_audit.py -q
```

### Security/control review

This remains high-risk crash-recovery/data-integrity work under the approved
spec, STRIDE pass, and rollback plan; no new ADR decision was made. Exact
lease type/liveness, fixed-schema nonce validation, no-follow
descriptor-relative authority, monotonic interruption recovery, atomic
no-replace retirement, same-profile sidecar binding, absent-sink refusal,
reverse lease release, fixed path-free categories, and synthetic owner-only
profiles are satisfied. Authz/RLS are not request boundaries in this local
coordination layer. Input/path validation and secret handling are satisfied.
No dependency or secret was added.
