# Plan D: Named Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** an export can be given a free-text name; its folder id is slugged from the name; the name is shown
everywhere an export appears and can be renamed later.

**Architecture:** a small `export/naming.py` module slugs names and reserves folder ids atomically; `run_export`
and augmented copies record `name` in `export.json`; `ExportInfo` carries it to the Train page; a PATCH route renames.
The web app shows `name` with the id beneath, falling back to the id.

**Tech Stack:** Python 3.11 / FastAPI / pydantic v2 / pytest (uv); React 18 + TypeScript + Tailwind + vitest.

**Spec:** `docs/superpowers/specs/2026-09-26-multi-object-labeling-and-export-selection-design.md` §11.

## Global Constraints

- Python only through uv (`uv run pytest`, `uv run ruff check server`). Web tools: `cd web && node
  node_modules/.bin/vitest run`, `node node_modules/.bin/tsc --noEmit`, `node node_modules/.bin/vite build` (there
  is no npm on this machine).
- Name: stripped, at most 80 characters, no control characters; empty after stripping = no name (null).
- Slug: lowercase; runs of characters outside `[a-z0-9._-]` become `-`; repeated `-` collapse; leading `-._` removed;
  at most 48 characters, then trailing `-._` removed. Empty slug → today's `YYYYmmdd-HHMMSS-xxxxxx` id.
- Folder ids are reserved with `mkdir()` (no `exist_ok`); a taken id tries `-2`, `-3`, ...
- A missing `name` key in an older `export.json` reads as null.
- Every control on the web pages lives in the right-hand sidebar.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01K7g9xYipT1aUyMQr6HKxE6`. Stage only your task's files.

---

### Task 1: Server — names, slugs, reservation, manifests, rename, listing

**Files:**
- Create: `server/vsp/export/naming.py`
- Modify: `server/vsp/export/runner.py`, `server/vsp/export/__init__.py` (export the new names if the package
  re-exports), `server/vsp/api/export.py`, `server/vsp/train/models.py`, `server/vsp/api/train.py`,
  `server/vsp/augment/derive.py`, `server/vsp/cli.py`
- Test: create `server/tests/test_export_naming.py`; extend `server/tests/test_export_runner.py`,
  `server/tests/test_api_export.py`, `server/tests/test_train_models.py`, `server/tests/test_api_train.py`,
  `server/tests/test_augment_derive.py`, `server/tests/test_train_cli.py`

**Interfaces:**
- Produces:
  - `naming.clean_name(name: str | None) -> str | None` (raises `BadRequest`)
  - `naming.slug(name: str | None) -> str`
  - `naming.reserve_export_dir(exports_root: Path, name: str | None) -> tuple[str, Path]`
  - `run_export(..., name: str | None = None)` → result dict gains `"name"`
  - `manifest_for(..., name: str | None = None)` → manifest gains `"name"`
  - `ExportInfo.name: str | None = None`, in `to_json()`
  - `rename_export(data_root, export_id, name) -> ExportInfo`
  - HTTP: `POST /api/projects/{id}/export` body `name: str | null`; `PATCH /api/exports/{export_id}` body
    `{name: str | null}` → the export's `to_json()`.

- [ ] **Step 1: Write the failing tests**

`test_export_naming.py`:

```python
import threading

import pytest
from vsp.export.naming import clean_name, reserve_export_dir, slug
from vsp.labeling.errors import BadRequest


@pytest.mark.parametrize("name, expected", [
    ("Cups v2", "cups-v2"),
    ("  Tassen / Schalen (final)  ", "tassen-schalen-final"),
    ("--weird__name..", "weird__name"),
    ("a" * 60, "a" * 48),
    ("x" * 47 + "-yz", "x" * 47),
    ("杯子", ""),
    ("", ""),
    (None, ""),
    ("v1.2_test", "v1.2_test"),
])
def test_slug(name, expected):
    assert slug(name) == expected


def test_clean_name():
    assert clean_name("  Cups v2 ") == "Cups v2"
    assert clean_name("   ") is None and clean_name(None) is None
    with pytest.raises(BadRequest, match="at most 80"):
        clean_name("a" * 81)
    with pytest.raises(BadRequest, match="control"):
        clean_name("two\nlines")


def test_reserve_uses_the_slug_then_numbers(tmp_path):
    assert reserve_export_dir(tmp_path, "Cups v2")[0] == "cups-v2"
    assert reserve_export_dir(tmp_path, "cups V2")[0] == "cups-v2-2"
    assert reserve_export_dir(tmp_path, "Cups-v2")[0] == "cups-v2-3"
    assert all((tmp_path / d).is_dir() for d in ("cups-v2", "cups-v2-2", "cups-v2-3"))


def test_reserve_falls_back_to_a_timestamp_id(tmp_path):
    export_id, path = reserve_export_dir(tmp_path, "杯子")
    assert len(export_id) == len("20260926-104433-a1b2c3") and export_id[8] == "-" and path.is_dir()


def test_concurrent_reservations_never_share_a_folder(tmp_path):
    got: list[str] = []
    lock = threading.Lock()

    def one():
        export_id, _ = reserve_export_dir(tmp_path, "same")
        with lock:
            got.append(export_id)

    threads = [threading.Thread(target=one) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(got)) == 12
```

Add to the existing test files (follow each file's fixtures):
- `test_export_runner.py`: `run_export(..., name="Cups v2")` → result `export_id == "cups-v2"`, `name == "Cups v2"`,
  `export.json["name"] == "Cups v2"`; a second run with the same name → `"cups-v2-2"`; without a name → timestamp id
  and `name` null; a run that fails (e.g. cancelled) removes its reserved folder.
- `test_api_export.py`: the POST body with `"name": "Cups v2"` → the job result has `export_id == "cups-v2"` and
  `name`; `"name": "a" * 81` → 400 before any job starts.
- `test_train_models.py`: `find_export` of a manifest without `name` → `name is None`; with `name` → the name;
  `list_exports` orders by `created_at` descending then id descending (build three exports with hand-written
  `created_at` values where the folder-name order differs); `rename_export` sets and clears (`None`) the name,
  keeps every other manifest key byte-identical in value (compare the parsed JSON minus `name`), rejects 81 chars.
- `test_api_train.py`: `PATCH /api/exports/{id}` `{"name": "Better"}` → 200 and `GET /api/exports` shows it;
  unknown id → 404; 81 characters → 400.
- `test_augment_derive.py`: an augmented copy of a source named "Cups v2" is named "Cups v2 ×3 augmented" (for
  `per_image=3`); of an unnamed source `<source id> ×3 augmented`.
- `test_train_cli.py`: `--list` prints the name for a named export and `-` for an unnamed one (extend the
  existing `--list` test).

- [ ] **Step 2: Run them to see them fail** — `uv run pytest server/tests -q -m "not gpu" -k "naming or runner or
  api_export or train_models or api_train or derive or train_cli"`.

- [ ] **Step 3: Implement**

`server/vsp/export/naming.py`:

```python
"""Export names (multi-object spec §11): a free-text display name, and a folder id slugged from it."""

from __future__ import annotations

import itertools
import re
import time
import unicodedata
import uuid
from pathlib import Path

from ..labeling.errors import BadRequest

MAX_NAME = 80
MAX_SLUG = 48
_BAD = re.compile(r"[^a-z0-9._-]+")
_DASHES = re.compile(r"-{2,}")


def clean_name(name: str | None) -> str | None:
    """The name as stored: stripped, or None when blank."""
    if name is None:
        return None
    name = name.strip()
    if not name:
        return None
    if len(name) > MAX_NAME:
        raise BadRequest(f"the name must be at most {MAX_NAME} characters")
    if any(unicodedata.category(c) == "Cc" for c in name):
        raise BadRequest("the name must not contain control characters (such as line breaks)")
    return name


def slug(name: str | None) -> str:
    """A folder id from a name: lowercase `[a-z0-9._-]`, starting with a letter or digit; empty when none is left."""
    if not name:
        return ""
    s = _DASHES.sub("-", _BAD.sub("-", name.strip().lower()))
    s = s.lstrip("-._")
    return s[:MAX_SLUG].rstrip("-._")


def timestamp_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def reserve_export_dir(exports_root: Path, name: str | None) -> tuple[str, Path]:
    """Create and return a fresh export folder. `mkdir` without `exist_ok` is the reservation, so two exports
    started at once can never share a folder: the loser of a race moves on to the next number."""
    exports_root = Path(exports_root)
    exports_root.mkdir(parents=True, exist_ok=True)
    base = slug(name)
    if not base:
        export_id = timestamp_id()
        (exports_root / export_id).mkdir()  # a collision here is astronomically unlikely; fail rather than share
        return export_id, exports_root / export_id
    for n in itertools.count(1):
        export_id = base if n == 1 else f"{base}-{n}"
        try:
            (exports_root / export_id).mkdir()
        except FileExistsError:
            continue
        return export_id, exports_root / export_id
    raise AssertionError("unreachable")
```

`runner.py`: `run_export(..., cancel=None, name: str | None = None)`. Replace the id/mkdir lines with
`name = clean_name(name)` (before any folder is made) and `export_id, out = reserve_export_dir(exports_root, name)`
(still outside the `try`). `manifest_for(..., name=name)`; add `"name": name` right after `"export_id"` in the
manifest dict; the returned dict adds `"name": name`. Remove the now-unused `time`/`uuid` imports.

`api/export.py`: `ExportBody.name: str | None = None`; call `clean_name(body.name)` next to `check_request` (a 400
before the job) and pass `name=` to `run_export`.

`train/models.py`: `ExportInfo.name: str | None = None` (after `derived_from`), `to_json` adds `"name": self.name`,
`find_export` passes `name=manifest.get("name")`. `list_exports`:

```python
def list_exports(data_root: Path) -> list[ExportInfo]:
    """Every export with a yolo_seg dataset, newest first by the manifest's `created_at` (then id)."""
    root = Path(data_root) / "exports"
    if not root.is_dir():
        return []
    out = []
    for d in (p for p in root.iterdir() if p.is_dir()):
        try:
            out.append(find_export(data_root, d.name))
        except (NotFound, BadRequest, KeyError, ValueError):  # half-written, not yolo_seg, or unreadable
            continue
    return sorted(out, key=lambda e: (e.created_at, e.id), reverse=True)
```

and

```python
_RENAME_LOCK = threading.Lock()  # renames are rare; one lock keeps two concurrent renames from losing an update


def rename_export(data_root: Path, export_id: str, name: str | None) -> ExportInfo:
    """Set (or clear, with None/blank) an export's display name. The folder id never changes, so runs keep working."""
    from ..export.naming import clean_name
    from .runs import atomic_write_json

    name = clean_name(name)
    info = find_export(data_root, export_id)  # 404 / not-yolo_seg checks
    with _RENAME_LOCK:
        path = info.dir / "export.json"
        manifest = json.loads(path.read_text())
        manifest["name"] = name
        atomic_write_json(path, manifest)
    return find_export(data_root, export_id)
```

(`threading` import; the local imports avoid an import cycle — move them to the top if there is none.)

`api/train.py`:

```python
class RenameBody(BaseModel):
    name: str | None = None


@router.patch("/api/exports/{export_id}")
def rename(export_id: str, body: RenameBody, state: State) -> dict:
    return rename_export(state.settings.project.data_root, export_id, body.name).to_json()
```

`augment/derive.py` (`run_augment`), after `result_manifest["derived_from"] = export_id`:

```python
        result_manifest["name"] = f"{manifest.get('name') or export_id} ×{opts.per_image} augmented"
```

and the returned dict adds `"name": result_manifest["name"]`.

`cli.py` `--list` line:

```python
            print(f"{e.id}  {e.name or '-'}  {e.project_name}  train {e.train_images}  val {e.val_images}  "
                  f"classes: {', '.join(e.classes)}")
```

- [ ] **Step 4: Run** — `uv run pytest server/tests -q -m "not gpu"` and `uv run ruff check server` → green.

- [ ] **Step 5: Commit** — "Exports: free-text names, folder ids slugged from them, rename".

---

### Task 2: Web — name on export, names on the Train page, rename

**Files:**
- Modify: `web/src/api/types.ts`, `web/src/api/rest.ts`, `web/src/lib/train.ts`, `web/src/pages/Export.tsx`,
  `web/src/components/TrainSidebar.tsx`, `web/src/components/RunsTable.tsx`, `web/src/pages/Train.tsx`
- Create: `web/src/lib/exportName.ts`
- Test: create `web/src/lib/exportName.test.ts`; extend `web/src/pages/Export.test.tsx`,
  `web/src/components/RunsTable.test.tsx`, `web/src/pages/Train.test.tsx`

**Interfaces:**
- Consumes: Task 1 HTTP shapes.
- Produces: `ExportInfo.name: string | null`; `ExportRequest.name?: string | null`;
  `ExportResult.name: string | null`; `api.renameExport(id, name) -> Promise<ExportInfo>`;
  `exportName.slug(name)` (same rules as the server), `exportLabel(e: { id: string; name: string | null })`.

- [ ] **Step 1: Write the failing tests**

`exportName.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { exportLabel, slug } from "./exportName";

describe("export names", () => {
  it("slugs like the server", () => {
    expect(slug("Cups v2")).toBe("cups-v2");
    expect(slug("  Tassen / Schalen (final)  ")).toBe("tassen-schalen-final");
    expect(slug("--weird__name..")).toBe("weird__name");
    expect(slug("a".repeat(60))).toBe("a".repeat(48));
    expect(slug("x".repeat(47) + "-yz")).toBe("x".repeat(47));
    expect(slug("杯子")).toBe("");
  });
  it("labels an export by name, else by id", () => {
    expect(exportLabel({ id: "cups-v2", name: "Cups v2" })).toBe("Cups v2");
    expect(exportLabel({ id: "20260926-104433-a1b2c3", name: null })).toBe("20260926-104433-a1b2c3");
  });
});
```

Component tests (follow each file's existing render helpers and API mocks):
- Export page: typing "Cups v2" in the "Dataset name" field shows the note `Folder: cups-v2` and sends
  `name: "Cups v2"` in `api.startExport`'s body; an empty field sends `name: null` and the note reads
  `Folder: a timestamp, e.g. 20260926-104433-…`; a name that slugs to nothing shows the same timestamp note; the
  finished result shows the name above the folder path.
- RunsTable: given `exports=[{id: "cups-v2", name: "Cups v2", ...}]`, a run with `export_id "cups-v2"` shows
  "Cups v2" with "cups-v2" beneath; a run whose export is unknown or unnamed shows `project_name` with the id beneath
  (today's rendering).
- Train page / sidebar: the export select's option text starts with the name when there is one; the "Rename"
  field is prefilled with the current name, Save calls `api.renameExport("cups-v2", "Better")` and then the exports
  refresh; Save is disabled while the text equals the current name (after trimming).

- [ ] **Step 2: Run to see them fail** — `cd web && node node_modules/.bin/vitest run`.

- [ ] **Step 3: Implement**

`web/src/lib/exportName.ts`:

```ts
/** Folder id from a dataset name; must match server/vsp/export/naming.py `slug`. Empty = a timestamp id is used. */
export function slug(name: string): string {
  let s = name.trim().toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/-{2,}/g, "-");
  s = s.replace(/^[-._]+/, "");
  return s.slice(0, 48).replace(/[-._]+$/, "");
}

export const exportLabel = (e: { id: string; name: string | null }): string => e.name ?? e.id;
```

Types: add `name: string | null` to `ExportInfo` and `ExportResult`, `name?: string | null` to `ExportRequest`.
`rest.ts`: `renameExport: (exportId: string, name: string | null) => request<ExportInfo>(\`/api/exports/${exportId}\`,
{ method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ name }) })` — use the file's
existing JSON helper (like `post`) if it has a PATCH-capable one, else add a `patch(body)` helper next to `post`.

Export page (right sidebar, at the top of the "Start" section, above the problem text):

```tsx
<label className="flex flex-col gap-1 text-sm text-dim">
  Dataset name
  <input aria-label="Dataset name" className="field" maxLength={80} placeholder="Optional, e.g. Cups v2"
    value={name} onChange={(e) => setName(e.target.value)} />
</label>
<p className="note">
  Folder: {slug(name) !== "" ? slug(name) : "a timestamp, e.g. 20260926-104433-…"}
  {slug(name) !== "" ? " (a number is added if it is taken)" : ""}
</p>
```

`startExport` sends `name: name.trim() === "" ? null : name.trim()`. In the "Dataset written" block, above the
path textarea: `{result.name !== null && <p className="text-sm text-ink">{result.name}</p>}`.

TrainSidebar: option text `{e.name !== null ? \`${e.name} · ${e.id}\` : \`${e.project_name} · ${e.id}\`} ·
{e.train_images} train / {e.val_images} val`. Below the select, when an export is selected:

```tsx
<form className="flex gap-1" onSubmit={(ev) => { ev.preventDefault(); void rename(); }}>
  <input aria-label="Rename export" className="field flex-1" maxLength={80} placeholder="Name"
    value={draftName} onChange={(e) => setDraftName(e.target.value)} />
  <button type="submit" className="btn" disabled={renaming || draftName.trim() === (selected?.name ?? "")}>Save</button>
</form>
{renameError !== null && <p role="alert" className="text-xs text-danger">{renameError}</p>}
```

`draftName` resets to the selected export's `name ?? ""` whenever the selected id (or its name) changes; `rename()`
calls `api.renameExport(id, draftName.trim() === "" ? null : draftName.trim())`, then `onExportsChanged()`; errors
show in `renameError`.

RunsTable: new prop `exports: ExportInfo[]`; the export cell shows
`exports.find((e) => e.id === r.export_id)?.name ?? r.project_name` over `r.export_id`. `Train.tsx` passes
`exports={exportsList}`.

- [ ] **Step 4: Run** — `cd web && node node_modules/.bin/vitest run && node node_modules/.bin/tsc --noEmit && node
  node_modules/.bin/vite build` → green.

- [ ] **Step 5: Commit** — "Web: dataset names on the Export and Train pages, rename".

---

### Task 3: Docs

- [ ] **Step 1:** README: in the export section, document the Dataset name field, the folder-id rule (slug,
  `-2`/`-3` on collision, timestamp when empty), renaming on the Train page (`PATCH /api/exports/{id}`), and that
  `vsp train --export` takes the folder id. Add a Manual QA step: "Export twice with the name "Cups v2": the folders
  are `cups-v2` and `cups-v2-2`; rename one on the Train page; the Runs table shows the new name."
- [ ] **Step 2:** Run the full non-GPU Python suite, ruff, and the web tests/typecheck/build.
- [ ] **Step 3: Commit** — "Docs: named datasets".
