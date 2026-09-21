# Vision Sampler GUI — design pass (approved 2026-09-21)

Branch: `gui-design-pass` (from `main` ef998f2). Visual/layout work only: no behaviour change, no server change.
Skill: frontend-design. Verify each step with a headless-Firefox screenshot (see "Screenshots" below).

## Brief
Audience: an operator scrubbing frames for long sessions, judging whether a coloured mask hugs a component.
The frame and the mask are the content; the chrome must not compete with them or skew how their colours read.

## Tokens
- Surround `#2B2D30` (neutral mid-dark grey, like a grading suite) · canvas well `#1E1F21` (only behind the frame) ·
  panel `#36383C` · hairline `#4A4D52` · text `#E9E7E2` · secondary text `#A3A5AA` · keyframe amber `#F59E0B` (keyframes only).
- Chrome has NO hue. Selected/active = an inverted light chip with dark text, not an accent colour (class colours already
  span the hue wheel, so any accent would collide with one).
- Type: Barlow (text) + Barlow Semi Condensed (dense inspector), self-hosted with `@fontsource/barlow` and
  `@fontsource/barlow-semi-condensed` so the app works offline. Tabular figures for frame numbers.

## Layout (Label page)
```
← clips   clip name   640×480, 10 fps                          ● SAM3 cuda:0
┌────┬───────────────────────────────────────────┬──────────────────────┐
│Clk │                                           │ Object   [class ▾] + │
│Box │                 frame                     │ Prompt   (contextual)│
│Txt │                                           │ Propagate ◀  ⟷  ▶  ■ │
│Fit │                                           │ ▸ Cleanup (folded)   │
├────┴───────────────────────────────────────────┴──────────────────────┤
│ ⏪ ◀ ▶ ▶▏ ⏩   012 / 039  0012.png        1 keyframe  39 masked  0 deleted │
│ timeline, 44 px: masked runs in the CLASS colour, amber keyframe diamonds,  │
│ hatched deleted frames, hollow empty frames; hover shows thumbnail + frame  │
└───────────────────────────────────────────────────────────────────────┘
```
- Tools (Click / Box / Text / Fit) move to a rail beside the canvas; clip statistics move to the timeline row; Cleanup
  folds into a `<details>` so the inspector no longer overflows at 950 px height; condensed type stops button wrapping.
- The ONE bold element is the timeline. Everything else stays quiet.
- Frame state must be readable by shape/texture, not hue alone.

## Copy
Plain verbs, sentence case; shortcuts as small key chips instead of "(Enter)" inside labels; empty and error states say
what to do next. No all-caps labels, no middle-dot meta strings, no "→" on buttons.

## Constraints
- Keep every accessible name the tests use (button names such as "Click", "Box", "Text", "Forward ▶", "◀ Backward",
  "Both", "Stop", "Next frame", "Play"/"Pause", labels "Frame", "Timeline", "Range start", "New class", …). 134 web tests
  must stay green; when a control moves to a new component (the tool rail), move its assertions with it — never delete.
- Quality floor: visible keyboard focus, reduced motion respected, contrast AA on text.
- `web/package.json` may gain exactly the two `@fontsource` packages.

## Screenshots
Firefox 111 headless takes the shot at the load event, before fetches finish. Delay it: serve a wrapper page from
`web/dist/shot-*.html` with an `<iframe src="/#/clips/<id>">` plus an `<img>` pointing at a tiny local server that sleeps
6 s (port 28999); unset the proxy env vars and use a fresh `--profile`. Seed data with
`uv run python scripts/seed_demo.py` against `HF_HUB_OFFLINE=1 uv run vsp serve`. Stop servers by PID from `ss -ltnp`
(never `pkill -f "vsp serve"` from a shell whose own command line contains that string). Remove the shot pages afterwards.

## Revisions after the first build (user feedback, 2026-09-21)
- Every control sits on the RIGHT on every page: on the Label page the tool rail stands between the frame and the
  inspector; on the Clips page the classes/import column is the right-hand sidebar. New pages follow the same rule.
- The interface was too small: `html { font-size: 112.5% }` in `web/src/styles.css` scales everything (all sizes are rem);
  the timeline (a canvas, sized in px) grew to 54 px. Tune that one percentage if it is still too small or too large.
- Clearing masks was not intuitive, so it changed from "type two numbers" to "mark, then clear" (a behaviour change, the
  only one in this pass): Shift-drag on the timeline, or `[` / `]`, marks a range that is drawn as a bracket on the strip;
  the button says exactly what it will do ("Clear masks in frames 14 to 22") and the note counts the propagated masks that
  will go (keyframes are kept); with nothing marked it acts on the current frame; the mark is dropped after a successful
  request. The number fields stay for precision. Cleanup is open by default (it can still be folded).
- Screenshot recipe addendum: headless Firefox can leave late-arriving text unpainted (seen on the clip cards). Have the
  wrapper page call `getBoundingClientRect()` on something inside the iframe about a second before the shot; a real
  browser flushes every frame, so this is not an application bug.
