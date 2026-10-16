# Manual QA checklist

Run on real footage before trusting a dataset.

1. Import a video and a frame directory (try a stride > 1); cancel one import midway and confirm no clip appears.
2. Label by each method: clicks (add a negative click), a box dragged bottom-right → top-left plus a refining click,
   and a text phrase with several candidates. Check the "no usable mask" message on an empty region.
3. Propagate forward, backward and both from a mid-clip keyframe; press Stop during one and confirm the masks so far stay.
4. Let the object leave the frame: propagation should stop early and seek to the "lost at" frame.
5. Correct a bad frame mid-clip (it becomes a keyframe) and propagate again: the run must stop at the next keyframe.
   Then Stop a run part-way and propagate again from the last masked frame: it continues from there, and the
   buttons stay disabled on a frame that has no mask.
6. Scrub the timeline and play at clip fps with the overlay on; delete a frame, restore it, prune the tail. Clear masks:
   Shift-drag a stretch of the timeline (or press `[` and `]`), check the button names the frames and the note counts
   the masks and keyframes, clear them, and confirm the keyframes inside the stretch are gone from the timeline too,
   the mark disappears, and a new prompt lands on the active object with its class.
7. Reload the page during a propagation: the panel should re-attach to the running job.
8. Stop the server: the banner says it is unreachable; start it without weights: the banner shows the reason and
   importing / cleanup still work.
9. Propagate twice in one page session (let the first run finish, then start a second from a keyframe) and reload
   the page partway through the second run: it must re-attach to the second job, not the first.
10. Open the same clip in two tabs; start a propagation in tab A, then click to prompt in tab B on a different clip:
    B must not move or show A's progress as its own.
11. Delete a keyframe: its marker disappears from the timeline and propagation is blocked from it until it is restored.
12. Read the timeline without relying on colour: masked runs are filled (in the class colour), frames without a mask
    are hollow, deleted frames are hatched and keyframes carry an amber diamond. Hover it for a thumbnail with the
    frame number and state. Zoom into the frame, then press Fit in the tool rail. Tab through the page: every control
    shows a focus ring.
13. Export (the "Export dataset" button on the Clips page): with two or more clips, move a clip between Training and
    Validation by hand and export all four formats; with one clip selected, check the note says the tail of the clip
    becomes the validation set. Cancel an export midway and confirm its folder under `data/exports/` is gone. Open
    `export.json`, then load the result in the tool you train with: `yolo_seg/data.yaml` or `yolo_det/data.yaml` in
    Ultralytics, `coco/annotations/instances_train.json` with pycocotools, and look at a few `masks/masks/` PNGs
    (pixel value = `mask_value` from the manifest's class map).
14. Upload footage from the Clips page over the SSH forward. Set stride/fps, then choose (or drop) several files
    at once, mixing a video, a zip and an invalid file (e.g. a `.txt`): uploading must start immediately with no
    button click, the invalid file must be skipped with a note naming it, and the rest must upload one at a time
    in the order chosen, each finished one becoming a clip whose name, frame count and fps are right. Watch the
    percentage move for the running file and the waiting files listed by name; remove a waiting file with its ×
    button and confirm it never uploads. Let one file fail (e.g. by editing `ingest.max_upload_bytes` down) and
    confirm it shows its error with a Dismiss button while the rest still upload. Press "Cancel upload" mid-transfer:
    it must stop without an error message and the queue must move on to the next file. With several files still
    queued, press "Cancel all": the running transfer and every waiting file must clear. Reload the page during an
    upload and confirm the transfer pauses, no partial clip appears, and choosing the same file again offers to
    continue it from where it stopped; "Cancel upload" discards the unfinished upload.
15. Upload a file larger than `ingest.max_upload_bytes`: the GUI must refuse it before the transfer starts, and
    `curl -X POST --data-binary @big.mp4 .../clips/upload?name=big.mp4` must answer 413.
16. Brush: pick the Brush tool on a masked frame, paint a missing corner (left-drag) and erase a spill (right-drag);
    the transport, the other tools, Propagate and Cleanup are disabled until you Save (Enter) or Discard (Esc). After
    Save the frame shows a keyframe diamond; propagate from it and check the run stops at the next keyframe. Erase the
    whole mask on another frame and Save: the frame becomes empty. The brush works with the engine unloaded.
17. Mask colours (right sidebar): pick a new Class colour for the active object's class — the fill on this frame,
    the timeline lane and text-prompt candidates all repaint; reopen the clip (or the project's other clips) and
    the colour stuck. Then, under Outline, untick "Outline follows class colour" and pick a contrasting colour, drag
    Thickness from 0 (no outline) to 12; the stored mask, the frames arriving during a propagation and text-prompt
    candidates all follow it, with no white halo outside the mask. Reload the page: the outline setting is kept
    (it is this browser only). Reset restores the outline to the class colour at 5 px, without touching the class
    colour itself.
18. Named datasets: export twice with the name "Cups v2"; the folders are `cups-v2` and `cups-v2-2`; rename one on the
    Train page; the Runs table shows the new name.
19. (server) With two objects on a clip, `POST /propagate` with `objects: "all"` writes both objects' masks in one
    run; with `[2]` only object 2's; an object that disappears is reported under `lost` while the other continues.
20. Class selection: on a project with three classes, untick one on the Export page (the counts follow the
    included clips) and export all four formats; `yolo_seg/data.yaml` lists the remaining classes as 0 and 1,
    COCO categories are 1 and 2, the mask PNG values are 1 and 2, and `export.json` maps them back with
    `project_class_id`. Untick every class: Start says "Choose at least one class." Put a small object on a
    large one: the mask PNG shows the small one, and `overlap_px` is non-zero. Make an augmented copy of an
    export with COCO: `augment.cutouts` is `per_object`; of one without COCO: the panel notes the fallback and
    it is `per_class_region`.
21. Several objects: on one clip label a cup (#1), press `N`, choose another class and label it (#2). The Objects
    panel lists both with frame counts; the timeline shows one lane per class with an object (a third, unused
    class has no lane until it is chosen in the class select for a new object), the active object's class tall; clicking a lane or its name selects that class's
    object, and `O` / `Shift+O` cycle. Hide #1 with its eye: it leaves the frame and its lane dims once every object
    of its class is hidden. Only the active object carries the outline; brushing #2 keeps #1 visible underneath. With
    #1 active, pick #2's class from the Class select: it starts a new object of that class (a third object) rather
    than relabelling #1 — use "Change class of #1" and **Change** instead to actually relabel it.
22. Propagate "All objects" from a frame where both have masks: both classes' lanes fill in one run. Then
    "Selected: #2" from another of its keyframes: only #2's class lane changes. Let one object leave the frame: the
    panel names it as lost while the other continues; start from a frame where only #1 has a mask with "All
    objects": #2 is named as skipped.
23. Shift-drag a stretch and clear it with "Active object", then again with "All objects"; check the note counts the
    right masks each time. Relabel #2 during a propagation with "Change class of #2" and **Change** (allowed); try to
    delete #1 during a propagation and during an export (refused), then delete it (the dialog names its frame count)
    and check #2 becomes active.
24. Text-prompt "cup" on a New object row with several cups in view: "Accept all as new objects" creates one object
    per candidate, all in the New object row's class — the timeline keeps showing a single lane for it, labelled
    "cup ×n". Add classes until the project has seven, labelling one object in each: the timeline stops growing at
    six lanes and scrolls with the wheel, and `O` onto an off-screen object's class scrolls its lane into view.
