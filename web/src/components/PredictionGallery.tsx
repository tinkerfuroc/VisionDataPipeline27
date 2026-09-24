import { useCallback, useEffect, useRef, useState } from "react";
import { api, valImageUrl } from "../api/rest";
import type { PredictionFilter, PredictionItem, PredictionPage, Run } from "../api/types";
import { SERIES_COLORS, TEXT_INK } from "../lib/chart";
import { GalleryCard } from "./GalleryCard";
import { PredictionOverlay } from "./PredictionOverlay";

const PAGE = 20;
const VISIBLE = 4;
const FILTERS: { id: PredictionFilter; label: string }[] = [
  { id: "all", label: "All" }, { id: "missed", label: "Missed" }, { id: "worst", label: "Worst IoU" },
];
const inField = (t: EventTarget | null) =>
  t instanceof HTMLElement && ["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName);

function caption(it: PredictionItem) {
  if (it.ground_truth.length > 0 && it.predictions.length === 0) {
    return (<><span aria-hidden className="text-danger">✕</span><span className="text-ink">#{it.index + 1} · missed</span></>);
  }
  return <span className="text-ink">#{it.index + 1}{it.iou === null ? "" : ` · IoU ${it.iou.toFixed(2)}`}</span>;
}

export function PredictionGallery({ run }: { run: Run }) {
  const [filter, setFilter] = useState<PredictionFilter>("all");
  const [items, setItems] = useState<PredictionItem[]>([]);
  const [meta, setMeta] = useState<Pick<PredictionPage, "total" | "counts" | "conf"> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [first, setFirst] = useState(0);
  const [open, setOpen] = useState<number | null>(null);
  const [showGt, setShowGt] = useState(true);
  const [showPred, setShowPred] = useState(true);
  const token = useRef(0);
  const loading = useRef(false);
  const strip = useRef<HTMLDivElement>(null);

  const load = useCallback(async (offset: number, reset: boolean) => {
    if (loading.current && !reset) return;
    const my = ++token.current;
    loading.current = true;
    try {
      const p = await api.predictions(run.id, offset, PAGE, filter);
      if (my !== token.current) return;
      setMeta({ total: p.total, counts: p.counts, conf: p.conf });
      setItems((prev) => (reset ? p.items : [...prev, ...p.items]));
      setError(null);
    } catch (e) {
      if (my === token.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (my === token.current) loading.current = false;
    }
  }, [run.id, filter]);

  useEffect(() => {
    setItems([]);
    setFirst(0);
    setOpen(null);
    void load(0, true);
  }, [load]);

  const total = meta?.total ?? 0;
  // near the end of what is loaded: fetch the next page
  useEffect(() => {
    const at = open ?? first;
    if (items.length < total && at + VISIBLE * 2 >= items.length) void load(items.length, false);
  }, [first, open, items.length, total, load]);

  const scrollTo = (index: number) => {
    const card = strip.current?.children[index] as HTMLElement | undefined;
    card?.scrollIntoView?.({ inline: "start", block: "nearest", behavior: "smooth" });
  };
  const page = (d: number) => {
    const next = Math.max(0, Math.min(first + d * VISIBLE, Math.max(0, total - 1)));
    setFirst(next);
    scrollTo(next);
  };
  const step = (d: number) => setOpen((o) => (o === null ? o : Math.max(0, Math.min(o + d, items.length - 1))));
  const close = () => {
    if (open !== null) setFirst(Math.max(0, open - (open % VISIBLE)));
    setOpen(null);
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (inField(e.target)) return;
      if (e.key === "ArrowLeft") (open === null ? page(-1) : step(-1));
      else if (e.key === "ArrowRight") (open === null ? page(1) : step(1));
      else if (e.key === "Escape" && open !== null) close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  // keep the counter honest when the user scrolls the strip by hand (browsers only; jsdom has no layout)
  const onScroll = () => {
    const el = strip.current;
    const card = el?.children[0] as HTMLElement | undefined;
    if (!el || !card || card.offsetWidth === 0) return;
    setFirst(Math.round(el.scrollLeft / (card.offsetWidth + 12)));
  };

  if (error !== null) return <p role="alert" className="alert">{error}</p>;
  if (meta === null) return <p className="note">Loading predictions…</p>;

  const toggles = (
    <>
      <label className="flex items-center gap-1.5 text-dim">
        <input type="checkbox" checked={showGt} onChange={(e) => setShowGt(e.target.checked)} />
        <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm border-2" style={{ borderColor: TEXT_INK }} />
        Ground truth
      </label>
      <label className="flex items-center gap-1.5 text-dim">
        <input type="checkbox" checked={showPred} onChange={(e) => setShowPred(e.target.checked)} />
        <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: SERIES_COLORS[0] }} />
        Predictions (conf ≥ {meta.conf ?? 0.25})
      </label>
    </>
  );

  if (open !== null && items[open]) {
    const it = items[open];
    return (
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <button type="button" className="btn" onClick={close}>Back to strip</button>
          <button type="button" className="btn" aria-label="Previous image" onClick={() => step(-1)}>‹</button>
          <span className="font-cond text-dim">{open + 1} / {total} · {it.name}</span>
          <button type="button" className="btn" aria-label="Next image" onClick={() => step(1)}>›</button>
          {toggles}
        </div>
        <div className="relative w-full overflow-hidden rounded border border-line bg-well">
          <img src={valImageUrl(run.export_id, it.name)} alt={it.name} className="block w-full" />
          <PredictionOverlay width={it.width} height={it.height} groundTruth={it.ground_truth}
            predictions={it.predictions} showGt={showGt} showPred={showPred} scores />
        </div>
      </div>
    );
  }

  const last = Math.min(first + VISIBLE, total);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        {FILTERS.map((f) => (
          <button key={f.id} type="button" className={`btn ${filter === f.id ? "btn-on" : ""}`}
            aria-pressed={filter === f.id} onClick={() => setFilter(f.id)}>
            {f.label} {f.id === "worst" ? "" : meta.counts[f.id]}
          </button>
        ))}
        <span className="flex-1" />
        {toggles}
      </div>
      {total === 0 ? (
        <p className="note">No frames match this filter.</p>
      ) : (
        <div className="flex items-center gap-2">
          <button type="button" className="btn self-center" aria-label="Previous four" disabled={first === 0} onClick={() => page(-1)}>‹</button>
          <div ref={strip} onScroll={onScroll}
            className="grid min-w-0 flex-1 snap-x snap-mandatory auto-cols-[calc((100%-2.25rem)/4)] grid-flow-col gap-3 overflow-x-auto pb-1">
            {items.map((it, i) => (
              <GalleryCard key={it.name} src={valImageUrl(run.export_id, it.name)} name={it.name} width={it.width}
                height={it.height} groundTruth={it.ground_truth} predictions={it.predictions} showGt={showGt}
                showPred={showPred} caption={caption(it)} onOpen={() => setOpen(i)} />
            ))}
          </div>
          <button type="button" className="btn self-center" aria-label="Next four" disabled={last >= total} onClick={() => page(1)}>›</button>
        </div>
      )}
      {total > 0 && <p className="note self-center">{first + 1}–{last} of {total}</p>}
    </div>
  );
}
