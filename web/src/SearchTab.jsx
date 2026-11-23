import React, { Suspense, lazy, useEffect, useState } from "react";
import { api, fmtSeconds } from "./api.js";
import ClipDetail from "./ClipDetail.jsx";
import ResultCard from "./ResultCard.jsx";

const ClipMap = lazy(() => import("./ClipMap.jsx"));      // three.js loads only when the map is opened

const LEVELS = [
  { id: "fast", name: "Fast", note: "LoRA-tuned SigLIP 2 · 0 VLM calls" },
  { id: "standard", name: "Standard", note: "LoRA + temporal re-ranker · 0 VLM calls" },
  { id: "verified", name: "Verified", note: "+ the VLM checks the top k · k VLM calls" },
];
const COLLECTIONS = [["alerts", "Your videos"], ["meva", "MEVA eval split"], ["both", "Both"]];
const STAGE_NAME = { text: "text", lora: "LoRA", student: "re-ranker", frames: "frames", vlm: "VLM", frozen: "frozen" };

function Stats({ cost }) {
  const stages = Object.entries(cost.seconds).map(([k, v]) => `${STAGE_NAME[k] ?? k} ${fmtSeconds(v)}`).join(" · ");
  return (
    <>
      <div className="stats">
        <div className="stat"><span>Clips searched</span><b>{cost.n_searched.toLocaleString()}</b></div>
        <div className="stat"><span>Search time</span><b>{fmtSeconds(cost.total)}</b></div>
        <div className="stat"><span>VLM calls</span><b style={{ color: "var(--accent)" }}>{cost.vlm_calls}</b></div>
        <div className="stat">
          <span>VLM on every clip</span>
          <b style={{ color: "var(--amber)" }}>≈ {fmtSeconds(cost.vlm_all_seconds)}</b>
          <small>{cost.n_searched.toLocaleString()} calls × {fmtSeconds(cost.vlm_per_call)} ({cost.vlm_per_call_source})</small>
        </div>
      </div>
      <span className="stages">{stages}</span>
    </>
  );
}

function Results({ items, showLabels, onOpen, empty }) {
  if (!items.length) return <div className="empty">{empty}</div>;
  return <div className="grid">{items.map((it) => <ResultCard key={it.id} item={it} showLabels={showLabels} onOpen={onOpen} />)}</div>;
}

// ?q=...&level=...&collection=...&view=map&labels&compare: a link that opens with that search (handy when recording)
const url = new URLSearchParams(location.search);

export default function SearchTab({ status, preset }) {
  const [query, setQuery] = useState(url.get("q") ?? status.examples[0] ?? "");
  // without the MEVA media the server offers "alerts" only (status.collections)
  const offered = COLLECTIONS.filter(([id]) => status.collections.includes(id));
  const wanted = url.get("collection") ?? (status.alerts ? "both" : "meva");
  const [collection, setCollection] = useState(status.collections.includes(wanted) ? wanted : status.collections[0]);
  const [level, setLevel] = useState(url.get("level") ?? "standard");
  const [k, setK] = useState(10);
  const [classes, setClasses] = useState([]);
  const [videos, setVideos] = useState([]);
  const [filters, setFilters] = useState({ site: "", camera: "", date: "", hour_from: "", hour_to: "" });
  const [showLabels, setShowLabels] = useState(url.has("labels"));
  const [compare, setCompare] = useState(url.has("compare"));
  const [view, setView] = useState(url.get("view") ?? "grid");
  const [res, setRes] = useState(null);
  const [base, setBase] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(null);

  // ?q=... opens with that search already run (the first render's state comes from the URL)
  useEffect(() => { if (url.get("q")) run(); }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!preset) return;
    setCollection(status.collections.includes(preset.collection) ? preset.collection : "alerts");
    setVideos(preset.videos ?? []);
  }, [preset]);

  const body = (system = "final") => ({
    query, collection, level: system === "frozen" ? "fast" : level, k, classes, videos, system, top: compare ? 12 : 24,
    filters: {
      site: filters.site ? [filters.site] : [], camera: filters.camera ? [filters.camera] : [],
      date: filters.date ? [filters.date] : [],
      hour_from: filters.hour_from === "" ? null : Number(filters.hour_from),
      hour_to: filters.hour_to === "" ? null : Number(filters.hour_to),
    },
  });

  const run = async (e) => {
    e?.preventDefault();
    if (!query.trim()) return;
    setBusy(true); setError(null);
    try {
      const [a, b] = await Promise.all([api.search(body()), compare ? api.search(body("frozen")) : Promise.resolve(null)]);
      setRes(a); setBase(b);
    } catch (err) { setError(err.message); }
    setBusy(false);
  };

  const toggleClass = (c) => setClasses((xs) => (xs.includes(c) ? xs.filter((x) => x !== c) : [...xs, c]));
  const hours = Array.from({ length: status.filters.hours[1] - status.filters.hours[0] + 1 }, (_, i) => status.filters.hours[0] + i);
  const lvl = LEVELS.find((l) => l.id === level);
  const usesMeva = collection !== "alerts";
  const usesAlerts = collection !== "meva";

  return (
    <div className="page">
      <aside className="side" aria-label="Search settings">
        <div className="group" role="radiogroup" aria-label="Collection">
          <span className="label">Collection</span>
          {offered.map(([id, name]) => (
            <label className="radio" key={id}>
              <input type="radio" name="collection" checked={collection === id} onChange={() => setCollection(id)} />
              {name}{id === "alerts" && <span className="muted"> · {status.alerts} alerts</span>}
            </label>
          ))}
          {videos.length > 0 && usesAlerts && (
            <span className="row muted" style={{ fontSize: 12 }}>
              only video #{videos.join(", #")} <button className="btn small" onClick={() => setVideos([])}>all videos</button>
            </span>
          )}
        </div>

        <div className="group">
          <span className="label">Cost level</span>
          <div className="seg">
            {LEVELS.map((l) => (
              <button key={l.id} aria-pressed={level === l.id} disabled={l.id === "verified" && !status.vlm}
                      onClick={() => setLevel(l.id)}>{l.name}</button>
            ))}
          </div>
          <span className="muted" style={{ fontSize: 12 }}>{lvl.note}</span>
          {level === "verified" && (
            <label className="field">k = {k} clips checked by the VLM
              <input type="range" min="5" max="20" step="5" value={k} onChange={(e) => setK(Number(e.target.value))} />
            </label>
          )}
        </div>

        {usesAlerts && (
          <div className="group">
            <span className="label">Alert type</span>
            <div className="chips">
              {status.classes.map((c) => (
                <button key={c} className="chip" aria-pressed={classes.includes(c)} onClick={() => toggleClass(c)}>{c}</button>
              ))}
            </div>
            <span className="muted" style={{ fontSize: 12 }}>{classes.length ? "only the selected types" : "all types"}</span>
          </div>
        )}

        {usesMeva && (
          <div className="group">
            <span className="label">MEVA filters</span>
            {[["site", "Site"], ["camera", "Camera"], ["date", "Day"]].map(([key, name]) => (
              <label className="field" key={key}>{name}
                <select value={filters[key]} onChange={(e) => setFilters({ ...filters, [key]: e.target.value })}>
                  <option value="">All</option>
                  {status.filters[key].map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
            ))}
            <div className="row">
              {[["hour_from", "From hour"], ["hour_to", "To hour"]].map(([key, name]) => (
                <label className="field" key={key} style={{ flex: 1 }}>{name}
                  <select value={filters[key]} onChange={(e) => setFilters({ ...filters, [key]: e.target.value })}>
                    <option value="">Any</option>
                    {hours.map((h) => <option key={h} value={h}>{String(h).padStart(2, "0")}:00</option>)}
                  </select>
                </label>
              ))}
            </div>
          </div>
        )}

        <div className="group">
          <span className="label">Display</span>
          <label className="switch"><input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />Show MEVA labels (✓ / ✗)</label>
          <label className="switch"><input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />Compare with frozen SigLIP 2</label>
        </div>
      </aside>

      <section className="main">
        <form className="query" onSubmit={run}>
          <label>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true"><circle cx="9" cy="9" r="6" stroke="#8B98A5" strokeWidth="1.8" /><path d="M14 14l4 4" stroke="#8B98A5" strokeWidth="1.8" strokeLinecap="round" /></svg>
            <span className="sr">Query</span>
            <input value={query} onChange={(e) => setQuery(e.target.value)} list="examples" placeholder="Describe what you are looking for" />
            <datalist id="examples">{status.examples.map((x) => <option key={x} value={x} />)}</datalist>
          </label>
          <button className="btn primary big" disabled={busy || !query.trim()}>{busy ? <span className="spin" /> : "Search"}</button>
        </form>

        {error && <div className="error">{error}</div>}
        {res && <Stats cost={res.cost} />}

        {res && (
          <>
            <div className="head2">
              <h2>{compare ? "Frozen SigLIP 2 vs the final system" : "Top results"}</h2>
              <div className="row">
                <span className="muted" style={{ fontSize: 12 }}>Confidence is relative to the MEVA corpus, not a probability</span>
                {!compare && (
                  <div className="view" role="group" aria-label="View">
                    <button aria-pressed={view === "grid"} onClick={() => setView("grid")}>Grid</button>
                    <button aria-pressed={view === "map"} onClick={() => setView("map")}>3D clip map</button>
                  </div>
                )}
              </div>
            </div>
            {res.expected_activity && showLabels && (
              <span className="muted" style={{ fontSize: 13 }}>
                This is an evaluation phrasing for <b className="mono">{res.expected_activity}</b>:{" "}
                {res.results.slice(0, 10).filter((r) => r.match).length} of the top 10 MEVA results carry that label
                {base && ` (frozen: ${base.results.slice(0, 10).filter((r) => r.match).length})`}.
              </span>
            )}
            {compare && base ? (
              <div className="compare">
                <div className="group"><span className="label">Frozen SigLIP 2 (baseline, Result 02)</span>
                  <Results items={base.results} showLabels={showLabels} onOpen={setOpen} empty="No clips match the filters." /></div>
                <div className="group"><span className="label">Final system · {lvl.name}</span>
                  <Results items={res.results} showLabels={showLabels} onOpen={setOpen} empty="No clips match the filters." /></div>
              </div>
            ) : view === "map" ? (
              <Suspense fallback={<div className="map"><span className="spin" style={{ margin: 20 }} /></div>}>
                <ClipMap results={res.results} queryPoint={res.query_point} onOpen={setOpen} />
              </Suspense>
            ) : (
              <Results items={res.results} showLabels={showLabels} onOpen={setOpen}
                       empty={collection === "alerts" && !status.alerts ? "No alerts yet: upload a video in the Detect tab." : "No clips match the filters."} />
            )}
          </>
        )}
        {!res && !error && <div className="empty">Type a description, or pick an example, and press Search.</div>}
      </section>
      {open && <ClipDetail item={open} onClose={() => setOpen(null)} />}
    </div>
  );
}
