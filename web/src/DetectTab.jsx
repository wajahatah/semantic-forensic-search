import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, fmtTime } from "./api.js";

const CLASS_COLOR = { person: "#22D3EE", backpack: "#A78BFA" };     // vehicles: amber (as drawn in the video)
const STAGES = [["detect", "Detect + track"], ["embed", "Cut clips + embed"], ["done", "Searchable"]];

function JobCard({ job }) {
  const at = STAGES.findIndex(([id]) => id === job.stage);
  return (
    <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
      <div className="head2">
        <b>{job.name}</b>
        <span className="mono muted" style={{ fontSize: 12 }}>{job.state === "failed" ? "failed" : job.message || job.state}</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 10 }}>
        {STAGES.map(([id, name], i) => {
          const pct = job.state === "done" || i < at ? 100 : i === at ? Math.round(job.progress * 100) : 0;
          return (
            <div key={id} className="group" style={{ gap: 6 }}>
              <div className="head2" style={{ fontSize: 12 }}><span style={{ color: "var(--text-2)" }}>{name}</span><span className="mono muted">{pct}%</span></div>
              <div className="progress"><div style={{ width: `${pct}%` }} /></div>
            </div>
          );
        })}
      </div>
      {job.error && <div className="error">{job.error}</div>}
    </div>
  );
}

function Player({ video, alerts, selected, onSelect }) {
  const ref = useRef(null);
  const [t, setT] = useState(0);
  const sel = alerts.find((a) => a.id === selected);
  useEffect(() => {
    const v = ref.current;
    if (v && sel) { v.currentTime = sel.start_s; v.play().catch(() => {}); }
  }, [sel]);
  useEffect(() => {
    const v = ref.current;
    if (!v) return undefined;
    const tick = () => {
      setT(v.currentTime);
      if (sel && v.currentTime >= sel.end_s) v.currentTime = sel.start_s;     // loop the alert's 5 s clip
    };
    v.addEventListener("timeupdate", tick);
    return () => v.removeEventListener("timeupdate", tick);
  }, [sel]);
  const dur = video.duration_s || 1;
  return (
    <div className="group">
      <div className="player">
        <video ref={ref} src={`/media/alerts/${video.id}/annotated.webm`} controls muted playsInline preload="metadata" />
      </div>
      <div className="timeline" aria-label="Alerts on the video's timeline">
        {alerts.map((a) => (
          <button key={a.id} title={`${a.cls} at ${a.time}`} aria-label={`${a.cls} alert at ${a.time}`}
                  style={{ left: `${(a.time_s / dur) * 100}%`, background: CLASS_COLOR[a.cls] ?? "#F59E0B",
                           outline: a.id === selected ? "2px solid #E6EDF3" : "none" }}
                  onClick={() => onSelect(a.id)} />
        ))}
        <span className="head" style={{ left: `${(t / dur) * 100}%` }} />
      </div>
      <span className="muted" style={{ fontSize: 12 }}>
        {sel ? `Looping alert #${sel.alert_id}: ${fmtTime(sel.start_s)}–${fmtTime(sel.end_s)} (5 s)` : "Click an alert to play its 5 s clip"}
      </span>
    </div>
  );
}

export default function DetectTab({ status, onChanged, onSearch }) {
  const [videos, setVideos] = useState([]);
  const [current, setCurrent] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [selected, setSelected] = useState(null);
  const [classes, setClasses] = useState(status.classes);
  const [file, setFile] = useState(null);
  const [over, setOver] = useState(false);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);

  const loadVideos = useCallback(() => api.videos().then((vs) => {
    setVideos(vs);
    setCurrent((c) => c ?? vs.find((v) => v.status === "ready")?.id ?? null);
    return vs;
  }), []);
  useEffect(() => { loadVideos().catch((e) => setError(e.message)); }, [loadVideos]);
  useEffect(() => {
    setSelected(null);
    if (current == null) { setAlerts([]); return; }
    api.alerts(current).then(setAlerts).catch((e) => setError(e.message));
  }, [current]);

  // poll the running job until it is done
  useEffect(() => {
    if (!job || job.state === "done" || job.state === "failed") return undefined;
    const t = setTimeout(() => api.job(job.id).then(async (j) => {
      setJob(j);
      if (j.state === "done") {
        await loadVideos();
        setCurrent(j.video_id);
        onChanged();
      } else if (j.state === "failed") { loadVideos(); }
    }).catch((e) => setError(e.message)), 500);
    return () => clearTimeout(t);
  }, [job, loadVideos, onChanged]);

  const upload = async () => {
    if (!file) return;
    setError(null);
    try {
      setJob({ name: file.name, state: "queued", stage: "detect", progress: 0, message: "uploading" });
      setJob(await api.upload(file, classes));
      setFile(null);
    } catch (e) { setError(e.message); setJob(null); }
  };
  const remove = async (v) => {
    if (!window.confirm(`Delete "${v.name}" and its ${v.n_alerts} alerts? This cannot be undone.`)) return;
    try {
      await api.deleteVideo(v.id);
      if (current === v.id) setCurrent(null);
      await loadVideos();
      onChanged();
    } catch (e) { setError(e.message); }
  };
  const toggle = (c) => setClasses((xs) => (xs.includes(c) ? xs.filter((x) => x !== c) : [...xs, c]));
  const video = videos.find((v) => v.id === current);
  const running = job && (job.state === "queued" || job.state === "running");

  return (
    <div className="page">
      <section className="main" style={{ flex: "1 1 420px", maxWidth: 520 }}>
        <label className={`drop${over ? " over" : ""}`}
               onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
               onDrop={(e) => { e.preventDefault(); setOver(false); setFile(e.dataTransfer.files[0] ?? null); }}>
          <svg width="36" height="36" viewBox="0 0 36 36" fill="none" aria-hidden="true"><path d="M18 24V8M11 15l7-7 7 7" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /><path d="M6 24v4h24v-4" stroke="#8B98A5" strokeWidth="2" strokeLinecap="round" /></svg>
          <b style={{ fontSize: 15 }}>{file ? file.name : "Drop a video here, or choose a file"}</b>
          <span className="muted" style={{ fontSize: 12 }}>MP4, MOV, AVI, MKV or WebM · YOLO runs on every frame</span>
          <input type="file" accept="video/*,.mkv,.avi" onChange={(e) => setFile(e.target.files[0] ?? null)} style={{ fontSize: 12, color: "#A7B4C0" }} />
        </label>
        <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 10 }}>
          <span className="label">Objects that raise an alert</span>
          <div className="chips">
            {status.classes.map((c) => <button key={c} className="chip" aria-pressed={classes.includes(c)} onClick={() => toggle(c)}>{c}</button>)}
          </div>
          <span className="muted" style={{ fontSize: 12 }}>
            An alert fires when an object is seen for 0.5 s; it is held until the object is gone for 2 s.
            Each alert keeps 5 frames before and 5 after (2 fps): a 5 s clip.
          </span>
          <button className="btn primary" disabled={!file || !classes.length || running} onClick={upload}>
            {running ? "Processing…" : "Detect alerts"}
          </button>
        </div>
        {job && <JobCard job={job} />}
        {error && <div className="error">{error}</div>}

        <div className="card">
          <div className="head2" style={{ padding: "14px 14px 10px" }}><h2>Videos</h2><span className="muted" style={{ fontSize: 12 }}>{videos.length}</span></div>
          <div className="vlist">
            {videos.length === 0 && <span className="muted" style={{ padding: "0 14px 14px", fontSize: 13 }}>No videos yet.</span>}
            {videos.map((v) => (
              <div key={v.id} className="row" style={{ borderTop: "1px solid var(--line)", gap: 0 }}>
                <button className="vitem" aria-current={v.id === current} onClick={() => setCurrent(v.id)} disabled={v.status !== "ready"} style={{ flex: 1 }}>
                  <span style={{ flex: 1, minWidth: 0 }}>
                    <b style={{ display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{v.name}</b>
                    <span className="mono muted" style={{ fontSize: 12 }}>
                      {v.status === "ready" ? `${v.n_alerts} alerts · ${v.width}×${v.height} · ${fmtTime(v.duration_s ?? 0)}` : v.status}
                    </span>
                  </span>
                </button>
                <button className="btn danger small" style={{ margin: "0 12px" }} onClick={() => remove(v)} disabled={v.status === "processing"}
                        aria-label={`Delete ${v.name}`}>Delete</button>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="main card" style={{ padding: 18 }}>
        {!video ? (
          <div className="empty">Upload a video to see its alerts here.</div>
        ) : (
          <>
            <div className="head2">
              <h2>{video.name} <span className="muted" style={{ fontWeight: 400 }}>· {alerts.length} alerts</span></h2>
              <button className="btn" onClick={() => onSearch({ collection: "alerts", videos: [video.id] })}>Search these alerts →</button>
            </div>
            <Player video={video} alerts={alerts} selected={selected} onSelect={setSelected} />
            <div style={{ overflowX: "auto" }}>
              <table className="table">
                <thead><tr><th>Clip</th><th>Type</th><th>Near</th><th>Time</th><th>YOLO</th><th>Status</th></tr></thead>
                <tbody>
                  {alerts.map((a) => (
                    <tr key={a.id} className="click" aria-current={a.id === selected} onClick={() => setSelected(a.id)}>
                      <td><img className="tthumb" src={a.thumb} alt={`${a.cls} at ${a.time}`} loading="lazy" /></td>
                      <td><span className="cls" style={{ color: CLASS_COLOR[a.cls] ?? "#F59E0B" }}>{a.cls}</span></td>
                      <td className="muted">{a.near.length ? a.near.join(", ") : "—"}</td>
                      <td className="mono">{a.time}</td>
                      <td className="mono muted">{a.conf.toFixed(2)}</td>
                      <td className="muted">{a.full ? "Embedded ✓" : "Embedded ✓ (video < 5 s)"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
