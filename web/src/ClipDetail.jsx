import React, { useEffect, useState } from "react";
import { api, fmtTime } from "./api.js";
import { Segment } from "./Clip.jsx";

const STAGE = { lora: "LoRA (fast)", student: "re-ranker", standard: "fused (standard)", frozen: "frozen SigLIP 2", vlm: "VLM margin", final: "final" };

export default function ClipDetail({ item, onClose }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => { api.clip(item.id).then(setDetail).catch((e) => setError(e.message)); }, [item.id]);
  useEffect(() => {
    const esc = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  const isAlert = item.source === "alert";
  return (
    <div className="modal-bg" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={`Clip ${item.rank}`}>
        <button className="close" onClick={onClose} aria-label="Close">×</button>
        <div className="mleft">
          <div className="player">
            {isAlert
              ? <Segment src={item.media} start={item.start_s} end={item.end_s} controls />
              : <img src={item.media} alt="MEVA clip, context view" style={{ width: "100%", height: "100%", objectFit: "contain" }} />}
          </div>
          <span className="label">{isAlert ? "The 11 frames that were embedded (2 fps, boxes drawn)" : "Tight crops in time order (what the models see)"}</span>
          {error && <div className="error">{error}</div>}
          {!detail && !error && <span className="spin" />}
          {detail && (
            <div className={`strip${isAlert ? " wide" : ""}`}>
              {detail.frames.map((src, j) => <img key={src} src={src} alt={`frame ${j + 1}`} loading="lazy" />)}
            </div>
          )}
        </div>
        <div className="mright">
          <h2 style={{ margin: "0 40px 0 0", fontSize: 18 }}>#{item.rank} · {item.title}</h2>
          <span className={`band ${item.band}`} style={{ alignSelf: "flex-start" }}>{item.band} confidence</span>
          <dl className="kv">
            {isAlert ? (
              <>
                <dt>Alert</dt><dd>{item.cls}{item.near.length ? ` · near ${item.near.join(", ")}` : ""}</dd>
                <dt>Time</dt><dd>{fmtTime(item.time_s)} (clip {fmtTime(item.start_s)}–{fmtTime(item.end_s)})</dd>
                <dt>YOLO</dt><dd>{item.conf.toFixed(2)}</dd>
              </>
            ) : (
              <>
                <dt>Camera</dt><dd>{item.camera} · {item.site}</dd>
                <dt>When</dt><dd>{item.date} · {String(item.hour).padStart(2, "0")}h</dd>
                <dt>Label</dt><dd>{item.activity}</dd>
              </>
            )}
            {Object.entries(item.scores).filter(([, v]) => v != null).map(([k, v]) => (
              <React.Fragment key={k}><dt>{STAGE[k] ?? k}</dt><dd>{v.toFixed(3)}</dd></React.Fragment>
            ))}
            {item.vlm_prob != null && (<><dt>VLM says yes</dt><dd>{Math.round(item.vlm_prob * 100)}%</dd></>)}
          </dl>
          <p className="muted" style={{ fontSize: 12, margin: 0 }}>
            Confidence = where this clip's score falls among all 3,726 MEVA clips for the same query
            (High: top 1 %, Medium: top 5 %). It is relative, not a probability.
          </p>
        </div>
      </div>
    </div>
  );
}
