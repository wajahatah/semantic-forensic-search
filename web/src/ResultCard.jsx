import React, { useState } from "react";
import { ClipMedia } from "./Clip.jsx";

const BAND = { high: "High", medium: "Medium", low: "Low" };

export default function ResultCard({ item, showLabels, onOpen }) {
  const [hover, setHover] = useState(false);
  const mark = showLabels && item.match !== undefined ? (item.match ? " match" : " miss") : "";
  const score = item.scores.final;
  return (
    <button className={`result${mark}`} onClick={() => onOpen(item)} onMouseEnter={() => setHover(true)}
            onMouseLeave={() => setHover(false)} onFocus={() => setHover(true)} onBlur={() => setHover(false)}
            aria-label={`Result ${item.rank}: ${item.title}, ${item.meta}, ${BAND[item.band]} confidence`}>
      <div className="thumb">
        <ClipMedia item={item} hover={hover} />
        <span className="badge tl">#{item.rank}</span>
        <span className={`badge tr src${item.source === "alert" ? " alert" : ""}`}>{item.source === "alert" ? "your video" : "MEVA"}</span>
        {item.kind === "video" && <span className="badge br">5.0 s</span>}
      </div>
      <div className="rbody">
        <div className="rtitle">
          <span>{item.title}</span>
          <span className={`band ${item.band}`}>{BAND[item.band]}</span>
        </div>
        <span className="rmeta">{item.meta}</span>
        {showLabels && item.activity && <span className="rmeta">label: {item.activity}{item.match !== undefined ? (item.match ? " ✓" : " ✗") : ""}</span>}
        <span className="rscore">
          score {score == null ? "–" : score.toFixed(2)}
          {item.vlm_prob != null && ` · VLM yes ${Math.round(item.vlm_prob * 100)}%`}
        </span>
      </div>
    </button>
  );
}
