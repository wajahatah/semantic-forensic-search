import React, { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import DetectTab from "./DetectTab.jsx";
import SearchTab from "./SearchTab.jsx";

function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 28 28" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="8" stroke="#22D3EE" strokeWidth="2" />
      <path d="M18 18l7 7" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" />
      <rect x="8" y="9" width="8" height="6" rx="1" stroke="#22D3EE" strokeWidth="1.5" />
    </svg>
  );
}

export default function App() {
  const [tab, setTab] = useState(() => (location.hash === "#detect" ? "detect" : "search"));
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [searchPreset, setSearchPreset] = useState(null);   // set by "Search these alerts" in the Detect tab

  const refresh = useCallback(() => api.status().then(setStatus).catch((e) => setError(e.message)), []);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { history.replaceState(null, "", `#${tab}`); }, [tab]);

  const goSearch = (preset) => { setSearchPreset({ ...preset, at: Date.now() }); setTab("search"); };

  return (
    <>
      <header className="top">
        <div className="brand">
          <Logo />
          <div>
            <b>Forensic Search</b>
            <small>Natural-language search over surveillance alerts</small>
          </div>
        </div>
        <nav className="tabs" role="tablist" aria-label="Dashboard">
          <button role="tab" aria-selected={tab === "detect"} onClick={() => setTab("detect")}>Detect</button>
          <button role="tab" aria-selected={tab === "search"} onClick={() => setTab("search")}>Search</button>
        </nav>
        <div className="chips-right">
          {status && (
            <>
              <span className="pill"><span className="dot" />{status.gpu ? `models loaded · ${status.gpu.replace("NVIDIA GeForce ", "")}` : "CPU only"}</span>
              <span className="pill">{status.meva_clips ? `${status.meva_clips.toLocaleString()} MEVA · ` : ""}{status.alerts} alerts</span>
            </>
          )}
        </div>
      </header>
      {error && <div className="page"><div className="error main">Back end not reachable: {error}</div></div>}
      {status && (tab === "search"
        ? <SearchTab status={status} preset={searchPreset} />
        : <DetectTab status={status} onChanged={refresh} onSearch={goSearch} />)}
    </>
  );
}
