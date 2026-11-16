// Thin wrappers over the FastAPI back end (src/sfs/web/server.py). Errors carry the server's message.
async function call(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  status: () => call("/api/status"),
  search: (body) => call("/api/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  clip: (id) => call(`/api/clips/${id}`),
  map: () => call("/api/map"),
  videos: () => call("/api/videos"),
  alerts: (videoId) => call(`/api/videos/${videoId}/alerts`),
  job: (id) => call(`/api/jobs/${id}`),
  jobs: () => call("/api/jobs"),
  deleteVideo: (id) => call(`/api/videos/${id}`, { method: "DELETE" }),
  upload: (file, classes) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("classes", classes.join(","));
    return call("/api/videos", { method: "POST", body: fd });
  },
};

export function fmtSeconds(s) {
  if (s == null) return "–";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${(s / 60).toFixed(1)} min`;
}

export function fmtTime(s) {
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}
