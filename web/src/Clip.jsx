import React, { useEffect, useRef } from "react";

// An alert clip is the 5 s segment [start, end] of its video's annotated WebM: play it in a loop.
export function Segment({ src, start, end, playing = true, controls = false, className }) {
  const ref = useRef(null);
  useEffect(() => {
    const v = ref.current;
    if (!v) return undefined;
    const toStart = () => { v.currentTime = start; };
    const loop = () => { if (v.currentTime >= end || v.currentTime < start - 0.05) v.currentTime = start; };
    v.addEventListener("loadedmetadata", toStart);
    v.addEventListener("timeupdate", loop);
    if (v.readyState >= 1) toStart();
    return () => { v.removeEventListener("loadedmetadata", toStart); v.removeEventListener("timeupdate", loop); };
  }, [src, start, end]);
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    if (playing) v.play().catch(() => {});
    else v.pause();
  }, [playing, src]);
  return <video ref={ref} className={className} src={`${src}#t=${start},${end}`} muted playsInline preload="metadata" controls={controls} />;
}

export function ClipMedia({ item, hover }) {
  if (item.kind === "video") {
    return hover
      ? <Segment src={item.media} start={item.start_s} end={item.end_s} />
      : <img src={item.thumb} alt={`${item.cls} alert at ${item.meta}`} loading="lazy" />;
  }
  return <img src={item.media} alt={`MEVA clip, ${item.meta}`} loading="lazy" />;
}
