import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { api } from "./api.js";

// 3D clip map (owner D5): every clip is a point (PCA of its LoRA embedding, sfs/web/clipmap.py).
// Similar clips sit close together; the top results light up; the query sits at the centre of its top 10.
// An illustration only: the search never uses these positions.
const SCALE = 10;

function points(coords, color, size) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(coords.flat().map((v) => v * SCALE), 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ color, size, sizeAttenuation: true, transparent: true, opacity: 0.85 }));
}

export default function ClipMap({ results, queryPoint, onOpen }) {
  const box = useRef(null);
  const [tags, setTags] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stop = false;
    let cleanup = () => {};
    api.map().then((m) => {        // ~4k points, local: fetched fresh so new alerts always appear
      if (stop || !box.current) return;
      const el = box.current;
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 0.1, 400);
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      renderer.setSize(el.clientWidth, el.clientHeight);
      renderer.setClearColor(0x070a0e);
      el.prepend(renderer.domElement);
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 0.6;

      const cloud = points(m.meva.points, 0x7687ad, 0.2);
      scene.add(cloud);
      if (m.alerts.points.length) scene.add(points(m.alerts.points, 0x2dd4bf, 0.42));
      // frame the whole MEVA cloud: look at its centre from outside its bounding sphere
      cloud.geometry.computeBoundingSphere();
      const { center, radius } = cloud.geometry.boundingSphere;
      const grid = new THREE.GridHelper(radius * 2.4, 24, 0x1f2a37, 0x131b25);
      grid.position.set(center.x, center.y - radius * 0.9, center.z);
      scene.add(grid);
      camera.position.copy(center).add(new THREE.Vector3(1, 0.55, 1.15).normalize().multiplyScalar(radius * 2.3));
      controls.target.copy(center);
      const hits = results.slice(0, 12);
      const hitPos = hits.map((r) => new THREE.Vector3(...r.point.map((v) => v * SCALE)));
      hitPos.forEach((p, i) => {
        const s = new THREE.Mesh(new THREE.SphereGeometry(i < 3 ? 0.36 : 0.26, 16, 12), new THREE.MeshBasicMaterial({ color: 0xf59e0b }));
        s.position.copy(p);
        scene.add(s);
      });
      let q = null;
      if (queryPoint) {
        q = new THREE.Vector3(...queryPoint.map((v) => v * SCALE));
        const star = new THREE.Mesh(new THREE.OctahedronGeometry(0.6), new THREE.MeshBasicMaterial({ color: 0x22d3ee, wireframe: true }));
        star.position.copy(q);
        scene.add(star);
        hitPos.slice(0, 5).forEach((p) => {
          const g = new THREE.BufferGeometry().setFromPoints([q, p]);
          scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.35 })));
        });
      }

      const project = () => hitPos.map((p, i) => {
        const v = p.clone().project(camera);
        return { i, x: (v.x * 0.5 + 0.5) * el.clientWidth, y: (-v.y * 0.5 + 0.5) * el.clientHeight, show: v.z < 1 };
      });
      let raf = 0;
      let frame = 0;
      const loop = () => {
        raf = requestAnimationFrame(loop);
        controls.update();
        renderer.render(scene, camera);
        if (frame++ % 3 === 0) setTags(project());
      };
      loop();
      const onResize = () => {
        camera.aspect = el.clientWidth / el.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(el.clientWidth, el.clientHeight);
      };
      window.addEventListener("resize", onResize);
      cleanup = () => {
        cancelAnimationFrame(raf);
        window.removeEventListener("resize", onResize);
        controls.dispose();
        renderer.dispose();
        renderer.domElement.remove();
      };
    }).catch((e) => setError(e.message));
    return () => { stop = true; cleanup(); };
  }, [results, queryPoint]);

  return (
    <div className="map" ref={box}>
      {error && <div className="error" style={{ margin: 16 }}>{error}</div>}
      <div className="legend" aria-hidden="true">
        <b style={{ fontSize: 13 }}>Clip map</b>
        <span>Similar clips sit close together · drag to rotate, scroll to zoom</span>
        <span><i className="dot" style={{ background: "#5b6b8c" }} />MEVA clips</span>
        <span><i className="dot" style={{ background: "#2dd4bf" }} />your alerts</span>
        <span><i className="dot" style={{ background: "#f59e0b" }} />top results</span>
        <span><i className="dot" style={{ background: "#22d3ee" }} />query (centre of its top 10)</span>
      </div>
      {tags.map((t) => t.show && (
        <button key={t.i} className="tag" style={{ left: t.x, top: t.y }} onClick={() => onOpen(results[t.i])}
                aria-label={`Open result ${t.i + 1}`}>{t.i + 1}</button>
      ))}
      <span className="note">3D projection (PCA) of the LoRA clip embeddings · illustration only</span>
    </div>
  );
}

