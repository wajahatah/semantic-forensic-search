// Builds the 6-slide presentation, report/Ahmed_Wajahat_DLPresentation_6.pptx, to the owner's outline
// (docs/planning/spec-submission.md §3), from report/numbers.json and report/figures/. Theme, layouts and helpers:
// scripts/deck_common.js (shared with the 12-slide deck). Speaker notes carry the timings (about 7 minutes).
//
//   NODE_PATH=report/build/node_modules node scripts/deck6_build.js
//
"use strict";
const fs = require("fs");
const path = require("path");
const { ROOT, N, H, num, createDeck } = require("./deck_common");
const { D, f, f2, base, lora, cas, g1, bsd, sg } = num;
// DECK_OUT writes elsewhere, e.g. while the deck is open in PowerPoint (Windows locks the open file)
const OUT = process.env.DECK_OUT || path.join(ROOT, "report", "Ahmed_Wajahat_DLPresentation_6.pptx");

const { pres, C, stat, pill, chartFrame, img, write } =
  createDeck("Cost-Bounded Natural-Language Forensic Search over Surveillance Video");

const CV = N.clip.val.encoders;                                   // zero-shot validation, Result 01 protocol
const CE = (P, q, m) => N.clip.eval.protocols[P].systems["clip-tight"][q][m];
const RUN = { A: "A-middle", B: "B-uniform" };                    // LoRA run selected on validation per protocol
const vlmS = (blk) => N.cascade[blk].vlm_seconds_per_call.instruct;

// CLIP's validation AUC lead without its one outlier activity (Result 01 addendum, D-067)
const without = (enc, act) => {
  const a = Object.entries(CV[enc].per_activity_auc).filter(([k]) => k !== act).map(([, v]) => v);
  return a.reduce((s, v) => s + v, 0) / a.length;
};

// ================================================================ slides
// 1 introduction: cover with the dashboard on the right
pres.defineSlideMaster({
  title: "COVER", background: { color: H.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.55, y: 1.55, w: 4.6, h: 1.75, fontSize: 28, bold: true, color: C.text1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.55, y: 3.35, w: 4.4, h: 0.7, fontSize: 14, color: C.text2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "COVER", sectionTitle: "Introduction" });
// cover_dash.png: the owner's dashboard screenshot (cover_screenshot.png) cropped to the panel shape (columns 490..1588)
s.addImage({ path: path.join(ROOT, "report", "figures", "cover_dash.png"), x: 5.4, y: 0, w: 4.6, h: 5.625, altText: "The search dashboard: the query persons chatting and the ranked alerts from uploaded CCTV videos", objectName: "cover screenshot" });
s.addShape(pres.ShapeType.rect, { x: 5.4, y: 0, w: 4.6, h: 5.625, fill: { color: H.lt1, transparency: 60 }, line: { color: H.lt1, width: 0 }, objectName: "cover shade" });
s.addShape(pres.ShapeType.rect, { x: 5.4, y: 0, w: 0.04, h: 5.625, fill: { color: H.accent1 }, line: { color: H.accent1, width: 0 }, objectName: "cover edge" });
pill(s, 0.55, 0.8, 3.3, "persons chatting", C.accent1);
s.addText("Cost-Bounded Natural-Language Forensic Search over Surveillance Video", { placeholder: "title" });
s.addText("Type what happened. Get the clips. Spend almost no VLM calls.", { placeholder: "body" });
s.addText([{ text: "Wajahat Ahmed", options: { bold: true, color: H.dk1, breakLine: true } }, { text: "CS-141/2026 · CS-5138 Deep Learning (Graduate)", options: { breakLine: true } }, { text: "NED University · November 2026", options: {} }],
  { x: 0.55, y: 4.3, w: 4.6, h: 0.85, fontSize: 11.5, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "author block" });
s.addNotes("0:00–0:30. Introduce yourself: Wajahat Ahmed, CS-141/2026, CS-5138 Deep Learning. The project in one line: search hours of CCTV by typing a sentence such as the one in the pill, and get the matching clips back, without paying for a large vision-language model on every clip. The picture on the right is the working dashboard.");

// 2 project description: the idea as a picture, the cost gap as two numbers
pres.addSection({ title: "Project" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Project" });
s.addText("Search CCTV by typing what happened", { placeholder: "title" });
{
  const cap = (x, y, w, text, color) => s.addText(text, { x, y, w, h: 0.26, fontSize: 10, bold: true, color, align: "center", margin: 0, isTextBox: true, charSpacing: 1.5, objectName: `cap ${text}` });
  cap(0.5, 1.45, 3.5, "YOU TYPE", C.accent1);
  pill(s, 0.5, 1.78, 3.5, "a person opening a car door", C.accent1, { h: 0.5, size: 13 });
  s.addShape(pres.ShapeType.line, { x: 4.15, y: 2.03, w: 0.95, h: 0, line: { color: H.accent1, width: 2.5, endArrowType: "triangle" }, objectName: "idea arrow" });
  cap(5.3, 1.08, 4.2, "YOU GET", C.accent3);
  const hImg = img(s, "deck_meva_strip.png", 5.3, 1.36, 4.2, "Ranked MEVA clips returned for the query");
  s.addShape(pres.ShapeType.roundRect, { x: 5.3, y: 1.36, w: 4.2, h: hImg, fill: { color: H.lt1, transparency: 100 }, line: { color: H.accent3, width: 1 }, rectRadius: 0.04, objectName: "strip frame" });
  const Y = 3.45, tiles = [
    [H.accent5, "≈ 10 min", "per query if a VLM reads every clip"],
    [H.accent3, "0.2 s", "per query with our system, 0 VLM calls"],
    [H.accent1, D.instances.toLocaleString(), `MEVA clips · ${D.camera.distinct} cameras · 66 test queries`],
  ];
  tiles.forEach(([color, big, lab], i) => {
    const x = 0.5 + i * 3.07;
    s.addShape(pres.ShapeType.roundRect, { x, y: Y, w: 2.87, h: 1.45, fill: { color: C.background2 }, line: { color: "1F2A37", width: 0.75 }, rectRadius: 0.08, objectName: `tile ${i + 1}` });
    s.addShape(pres.ShapeType.rect, { x: x + 0.18, y: Y + 0.2, w: 0.06, h: 1.05, fill: { color }, line: { color, width: 0 }, objectName: `tile mark ${i + 1}` });
    s.addText(big, { x: x + 0.4, y: Y + 0.14, w: 2.35, h: 0.68, fontSize: 30, bold: true, color, margin: 0, isTextBox: true, valign: "middle", objectName: `tile big ${i + 1}` });
    s.addText(lab, { x: x + 0.4, y: Y + 0.84, w: 2.35, h: 0.5, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: `tile label ${i + 1}` });
  });
}
s.addNotes("0:30–1:45. The idea in one picture: you type what happened, you get the matching clips, ranked. Why it is hard: CCTV archives are not indexed by content, and the obvious tool, a vision-language model that reads every clip, takes about 10 minutes for one query over 3,726 clips. Our system answers in about 0.2 seconds with no VLM call. Dataset: MEVA, public, real CCTV of scripted everyday activities, 10,678 clips from 24 cameras; it had no text queries, so we wrote 66 (22 activities × 3 wordings) and froze them before any result. Two measurements, used on the results slide: Q1, find every clip of an activity (mAP); Q2, the investigator's case, find the event on a known camera and hour (Success@1: is the right clip first).");

// 3 methodology: one row per stage, each a short chain of steps ending in its output
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Methodology: how the system was built, step by step", { placeholder: "title" });
{
  const LX = 0.5, LW = 0.95, BW = 1.42, GAP = 0.24, BH = 0.98, Y = [1.1, 2.33, 3.56];
  const bx = (j) => LX + LW + j * (BW + GAP);
  const txt = (x, y, w, h, runs, size, name, o = {}) => s.addText(runs, { x, y, w, h, fontSize: size, color: o.color || C.text1, bold: !!o.bold, align: o.align || "center", valign: o.valign || "middle", margin: 0.04, isTextBox: true, objectName: name });
  const rows = [
    [C.accent1, "1", "Baseline", [
      ["MEVA clips", `${D.instances.toLocaleString()} activity clips, ${D.camera.distinct} cameras`],
      ["Person crops", "2 fps, up to 16 frames per clip"],
      ["3 encoders, no training", "CLIP · SigLIP · SigLIP 2"],
      ["Test on validation", "66 frozen queries"],
      ["SigLIP 2 + person crop", `top-10 precision ${f2(CV.siglip2.p10)} (SigLIP ${f2(CV.siglip.p10)}; CLIP ${f2(CV.clip.p10)}, added later)`]]],
    [H.accent4, "2", "Fine-tune", [
      ["SigLIP 2", "from stage 1, weights frozen"],
      ["+ LoRA adapters", "1.18 M trainable (1.25 %)"],
      ["Train on 192 captions", "crop ↔ its activity's captions"],
      ["4 runs, keep the best", "A / B × 2 frame samplings, picked on validation"],
      [`Q1 mAP
${f(lora("A-middle", "q1", "ap"))} / ${f(lora("B-uniform", "q1", "ap"))}`, `was ${f(base("A", "siglip2-tight", "q1", "ap"))} / ${f(base("B", "siglip2-tight", "q1", "ap"))} (protocol A / B)`]]],
    [C.accent3, "3", "Re-rank", [
      ["Per-frame input", "LoRA crop + context + box motion"],
      ["Temporal transformer", "2 layers, 2.2 M parameters"],
      ["Train on human labels", "both protocols, ~12,200 pairs each"],
      ["Add to LoRA's score", "1 : 1, chosen on validation"],
      [`Q1 mAP
${f(cas("2b", "A", "q1", "student", "ap"))} / ${f(cas("2b", "B", "q1", "student", "ap"))}`, `Success@1 ${f2(cas("2b", "A", "q2", "student", "s1"))} / ${f2(cas("2b", "B", "q2", "student", "s1"))}, 0 VLM calls`]]],
  ];
  rows.forEach(([color, n, name, steps], r) => {
    const y = Y[r];
    s.addShape(pres.ShapeType.ellipse, { x: LX + 0.22, y: y + 0.08, w: 0.5, h: 0.5, fill: { color: C.background1 }, line: { color, width: 1.5 }, objectName: `stage num ${n}` });
    txt(LX + 0.22, y + 0.08, 0.5, 0.5, n, 16, `stage num text ${n}`, { bold: true, color });
    txt(LX, y + 0.62, LW - 0.05, 0.32, name, 11, `stage name ${n}`, { bold: true, color });
    steps.forEach(([head, sub], j) => {
      const x = bx(j), out = j === steps.length - 1;
      s.addShape(pres.ShapeType.roundRect, { x, y, w: BW, h: BH, fill: { color: C.background2 }, line: { color: out ? color : "334155", width: out ? 2 : 1 }, rectRadius: 0.07, objectName: `step ${n}.${j + 1}` });
      txt(x + 0.04, y + 0.06, BW - 0.08, 0.4, head, out ? 11.5 : 10.5, `step head ${n}.${j + 1}`, { bold: true, color: out ? color : C.text1, valign: "bottom" });
      txt(x + 0.04, y + 0.48, BW - 0.08, BH - 0.52, sub, 9, `step sub ${n}.${j + 1}`, { color: C.text2, valign: "top" });
      if (j < steps.length - 1) s.addShape(pres.ShapeType.line, { x: x + BW + 0.03, y: y + BH / 2, w: GAP - 0.06, h: 0, line: { color: H.dk2, width: 1.5, endArrowType: "triangle" }, objectName: `arrow ${n}.${j + 1}` });
    });
    txt(bx(4), y - 0.2, BW, 0.2, "OUTPUT", 8, `output tag ${n}`, { bold: true, color, valign: "bottom" });
  });
  [["A", "Official split", "test cameras also seen in training", C.accent1, 0.5],
   ["B", "Unseen cameras", "8 of 24 cameras kept out of training", C.accent3, 5.05]].forEach(([k, head, body, color, x]) => {
    s.addShape(pres.ShapeType.roundRect, { x, y: 4.66, w: 4.45, h: 0.46, fill: { color: C.background2 }, line: { color, width: 1 }, rectRadius: 0.08, objectName: `protocol ${k}` });
    s.addText([{ text: `Protocol ${k} · ${head}  `, options: { bold: true, color } }, { text: body, options: { color: H.dk1 } }],
      { x: x + 0.15, y: 4.66, w: 4.2, h: 0.46, fontSize: 11, margin: 0, isTextBox: true, valign: "middle", objectName: `protocol text ${k}` });
  });
}
s.addNotes("1:45–3:00. Read each row left to right; the last box of a row is its result. Row 1, baseline: MEVA clips are cut to person crops at 2 fps; three encoders score them against the 66 frozen queries on the validation split without any training; SigLIP 2 on the person crop wins (CLIP was added later as a reference and does not change that). Row 2, fine-tune: SigLIP 2's weights stay frozen, LoRA adds 1.18 M trainable parameters, trained on 192 captions; four runs, the best per protocol kept on validation: Q1 mAP 0.179 → 0.313, and 0.174 → 0.280 on cameras never seen in training. Row 3, re-rank: each frame becomes the LoRA crop vector, a context vector and the box motion; the temporal transformer is trained on the human labels for both protocols and its score is added to LoRA's: 0.476 / 0.442 mAP, Success@1 0.53 / 0.52, with no VLM call.");

// 4 architecture: flow diagram of the three models (top) and one real clip through them (bottom)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Model architecture", { placeholder: "title" });
const EX = JSON.parse(fs.readFileSync(path.join(ROOT, "report", "figures", "arch_example.json"), "utf8"));
const vec = (v) => `[${v.map((x) => x.toFixed(3).replace("-", "−")).join(", ")}, …]`;
function block(x, y, w, h, color, name) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: C.background2 }, line: { color, width: 1.25 }, rectRadius: 0.08, objectName: name });
}
function label(x, y, w, h, runs, o = {}) {
  s.addText(runs, { x, y, w, h, fontSize: o.size || 10.5, color: o.color || C.text1, bold: !!o.bold, align: o.align || "left", valign: o.valign || "top", margin: 0, isTextBox: true, objectName: `${o.name || "label"} ${x}-${y}` });
}
function arrow(x1, y, x2, name, text) {
  s.addShape(pres.ShapeType.line, { x: x1, y, w: x2 - x1, h: 0, line: { color: H.dk2, width: 1.75, endArrowType: "triangle" }, objectName: name });
  if (text) label(x1 - 0.05, y - 0.36, x2 - x1 + 0.1, 0.32, text, { size: 9, color: C.text2, align: "center", valign: "bottom", name: `${name} text` });
}
// ---- top: the blocks; what flows between them is written on the arrows
const T = 1.12, TH = 1.4;
label(0.5, T + 0.28, 1.0, 0.5, "person\ncrops", { size: 11, bold: true, align: "center", valign: "middle", name: "in crops" });
arrow(0.6, T + 0.95, 1.5, "arrow in");
block(1.55, T, 3.0, TH, C.accent1, "siglip block");
label(1.7, T + 0.08, 2.7, 0.3, "SigLIP 2 image encoder", { size: 13, bold: true, color: C.accent1, name: "siglip head" });
label(1.7, T + 0.38, 2.7, 0.22, "ViT-B/16 · 12 layers · 94 M weights, frozen", { size: 9.5, color: C.text2, name: "siglip spec" });
s.addShape(pres.ShapeType.roundRect, { x: 1.7, y: T + 0.68, w: 2.7, h: 0.58, fill: { color: C.background1 }, line: { color: H.accent4, width: 1.25, dashType: "dash" }, rectRadius: 0.06, objectName: "lora block" });
label(1.8, T + 0.71, 2.5, 0.52, [
  { text: "LoRA, trained, inside it", options: { bold: true, color: H.accent4, breakLine: true } },
  { text: "rank 16 · all 12 layers · 1.18 M", options: {} },
], { size: 9.5, name: "lora text" });

arrow(4.62, T + TH / 2, 5.48, "arrow 1", "768 / frame");

block(5.55, T, 2.25, TH, C.accent3, "transformer block");
label(5.68, T + 0.08, 2.0, 0.3, "Temporal transformer", { size: 13, bold: true, color: C.accent3, name: "tf head" });
label(5.68, T + 0.38, 2.0, 0.22, "2 layers · 2.2 M, trained", { size: 9.5, color: C.text2, name: "tf spec" });
label(5.68, T + 0.72, 2.0, 0.55, "reads the query + all frames, in order", { size: 10.5, name: "tf reads" });

arrow(7.87, T + TH / 2, 8.38, "arrow 2", "1 score");

block(8.45, T, 1.05, TH, C.accent2, "ranking block");
label(8.5, T + 0.08, 0.95, 0.3, "Ranking", { size: 13, bold: true, color: C.accent2, align: "center", name: "rank head" });
label(8.5, T + 0.5, 0.95, 0.8, "+ LoRA score → best first", { size: 10, align: "center", name: "rank text" });

// ---- bottom: one real clip through the blocks
const B = 2.85;
label(0.5, B, 9.0, 0.28, [
  { text: "Example  ", options: { bold: true, color: H.accent1 } },
  { text: `"${EX.query}" · one test clip, its rank among ${EX.corpus.toLocaleString()}`, options: {} },
], { size: 10.5, name: "example head" });
const fw = 1.45, fy = B + 0.4;
for (let k = 0; k < EX.frames_shown; k++) {
  s.addImage({ path: path.join(ROOT, "report", "figures", `arch_frame_${k}.png`), x: 0.5 + k * (fw + 0.08), y: fy, w: fw, h: fw, altText: `Example clip, frame ${k + 1}: person crop`, objectName: `example frame ${k}` });
}
const bx0 = 0.5 + EX.frames_shown * (fw + 0.08) + 0.3, bw = (9.5 - bx0 - 2 * 0.32) / 3, bh = 1.55;
const steps = [
  [C.accent1, "SigLIP 2, frozen", `${vec(EX.frame0_frozen_first3)}`, `rank ${EX.rank_frozen}`],
  [H.accent4, "+ LoRA", `${vec(EX.frame0_lora_first3)}`, `rank ${EX.rank_lora}`],
  [C.accent3, "+ transformer", "sees the motion: bends down → stands up", `rank ${EX.rank_standard}`],
];
arrow(bx0 - 0.27, fy + fw / 2, bx0 - 0.03, "arrow ex0");
steps.forEach(([color, head, body, rk], i) => {
  const x = bx0 + i * (bw + 0.32);
  block(x, fy - 0.05, bw, bh, color, `example step ${i}`);
  label(x + 0.1, fy, bw - 0.2, 0.26, head, { size: 11, bold: true, color, name: `ex head ${i}` });
  label(x + 0.1, fy + 0.3, bw - 0.2, 0.45, body, { size: 9.5, color: C.text2, name: `ex body ${i}` });
  label(x + 0.1, fy + 0.85, bw - 0.2, 0.55, rk, { size: 26, bold: true, color, name: `ex rank ${i}` });
  if (i < 2) arrow(x + bw + 0.03, fy + fw / 2, x + bw + 0.29, `arrow ex${i + 1}`);
});
s.addNotes(`3:00–4:15. Top row, left to right. (1) SigLIP 2's image encoder turns each person crop into 768 numbers. Its 94 M weights stay frozen; LoRA adds small trained matrices inside all 12 attention layers (48 adapters, 1.18 M parameters), so the output is still 768 numbers per frame, only adapted to surveillance activities. The query sentence goes through SigLIP 2's text encoder, unchanged, into 768 numbers. (2) The temporal transformer gets the query as its first token and one token per frame (the LoRA vector, a frozen vector of the wider context crop and 12 box-motion numbers), and outputs one score for the clip. (3) That score is added to the LoRA score and the clips are sorted. Bottom row: a real test clip for "${EX.query}". Frozen SigLIP 2 puts it at rank ${EX.rank_frozen}, LoRA at ${EX.rank_lora}; the transformer sees the change between the two frames, bending down then standing up with the item, and puts it first. (One example to show the flow; the measured gains are on the next slide.)`);

// 5 results and comparison
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Results: every stage adds, also on unseen cameras", { placeholder: "title" });
const stages = ["Zero-shot CLIP", "Frozen SigLIP 2", "+ LoRA", "+ student"];
const q1 = (P) => [CE(P, "q1", "ap"), base(P, "siglip2-tight", "q1", "ap"), lora(RUN[P], "q1", "ap"), cas("2b", P, "q1", "student", "ap")];
const q2 = (P) => [CE(P, "q2", "s1"), cas("2b", P, "q2", "frozen", "s1"), lora(RUN[P], "q2", "s1"), cas("2b", P, "q2", "student", "s1")];
const barOpts = (title, max) => ({
  barDir: "col", barGrouping: "clustered", barGapWidthPct: 55, chartColors: [H.accent1, H.accent3],
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: H.dk1, dataLabelFontSize: 9, dataLabelFormatCode: "0.00",
  valAxisMinVal: 0, valAxisMaxVal: max, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", showLegend: true, legendPos: "b",
  title, ...chartFrame(),
});
s.addChart(pres.ChartType.bar, [
  { name: "A · official split", labels: stages, values: q1("A") },
  { name: "B · unseen cameras", labels: stages, values: q1("B") },
], { x: 0.5, y: 1.05, w: 4.45, h: 3.0, ...barOpts("Q1 category search · mAP", 0.6) });
s.addChart(pres.ChartType.bar, [
  { name: "A · official split", labels: stages, values: q2("A") },
  { name: "B · unseen cameras", labels: stages, values: q2("B") },
], { x: 5.05, y: 1.05, w: 4.45, h: 3.0, ...barOpts("Q2 forensic search · Success@1", 0.7) });
{
  const gainShare = (lora("B-uniform", "q1", "ap") - base("B", "siglip2-tight", "q1", "ap")) / (lora("A-middle", "q1", "ap") - base("A", "siglip2-tight", "q1", "ap"));
  [[`${f(base("A", "siglip2-tight", "q1", "ap"))} → ${f(cas("2b", "A", "q1", "student", "ap"))}`, "Q1 mAP, frozen → full system (A), 0 VLM calls", C.accent3],
   [`≈ ${Math.round(gainShare * 10) * 10} %`, "of LoRA's gain holds on cameras never seen in training", C.accent1],
   ["SigLIP 2 > CLIP", "on every test metric, both protocols", C.accent4]].forEach(([big, lab, color], i) => {
    stat(s, 0.5 + i * 3.05, 4.12, 2.85, big, lab, { bigSize: 22, bigH: 0.45, labelH: 0.45, labelSize: 10.5, color });
  });
}
s.addNotes(`4:15–5:45. Read the charts left to right. Q1 mAP on A: CLIP ${f(CE("A", "q1", "ap"))}, frozen SigLIP 2 ${f(base("A", "siglip2-tight", "q1", "ap"))}, LoRA ${f(lora("A-middle", "q1", "ap"))}, + student ${f(cas("2b", "A", "q1", "student", "ap"))}. Forensic Success@1 on A: ${f2(CE("A", "q2", "s1"))} → ${f2(cas("2b", "A", "q2", "frozen", "s1"))} → ${f2(lora("A-middle", "q2", "s1"))} → ${f2(cas("2b", "A", "q2", "student", "s1"))}: the right clip is first for about half of the forensic queries, with no VLM call. Protocol B (green) repeats every step on cameras never seen in training, so it is not camera memorisation. CLIP was measured last, as a reference: nothing was chosen on it.`);

// 6 VLM part + thanks + questions
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("VLMs: a final check on the top 10, not a judge of every clip", { placeholder: "title" });
const labels = ["0 calls", "top 5", "top 10", "top 20"];
const series = (blk) => [cas(blk, "A", "q2", "student", "s1"), ...["5", "10", "20"].map((k) => cas(blk, "A", "q2", `student>instruct@${k}`, "s1"))];
s.addChart(pres.ChartType.line, [
  { name: "student → 2B", labels, values: series("2b") },
  { name: "student → 8B", labels, values: series("8b") },
  { name: "student → 32B", labels, values: series("32b") },
], { x: 0.5, y: 1.05, w: 4.7, h: 3.05, chartColors: [H.accent3, H.accent4, H.accent2], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 6,
  valAxisMinVal: 0.5, valAxisMaxVal: 0.62, valAxisMajorUnit: 0.02, valAxisLabelFormatCode: "0.00", showLegend: true, legendPos: "b",
  title: "Protocol A · Success@1 against VLM calls per query", ...chartFrame() });
{
  const rows = [["2B", "2b_top50", "instruct-tight"], ["2B rerank", "2b_top50", "reranker-tight"], ["2B 16 fr.", "2b_16frames", "instruct-tight"],
    ["Gemma 4B", "gemma_top50", "gemma-e4b-tight"], ["Qwen2.5 7B", "qwen25_7b_top50", "q25-7b-fp8-tight"], ["8B", "8b_top50", "instruct-tight"], ["32B", "32b_top50", "instruct-tight"]];
  const gain = rows.map(([, k, c]) => Math.max(g1(k, c, "A"), g1(k, c, "B")));       // the gate passes on either protocol
  s.addChart(pres.ChartType.bar, [{ name: "validation mAP added", labels: rows.map((r) => r[0]), values: gain }], {
    x: 5.4, y: 1.05, w: 4.1, h: 3.05, barDir: "bar", barGapWidthPct: 45, chartColors: gain.map((v) => (v >= 0.02 ? H.accent3 : "64748B")),
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: H.dk1, dataLabelFontSize: 9, dataLabelFormatCode: "+0.000",
    valAxisMinVal: 0, valAxisMaxVal: 0.04, valAxisMajorUnit: 0.02, valAxisLabelFormatCode: "0.00", catAxisOrientation: "maxMin", showLegend: false,
    title: "7 VLMs tested: what each adds (pass ≥ +0.02)", ...chartFrame() });
}
s.addText("Adapt the encoder, re-rank cheaply, spend the VLM calls at the end.", { x: 0.5, y: 4.22, w: 9.0, h: 0.45, fontSize: 19, bold: true, color: C.accent1, margin: 0, isTextBox: true, align: "center", valign: "middle", objectName: "takeaway" });
s.addText("Thank you — questions?", { x: 0.5, y: 4.7, w: 9.0, h: 0.38, fontSize: 15, color: C.text1, margin: 0, isTextBox: true, align: "center", valign: "middle", objectName: "thanks" });
s.addNotes(`5:45–7:00. Seven VLMs from 2B to 32B were tested as re-rankers through two gates fixed in advance; only the 32B passed. Inside the retriever's own short list the VLMs rank right clips above wrong ones less well than the adapted retriever does, which is why they fail. On the curve, the 32B on the student's top 10 adds ${sg(bsd("32b", "A: 32B@10 - student").diff)} Success@1 on A and ${sg(bsd("32b", "B: 32B@10 - student").diff)} on B, both significant; the 8B adds ${sg(bsd("8b", "A: 8B@10 - student").diff)} on A (significant) and ${sg(bsd("8b", "B: 8B@10 - student").diff)} on B (not); the 2B at most +0.03. Going from 10 to 20 calls adds less than +0.01. Distilling any of these VLMs into the student did not beat training on the human labels. Close with the takeaway line, thank the professor and the audience, and take questions.`);

// ---------------------------------------------------------------- write + theme
write(OUT).catch((e) => { console.error(e); process.exit(1); });
