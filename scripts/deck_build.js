// Builds the 10-minute presentation, report/Ahmed_Wajahat_DLPresentation.pptx (12 slides + speaker notes),
// from report/numbers.json and report/figures/ (docs/planning/spec-report.md §7). Theme, layouts and helpers:
// scripts/deck_common.js (shared with the 6-slide deck, scripts/deck6_build.js).
//
//   NODE_PATH=report/build/node_modules node scripts/deck_build.js
//
"use strict";
const path = require("path");
const { ROOT, N, H, num, createDeck } = require("./deck_common");
const { D, f, f2, base, lora, cas, g1, g0, bsd, pa, sg } = num;
const OUT = path.join(ROOT, "report", "Ahmed_Wajahat_DLPresentation.pptx");

const { pres, C, nm, card, stat, pill, bullets, chartFrame, img, write } =
  createDeck("Cost-Bounded Natural-Language Forensic Search over Surveillance Video");

// ================================================================ slides
// 1 title
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Opening" });
s.addText("Cost-Bounded Natural-Language Forensic Search over Surveillance Video", { placeholder: "title" });
s.addText("Domain-adapted retrieval and a learned re-ranker on MEVA, under a budget of VLM calls", { placeholder: "body" });
pill(s, 3.0, 0.75, 4.0, "a person opening a car door", C.accent1);
s.addText("Wajahat Ahmed · CS-141/2026 · CS-5138 Deep Learning (Graduate) · NED University · November 2026", { x: 0.7, y: 4.35, w: 8.6, h: 0.4, fontSize: 12, color: C.text2, align: "center", margin: 0, isTextBox: true, objectName: "author line" });
s.addNotes("0:00–0:15. Title. The query in the pill is the kind of input the system takes; the output is a ranked list of clips and the cost of the query. Final report: Ahmed_Wajahat_DLReport.pdf.");

// 2 problem
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
s.addText("The problem: find what happened without a VLM call per clip", { placeholder: "title" });
s.addText([
  { text: "Surveillance sites raise thousands of alerts a day. After an incident, investigators scrub footage by camera and time: nothing indexes what the footage shows.", options: { breakLine: true, paraSpaceAfter: 10 } },
  { text: "Goal: type what happened, get the right clips back.", options: { breakLine: true, paraSpaceAfter: 10, bold: true } },
  { text: "A vision-language model (VLM) can judge each clip, but one query over an archive then costs one VLM call per clip.", options: {} },
], { x: 0.5, y: 1.15, w: 5.3, h: 2.1, fontSize: 15, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "problem text" });
s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 3.4, w: 5.3, h: 1.45, fill: { color: C.background2 }, line: { color: C.accent1, width: 1 }, rectRadius: 0.08, objectName: "rq box" });
s.addText([{ text: "Research question  ", options: { bold: true, color: H.accent1 } },
  { text: "How much of the accuracy of an exhaustive VLM system can be kept while making 10–100× fewer VLM calls?", options: { italic: true } }],
  { x: 0.7, y: 3.5, w: 4.9, h: 1.25, fontSize: 15, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "rq text" });
stat(s, 6.3, 1.15, 3.2, D.split.counts.eval.toLocaleString(), "clips in the evaluation corpus; a forensic query searches one camera-hour (≈ 56 clips)");
stat(s, 6.3, 2.45, 3.2, "≈ 10 min", `one query if a 2B VLM reads every clip (${f2(N.cascade["2b"].vlm_seconds_per_call.instruct)} s × ${D.split.counts.eval.toLocaleString()})`, { color: C.accent2 });
stat(s, 6.3, 3.75, 3.2, "0.04 s", "the same query by exact vector search over precomputed embeddings", { color: C.accent3 });
s.addNotes("0:15–1:00. Frame the cost problem with the three numbers on the right: the corpus, what an exhaustive VLM costs per query, what a vector search costs. The research question is the proposal's, unchanged.");

// 3 dataset
pres.addSection({ title: "Data and method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
s.addText("Dataset: MEVA, and a query set built for it", { placeholder: "title" });
card(s, 0.5, 1.12, 2.75, 1.85, "MEVA (Kitware / IARPA DIVA)", bullets(["Fixed cameras, scripted everyday activities", "37 activity types with actor boxes", "CC BY 4.0; not Kaggle, not synthetic"], 12));
card(s, 3.4, 1.12, 2.75, 1.85, "What was used", bullets([`${D.instances.toLocaleString()} instances with video (~70 GB)`, `${D.camera.distinct} cameras, ${D.site.distinct} sites, ${D.activity.distinct} activity types`, "2 fps, ≤ 16 frames, actor crops"], 12));
card(s, 0.5, 3.07, 2.75, 1.85, "Two protocols", bullets([`A, official split: ${D.split.counts.train.toLocaleString()} / ${D.split.counts.val} / ${D.split.counts.eval.toLocaleString()} (train / val / eval)`, `B, unseen cameras: ${D.split_b.counts.train.toLocaleString()} / ${D.split_b.counts.val} / ${D.split_b.counts.eval.toLocaleString()}`, "A–B gap = reliance on seen cameras"], 12));
card(s, 3.4, 3.07, 2.75, 1.85, "Curated query set", bullets(["22 activities × 3 phrasings = 66, frozen by hash", "192 training captions, disjoint", "Nothing chosen on the evaluation split"], 12));
const hStrip = img(s, "deck_meva_strip.png", 6.35, 1.12, 3.15, "MEVA result cards from the dashboard: tight actor crops of parking-lot scenes");
s.addText("MEVA clips as the system sees them: actor-region crops from fixed cameras", { x: 6.35, y: 1.12 + hStrip + 0.05, w: 3.15, h: 0.45, fontSize: 10, color: C.text2, margin: 0, isTextBox: true, objectName: "strip caption" });
card(s, 6.35, 3.07, 3.15, 1.85, "Balance, as proposed", bullets(["√-balanced instance sampling", "Every metric macro-averaged over 22 activities", "3 phrasings per activity; 10 rare types kept as distractors"], 12.5));
s.addNotes("1:00–1:45. Dataset facts and the curated query set. Stress: the query set did not exist (MEVA had only activity labels); it was written from MEVA's definitions, reviewed blind to results and frozen by hash. Protocol B holds cameras out of training.");

// 4 gap
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
s.addText("What others did, and what was missing", { placeholder: "title" });
card(s, 0.5, 1.12, 2.9, 2.25, "1 · No cost axis", "Surveillance retrieval systems — Tran et al. (CVPRW 2025), SurveiLens, ForeSea (Qualcomm 2026) — report accuracy only. None reports how many model calls a query costs, or what accuracy does as the budget shrinks.", { bodySize: 12.5 });
card(s, 3.55, 1.12, 2.9, 2.25, "2 · MEVA never searched by language", "The largest permissively licensed fixed-camera activity corpus is used only for activity detection (TRECVID ActEV). No text-query benchmark existed.", { bodySize: 12.5 });
card(s, 6.6, 1.12, 2.9, 2.25, "3 · Domain shift unmeasured", "CLIP-style encoders are trained on web photos; UCA (CVPR 2024) shows web-trained models fail on surveillance. The shift had not been measured for a dual encoder, with and without adaptation, on seen and unseen cameras.", { bodySize: 12.5 });
s.addText([
  { text: "Building blocks reused: ", options: { bold: true, color: H.accent1 } },
  { text: "CLIP, SigLIP, SigLIP 2 (encoders) · LoRA (adaptation) · knowledge distillation and cross-architecture KD, cascade ranking (staged scoring) · Escorcia et al., QVHighlights (corpus moment retrieval) · QASIR (cheap screen, then VLM) · Qwen3-VL (the VLMs) · YOLOv8 + ByteTrack (alerts). Full list: report §3, 20 references, each checked at its source.", options: {} },
], { x: 0.5, y: 3.55, w: 9.0, h: 0.95, fontSize: 12.5, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "related work line" });
pill(s, 2.5, 4.6, 5.0, "This project measures all three gaps on one corpus", C.accent2);
s.addNotes("1:45–2:30. Three gaps, one line each. The related-work line names the building blocks so the novelty is clear: the cascade is prior work and reused; what is new is the measured cost axis, the MEVA query benchmark, and the adaptation measurement.");

// 5 approach
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
s.addText("Approach: adapt, re-rank cheaply, spend VLM calls last", { placeholder: "title" });
img(s, "fig1_pipeline.png", 0.5, 1.08, 6.05, "Pipeline: offline indexing, training, alert ingestion and the query path with three cost levels");
card(s, 6.8, 1.08, 2.7, 1.2, "FAST · 0 VLM calls", "LoRA-adapted SigLIP 2; exact search over every clip. ≈ 0.04 s", { headColor: C.accent3, bodySize: 12.5 });
card(s, 6.8, 2.42, 2.7, 1.2, "STANDARD · 0 VLM calls", "+ temporal student: transformer over frame vectors and box motion. ≈ 0.2 s", { headColor: C.accent1, bodySize: 12.5 });
card(s, 6.8, 3.76, 2.7, 1.2, "VERIFIED · k VLM calls", "VLM answers yes/no on the top k only, fused with the earlier score. ≈ 1–4 s", { headColor: C.accent2, bodySize: 12.5 });
s.addNotes("2:30–3:30. Walk the four lanes: offline indexing (crops, SigLIP 2 + LoRA, exact index); training (LoRA on 192 captions; the student on human labels, with teacher labels only admitted past gate G1); alert ingestion for new video; the query path with three cost levels. Every choice was made on the validation split.");

// 6 LoRA
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Fine-tuning the encoder works, and transfers to unseen cameras", { placeholder: "title" });
s.addChart(pres.ChartType.bar, [
  { name: "frozen SigLIP 2", labels: ["Q1 mAP · A", "Q1 mAP · B", "Q2 Success@1 · A", "Q2 Success@1 · B"], values: [base("A", "siglip2-tight", "q1", "ap"), base("B", "siglip2-tight", "q1", "ap"), cas("2b", "A", "q2", "frozen", "s1"), cas("2b", "B", "q2", "frozen", "s1")] },
  { name: "LoRA-adapted", labels: ["Q1 mAP · A", "Q1 mAP · B", "Q2 Success@1 · A", "Q2 Success@1 · B"], values: [lora("A-middle", "q1", "ap"), lora("B-uniform", "q1", "ap"), lora("A-middle", "q2", "s1"), lora("B-uniform", "q2", "s1")] },
], { x: 0.5, y: 1.08, w: 5.7, h: 3.95, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60, chartColors: [H.dk2, H.accent1],
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: H.dk1, dataLabelFontSize: 10, dataLabelFormatCode: "0.000",
  valAxisMinVal: 0, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", showLegend: true, legendPos: "b",
  title: "Evaluation split, 0 VLM calls per query (macro over 22 activities)", ...chartFrame() });
stat(s, 6.5, 1.1, 3.0, "+0.134", "Q1 mAP gain on A (95 % CI +0.085 to +0.194); every metric improves on both protocols", { bigSize: 28, bigH: 0.5, labelH: 0.7 });
stat(s, 6.5, 2.45, 3.0, "≈ 80 %", "of the gain transfers to cameras never seen in training (protocol B)", { bigSize: 28, bigH: 0.5, labelH: 0.7, color: C.accent3 });
stat(s, 6.5, 3.8, 3.0, `${f2(pa("person_talks_to_person", "A").ap_base)} → ${f2(pa("person_talks_to_person", "A").ap_lora)}`, "AP for \"person talks to person\" — from a 1.2 M-parameter adapter (1.25 % of the encoder), 1.5 h on a laptop GPU", { bigSize: 28, bigH: 0.5, labelH: 0.7 });
s.addNotes(`3:30–4:15. Frozen vs LoRA-adapted. A: mAP ${f(base("A", "siglip2-tight", "q1", "ap"))} to ${f(lora("A-middle", "q1", "ap"))}; B: ${f(base("B", "siglip2-tight", "q1", "ap"))} to ${f(lora("B-uniform", "q1", "ap"))}. Forensic Success@1 0.27 to 0.39. All 22 activities improve on A; the gains repeat on unseen cameras, and the training captions are disjoint from the evaluation phrasings, so it is neither camera memorisation nor query leakage. Motion-defined activities (pick up, put down, stand up) rise but stay the weakest: that is what the student is for.`);

// 7 headline cost curve
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("The headline: a cheap re-ranker replaces most of the VLM's value", { placeholder: "title" });
const labels = ["0 calls", "top 5", "top 10", "top 20"];
const series = (blk, pre) => [cas(blk, "A", "q2", "student", "s1"), ...["5", "10", "20"].map((k) => cas(blk, "A", "q2", `${pre}@${k}`, "s1"))];
s.addChart(pres.ChartType.line, [
  { name: "LoRA + student → Qwen3-VL-2B on top k", labels, values: series("2b", "student>instruct") },
  { name: "LoRA + student → 8B on top k", labels, values: series("8b", "student>instruct") },
  { name: "LoRA + student → 32B on top k", labels, values: series("32b", "student>instruct") },
  { name: "LoRA → 2B on top k (no student)", labels, values: [cas("2b", "A", "q2", "lora", "s1"), ...["5", "10", "20"].map((k) => cas("2b", "A", "q2", `lora>instruct@${k}`, "s1"))] },
], { x: 0.5, y: 1.08, w: 5.9, h: 3.95, chartColors: [H.accent3, H.accent4, H.accent2, H.accent6], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 7,
  valAxisMinVal: 0.35, valAxisMaxVal: 0.65, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00", showLegend: true, legendPos: "b",
  title: "Protocol A · forensic Success@1 against VLM calls per query", ...chartFrame() });
stat(s, 6.65, 1.08, 2.85, `${f2(cas("2b", "A", "q2", "lora", "s1"))} → ${f2(cas("2b", "A", "q2", "student", "s1"))}`, `forensic Success@1 at 0 VLM calls, LoRA → LoRA + student (A); B: ${f2(cas("2b", "B", "q2", "lora", "s1"))} → ${f2(cas("2b", "B", "q2", "student", "s1"))}`, { bigSize: 28, bigH: 0.5, labelH: 0.6 });
stat(s, 6.65, 2.3, 2.85, sg(bsd("32b", "A: 32B@10 - student").diff), `more at 10 calls with the 32B (95 % CI ${sg(bsd("32b", "A: 32B@10 - student").ci95[0])} to ${sg(bsd("32b", "A: 32B@10 - student").ci95[1])}); 8B ${sg(bsd("8b", "A: 8B@10 - student").diff)}, 2B ≤ +0.03; nothing beyond 20`, { bigSize: 28, bigH: 0.5, labelH: 0.75, color: C.accent2 });
stat(s, 6.65, 3.7, 2.85, f2(cas("2b", "A", "q2", "instruct-all", "s1")), `the 2B VLM on every candidate (~${Math.round(cas("2b", "A", "q2", "instruct-all", "cost"))} calls): below LoRA alone (${f2(cas("2b", "A", "q2", "lora", "s1"))})`, { bigSize: 28, bigH: 0.5, labelH: 0.6, color: C.accent5 });
s.addNotes("4:15–5:30. The main result. The temporal student (frame vectors + box motion, trained on the human labels) takes forensic Success@1 from 0.39 to 0.53 at zero VLM calls, with the biggest gains on motion-defined activities (pick up, put down, stand up: AP 0.19 to 0.56). A VLM on the student's short list then adds up to +0.065 with the 32B at 10 calls, significant on both protocols; the 32B is not better than the 8B at 13× the time per call. The exhaustive 2B VLM is below LoRA alone. The curve is flat first and concave only after a good short list; the knee is at 5–10 calls.");

// 8 VLM gates
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Seven VLMs were gated; one passed, and it added little", { placeholder: "title" });
const hdr = { bold: true, fill: { color: "1E293B" }, color: H.dk1 };
const vr = (name, size, g0k, g1k, cfg, verdict, bold) => [
  { text: name, options: { bold: !!bold } }, size, g0k ? f2(g0(g0k, "seconds_per_pair_uncached")) : "—",
  `${sg(g1(g1k, cfg, "A"))} / ${sg(g1(g1k, cfg, "B"))}`, { text: verdict, options: { color: bold ? H.accent3 : H.dk1, bold: !!bold } }];
s.addTable([
  [{ text: "VLM", options: hdr }, { text: "Size", options: hdr }, { text: "s / call", options: hdr }, { text: "G1 gain A / B (val. mAP)", options: hdr }, { text: "Verdict (bar: +0.02 on one protocol)", options: hdr }],
  vr("Qwen3-VL-2B-Instruct", "2B", "qwen3vl_2b_instruct", "2b_top50", "instruct-tight", "fail"),
  vr("Qwen3-VL-Reranker-2B", "2B", "qwen3vl_2b_reranker", "2b_top50", "reranker-tight", "fail · the 2B teacher by the pre-set rule"),
  vr("Qwen3-VL-2B, 16 frames", "2B", null, "2b_16frames", "instruct-tight", "fail · more frames do not help"),
  vr("Gemma 4 E4B-it", "4B", "gemma4_e4b", "gemma_top50", "gemma-e4b-tight", "fail · weaker than Qwen 2B on A"),
  vr("Qwen2.5-VL-3B / 7B (FP8)", "3B / 7B", "qwen25vl_7b_fp8", "qwen25_7b_top50", "q25-7b-fp8-tight", "fail (3B: +0.001 / +0.004)"),
  vr("Qwen3-VL-8B-Instruct (FP8, server)", "8B", "qwen3vl_8b_fp8", "8b_top50", "instruct-tight", "fail by 0.001 · kept as a short-list stage"),
  vr("Qwen3-VL-32B-Instruct (FP8, split)", "32B", "qwen3vl_32b_fp8", "32b_top50", "instruct-tight", `pass · whole validation ${sg(g1("32b_wholeval", "instruct-tight", "A"))} / ${sg(g1("32b_wholeval", "instruct-tight", "B"))}`, true),
], { x: 0.5, y: 1.08, w: 9.0, colW: [2.45, 0.65, 0.7, 1.75, 3.45], fontSize: 11.5, color: H.dk1, fill: { color: H.lt2 }, border: { type: "solid", pt: 0.5, color: "1F2A37" }, rowH: 0.33, valign: "middle", margin: 0.04, objectName: "vlm gate table" });
s.addText([
  { text: "Why they fail: ", options: { bold: true, color: H.accent1 } },
  { text: `inside the retriever's own top 50, the VLMs rank right clips above wrong ones with AUC ${f2(N.g1_auc["2b"].A["auc_instruct-tight"])} (2B), ${f2(N.g1_auc.gemma.A["Gemma4-E4B"])} (Gemma), ${f2(N.g1_auc["8b"].A.Qwen8B)} (8B); the adapted retriever reaches ${f2(N.g1_auc["8b"].A.LoRA)} (A). The signal is real but weaker than the stage it should improve, so validation always chose the largest retriever weight. Only the 32B adds information, and in the cascade its gain equals the 8B's at ${Math.round(g0("qwen3vl_32b_fp8", "seconds_per_pair_uncached") / g0("qwen3vl_8b_fp8", "seconds_per_pair_uncached"))}× the time per call.`, options: {} },
], { x: 0.5, y: 4.22, w: 9.0, h: 0.9, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "gate reading" });
s.addNotes("5:30–6:15. The teacher question, all of it: seven models, two gates (G0 fit, G1 information). Six fail; the 32B passes at both depths but, in the cascade, equals the 8B. Also: the tight actor view beats the context view for every VLM, and no model, student included, can tell a left turn from a right turn.");

// 9 negatives
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("What did not work, and what it taught us", { placeholder: "title" });
card(s, 0.5, 1.08, 4.4, 1.9, "Distillation from a VLM teacher", bullets([`Student validation mAP (A): human labels ${f(N.students.A.val_map.GT)} · + teacher ${f(N.students.A.val_map.GTKD)} · teacher only ${f(N.students.A.val_map.KD)}`, "Same with the 32B teacher: 0.606 / 0.573 / 0.447", "A teacher weaker than the labels (AUC 0.62–0.69 on the training pairs) cannot teach more than the labels"], 12.5), { headColor: C.accent5 });
card(s, 5.1, 1.08, 4.4, 1.9, "Image + text LoRA (added experiment)", bullets([`Two seeds: ${f(N.e8.verdict.mean_diff["A-middle"])} on seen cameras, +${f(N.e8.verdict.mean_diff["B-uniform"])} on unseen`, `Fails the pre-set rule and a text-drift guard (ρ ${f2(N.drift.results.B0.sentences.rho)}–${f2(N.drift.results.A1.sentences.rho)} < 0.90)`, "Promising for unseen cameras, once a retention objective keeps general language intact"], 12.5), { headColor: C.accent2 });
card(s, 0.5, 3.12, 4.4, 1.9, "A VLM on every candidate", bullets([`2B on all ${D.split.counts.eval.toLocaleString()} clips: Q1 mAP ${f(cas("q1all", "A", "q1", "instruct-all", "ap"))} at ~10 min per query, against LoRA's ${f(lora("A-middle", "q1", "ap"))} in 40 ms`, "Fused with LoRA it only recovers LoRA", "16 frames, Gemma 4, Qwen2.5-VL: all fail the gate too"], 12.5), { headColor: C.accent5 });
card(s, 5.1, 3.12, 4.4, 1.9, "Turn direction", bullets(["No component separates left from right turns (AUC 0.49–0.51 in windows with both)", "Image-plane box motion does not encode heading relative to the camera", "A merged \"vehicle turns\" class is found well: LoRA AP 0.57 (A) / 0.62 (B)"], 12.5), { headColor: C.accent4 });
s.addNotes("6:15–7:00. Negative results, reported with the same care: the proposal's distillation contribution is rejected; text-side adaptation is a lead, not a result; exhaustive VLM scoring is the wrong place to spend calls; turn direction needs heading-aware features.");

// 10 demo
pres.addSection({ title: "Demo and close" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo and close" });
s.addText("Demo: the pipeline as a dashboard (recorded video)", { placeholder: "title" });
const hShot = img(s, "fig4a_search.png", 0.5, 1.08, 5.9, "Dashboard Search tab: an evaluation phrasing at the Standard level with labels shown and the cost line");
s.addText("Search tab on a MEVA evaluation phrasing: 6 of the top 10 carry the label; cost line shows 0 VLM calls and what a VLM on every clip would cost", { x: 0.5, y: 1.08 + hShot + 0.05, w: 5.9, h: 0.4, fontSize: 10, color: C.text2, margin: 0, isTextBox: true, objectName: "shot caption" });
card(s, 6.6, 1.08, 2.9, 3.95, "In the video (1:30)", bullets(["Upload a CCTV clip: YOLO + ByteTrack raise 22 person / vehicle alerts, 5 s each, searchable 22 s later", "Search it in words at Standard: 0.6 s, 0 VLM calls", "Switch to Verified: 10 VLM calls instead of 3,726", "Labels on, frozen model side by side: fine-tuning visible", "The 3D clip map: similar clips cluster, results light up"], 12.5), { headColor: C.accent1 });
s.addNotes("7:00–8:30. Play the recording (about 90 s). The dashboard reproduces the evaluated system: Success@1 and @10 identical on all 1,206 evaluation queries. Alerts from new video go through the same crops, encoder and student features, so they share the MEVA score scale. Footage unlike MEVA is a demonstration, not a measured result.");

// 11 limitations
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo and close" });
s.addText("Limitations", { placeholder: "title" });
const lims = [
  ["Scripted data", "MEVA activities are acted at a training facility; absolute numbers may be optimistic, comparisons between arms remain valid."],
  ["One training seed", "Seed-to-seed spread measured at ±0.017 validation mAP; differences below ~0.02 on one seed are not conclusive. CIs are over activities."],
  ["Small validation sets", `${D.split.counts.val} / ${D.split_b.counts.val} instances; the chosen epoch and sampling variant are noisy picks (they tie on evaluation).`],
  ["VLM set-up", "8 frames, the tight view, one default prompt; 16 frames did not help, other prompts were not tried. The 32B ran with a CPU/GPU split."],
  ["Turn direction", "Left vs right is not separable by any component; heading relative to the camera is not encoded."],
  ["New footage is unlabelled", "Alerts from uploaded video are not measured; the confidence band is relative to MEVA, not a probability."],
];
lims.forEach(([h, t], i) => {
  const x = i < 3 ? 0.5 : 5.2, y = 1.1 + (i % 3) * 1.32;
  pill(s, x, y, 0.42, String(i + 1), C.accent1, { h: 0.42, size: 14 });
  s.addText(h, { x: x + 0.55, y, w: 3.75, h: 0.42, fontSize: 14, bold: true, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: nm("lim head") });
  s.addText(t, { x: x + 0.55, y: y + 0.44, w: 3.75, h: 0.82, fontSize: 12.5, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: nm("lim text") });
});
s.addNotes("8:30–9:00. Six limitations, one breath each. The single-seed point matters most for reading small differences.");

// 12 conclusion
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo and close" });
s.addText("Conclusion: what the proposal claimed, and what held", { placeholder: "title" });
const rows = [
  ["CONFIRMED", C.accent3, "Domain adaptation", `LoRA on 192 captions: Q1 mAP ${f(base("A", "siglip2-tight", "q1", "ap"))} → ${f(lora("A-middle", "q1", "ap"))}; about 80 % of the gain transfers to unseen cameras.`],
  ["NOT AS CLAIMED", C.accent5, "A distilled VLM teacher", `Six of seven VLMs fail the information gate; distillation never beats the human labels. What held instead: a temporal student on the labels, forensic Success@1 ${f2(cas("2b", "A", "q2", "lora", "s1"))} → ${f2(cas("2b", "A", "q2", "student", "s1"))} at 0 VLM calls.`],
  ["DELIVERED, DIFFERENT SHAPE", C.accent2, "The accuracy-vs-cost curve", `Flat for the 2B VLM (exhaustive 2B < LoRA alone); concave after the student; 8B / 32B add ${sg(bsd("32b", "A: 32B@10 - student").diff)} at 10 calls, nothing beyond 20.`],
];
rows.forEach(([tag, color, head, text], i) => {
  const y = 1.1 + i * 1.08;
  pill(s, 0.5, y + 0.05, 2.45, tag, color, { h: 0.42, size: 11 });
  s.addText(head, { x: 3.15, y, w: 6.35, h: 0.36, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: nm("claim head") });
  s.addText(text, { x: 3.15, y: y + 0.38, w: 6.35, h: 0.62, fontSize: 12.5, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: nm("claim text") });
});
s.addText("Adapt the encoder, re-rank cheaply, spend the VLM calls at the end.", { x: 0.5, y: 4.35, w: 9.0, h: 0.45, fontSize: 20, bold: true, color: C.accent1, margin: 0, isTextBox: true, align: "center", valign: "middle", objectName: "takeaway" });
s.addText("Thank you — questions", { x: 0.5, y: 4.82, w: 9.0, h: 0.3, fontSize: 12, color: C.text2, margin: 0, isTextBox: true, align: "center", objectName: "thanks" });
s.addNotes("9:00–10:00. Scorecard against the proposal, then the one-line takeaway. Future work if asked: heading-aware motion features, a retention objective for text-side adaptation, a second seed, a prompt study for the VLM gate, labelled non-MEVA footage for calibrated confidence.");

// ---------------------------------------------------------------- write + theme
write(OUT).catch((e) => { console.error(e); process.exit(1); });
