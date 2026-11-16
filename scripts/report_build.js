// Builds the final report, report/Ahmed_Wajahat_DLReport.docx, from report/numbers.json and report/figures/
// (docs/planning/spec-report.md). Every result number is read from numbers.json; the few confidence
// intervals quoted from the Result files are marked CI_FROM_RESULTS below.
//
//   cd report/build && node ../../scripts/report_build.js
//
"use strict";
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "..");
// docx-js lives under report/build (npm), not next to this script
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun, LevelFormat, PageNumber, Packer, Paragraph,
  ShadingType, Table, TableCell, TableRow, TextRun, WidthType, VerticalAlign, TableLayoutType,
} = require(require.resolve("docx", { paths: [path.join(ROOT, "report", "build")] }));

const N = JSON.parse(fs.readFileSync(path.join(ROOT, "report", "numbers.json"), "utf8"));
const FIG = (name) => path.join(ROOT, "report", "figures", name);
const OUT = process.env.REPORT_OUT || path.join(ROOT, "report", "Ahmed_Wajahat_DLReport.docx");   // REPORT_OUT: build elsewhere (e.g. while the file is open in Word)

// ------------------------------------------------------------------ numbers
const f = (x, d = 3) => Number(x).toFixed(d);
const f2 = (x) => f(x, 2);
const base = (P, sys, q, m) => N.baseline[P].systems[sys][q][m];
const lora = (run, q, m) => N.lora.runs[run][q][m];
const cas = (blk, P, q, sys, m) => N.cascade[blk].summary[`${P}/${q}`][sys][m];
const g1 = (key, cfg, P) => N.g1[key].configs[cfg][P].gain;
const g0 = (key, m) => N.g0[key][m];
const D = N.dataset;
const SEL = { A: "A-middle", B: "B-uniform" };
const vlmS = { "2b": N.cascade["2b"].vlm_seconds_per_call.instruct, "8b": g0("qwen3vl_8b_fp8", "seconds_per_pair_uncached"),
               "32b": g0("qwen3vl_32b_fp8", "seconds_per_pair_uncached") };
const studentS = N.cascade["2b"].students.A.seconds_per_query_whole_eval_corpus;

// motion-defined activities (baseline lift <= 2 on A, Result 02 rule): pick up, put down, stand up
const MOTION = ["person_picks_up_object", "person_puts_down_object", "person_stands_up"];
function apGain(P, group) {
  const rows = N.cascade.per_activity.filter((r) => r.protocol === P);
  const ap = {};
  for (const r of rows) (ap[r.activity] ||= {})[r.system] = Number(r.q1_ap);
  const acts = Object.keys(ap).filter((a) => group(a));
  const mean = (k) => acts.reduce((s, a) => s + ap[a][k], 0) / acts.length;
  return { n: acts.length, lora: mean("lora"), student: mean("student"), frozen: mean("frozen") };
}
const motionA = apGain("A", (a) => MOTION.includes(a));
const restA = apGain("A", (a) => !MOTION.includes(a));
const loraPA = (act, P) => N.lora.per_activity.find((r) => r.protocol === P && r.activity === act);
const pa = (act) => ({ A: loraPA(act, "A"), B: loraPA(act, "B") });
const turns = (sys) => N.cascade.turns_per_activity.find((r) => r.protocol === "A" && r.system === sys);
const turnsB = (sys) => N.cascade.turns_per_activity.find((r) => r.protocol === "B" && r.system === sys);

// CI_FROM_RESULTS: paired bootstrap over activities, 10,000 resamples, quoted from docs/results/03-lora.md
// and 04-cascade.md (the raw files hold the per-query tables, not these intervals)
const CI = {
  loraA_map: "+0.134 [+0.085, +0.194]", loraB_map: "+0.107 [+0.062, +0.158]",
  loraA_s1: "+0.115 [+0.065, +0.168]", loraB_s1: "+0.094 [+0.035, +0.151]",
  studA_s1: "+0.143 [+0.058, +0.228]", studB_s1: "+0.150 [+0.082, +0.220]",
  studA_map: "+0.163 [+0.097, +0.239]", studB_map: "+0.162 [+0.094, +0.235]",
  vlm2bA_s1: "+0.028 [+0.003, +0.054]", vlm2bB_s1: "+0.031 [−0.001, +0.067]",
  gtkdA: "−0.021 [−0.062, +0.014]", kdA: "−0.185 [−0.263, −0.118]",
};
const bs = (blk, key) => { const b = N.bootstrap[blk][key]; return `${b.diff > 0 ? "+" : ""}${f(b.diff, 3)} [${b.ci95[0] > 0 ? "+" : ""}${f(b.ci95[0], 3)}, +${f(b.ci95[1], 3)}]`; };
// students distilled from the 32B teacher (validation mAP; docs/results/05-vlm-followups.md, D-047)
const S32 = { A: { GT: 0.606, GTKD: 0.573, KD: 0.447 }, B: { GT: 0.571, GTKD: 0.525, KD: 0.352 } };

// ------------------------------------------------------------------ document helpers
const FONT = "Calibri";
const BODY = 21; // 10.5 pt
const PAGE_W = 11906, MARGIN = 1250, TEXT_W = PAGE_W - 2 * MARGIN; // A4, 2.2 cm

function runs(parts, size = BODY) {
  if (typeof parts === "string") parts = [parts];
  return parts.map((p) => typeof p === "string" ? new TextRun({ text: p, size, font: FONT })
    : new TextRun({ text: p.t, bold: !!p.b, italics: !!p.i, size: p.size || size, font: p.mono ? "Consolas" : FONT, superScript: !!p.sup }));
}
const P = (parts, o = {}) => new Paragraph({ children: runs(parts, o.size), alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after ?? 80, line: o.line ?? 252 }, keepNext: o.keepNext, keepLines: o.keepLines });
const H1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1 });
const H2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2 });
const H3 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_3 });
const UL = (parts) => new Paragraph({ children: runs(parts), numbering: { reference: "bullets", level: 0 }, spacing: { after: 40, line: 252 }, alignment: AlignmentType.JUSTIFIED });
const CAP = (t) => new Paragraph({ children: runs([{ t, i: true }], 18), alignment: AlignmentType.CENTER, spacing: { before: 40, after: 160 } });
const SP = (after = 60) => new Paragraph({ spacing: { after }, children: [] });

function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
function IMG(name, widthIn) {
  const buf = fs.readFileSync(FIG(name));
  const s = pngSize(buf);
  const w = Math.round(widthIn * 96), h = Math.round(w * s.h / s.w);
  return new Paragraph({ children: [new ImageRun({ type: "png", data: buf, transformation: { width: w, height: h } })],
    alignment: AlignmentType.CENTER, spacing: { before: 80, after: 20 }, keepNext: true });
}

const thin = { style: BorderStyle.SINGLE, size: 4, color: "9AA5B1" };
const BORDERS = { top: thin, bottom: thin, left: thin, right: thin, insideHorizontal: thin, insideVertical: thin };
function cell(text, width, o = {}) {
  const parts = Array.isArray(text) ? text
    : (text !== null && typeof text === "object") ? [{ ...text, b: text.b || !!o.b }]
    : [{ t: String(text), b: !!o.b }];
  return new TableCell({
    width: { size: width, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
    margins: { top: 30, bottom: 30, left: 70, right: 70 },
    children: [new Paragraph({ children: runs(parts, o.size || 17), alignment: o.align || AlignmentType.LEFT, spacing: { after: 0, line: 240 } })],
  });
}
function TABLE(header, rows, fracs, o = {}) {
  const widths = fracs.map((x) => Math.round(TEXT_W * x));
  const size = o.size || 17;
  const mk = (r, isHead, i) => new TableRow({ tableHeader: isHead, cantSplit: true,
    children: r.map((c, j) => cell(c, widths[j], { b: isHead || (o.boldFirst && j === 0), fill: isHead ? "DCE6F1" : (o.shade && o.shade(r, i) ? "F3F6F9" : undefined),
      align: !isHead && o.numeric && j >= (o.textCols ?? 1) ? AlignmentType.CENTER : AlignmentType.LEFT, size })) });
  return new Table({ columnWidths: widths, width: { size: TEXT_W, type: WidthType.DXA }, borders: BORDERS, layout: TableLayoutType.FIXED,
    rows: [mk(header, true, -1), ...rows.map((r, i) => mk(r, false, i))] });
}

// ------------------------------------------------------------------ content
const doc = [];
const title = "Cost-Bounded Natural-Language Forensic Search over Surveillance Video via Domain-Adapted Retrieval and Learned Re-Ranking";

doc.push(new Paragraph({ children: runs([{ t: "Deep Learning Project — Final Report", b: true }], 24), alignment: AlignmentType.CENTER, spacing: { after: 60 } }));
doc.push(new Paragraph({ children: runs([{ t: title, b: true }], 30), alignment: AlignmentType.CENTER, spacing: { after: 120 } }));
doc.push(TABLE(["Project Title", title], [
  ["Student Name & ID", "Wajahat Ahmed — CS-141/2026"], ["Course & Section", "CS-5138 Deep Learning (Graduate), NED University of Engineering & Technology"],
  ["Date", "November 2026"], ["Proposal", "Submitted 19 September 2026 (same title family; this report answers it section by section)"],
  ["Code and data", "github.com/wajahatah/semantic-forensic-search (private); MEVA, CC BY 4.0"],
], [0.22, 0.78], { boldFirst: true, size: 18 }));
doc.push(SP(120));

doc.push(H1("Abstract"));
doc.push(P(`Surveillance archives have no index of what happens in the footage, so post-incident review is manual. This project builds natural-language search over surveillance activity clips on the MEVA dataset under an explicit cost constraint: how much of the retrieval quality of a vision-language model (VLM) applied to every clip can be kept while making one to two orders of magnitude fewer VLM calls? We curated a frozen query set (22 activity types, 66 phrasings) over ${D.instances.toLocaleString()} annotated instances, domain-adapted the SigLIP 2 image encoder with LoRA, trained a temporal re-ranker over frame embeddings and box motion, and gated seven VLMs (2B–32B parameters) on whether they add information to the adapted retriever. On the official evaluation split, category-search mAP rises from ${f(base("A", "siglip2-tight", "q1", "ap"))} (frozen encoder) to ${f(lora("A-middle", "q1", "ap"))} with LoRA and ${f(cas("2b", "A", "q1", "student", "ap"))} with the re-ranker, at zero VLM calls; forensic Success@1 rises from ${f2(cas("2b", "A", "q2", "frozen", "s1"))} to ${f2(cas("2b", "A", "q2", "student", "s1"))}, with the same gains on cameras never seen in training. Only the 32B VLM passed the information gate; an 8B or 32B VLM on the top ten adds a further ${bs("32b", "A: 32B@10 - student").split(" ")[0]} Success@1 (significant), while the 2B VLM on every candidate scores below the cheap stages. Distillation from any teacher did not beat training on the human labels. The pipeline runs as a dashboard that searches MEVA and alerts detected in new videos, with the cost of each query shown.`));

// ---- 1
doc.push(H1("1. Problem Definition"));
doc.push(P(`A surveillance installation produces thousands of automated alerts a day. When an incident is reported after the fact, investigators scrub footage by camera and time because nothing indexes what the footage shows. The proposal framed this as cross-modal text-to-video retrieval: the input is a free-form query and a corpus of clips, the output a ranked list of clips. The distinguishing constraint is cost. A VLM that reads a clip and answers "does this show the query?" gives strong relevance judgements, but applying it to every clip of an archive is slow and expensive. The research question was therefore: how much of the accuracy of an exhaustive VLM system can be retained while reducing VLM invocations by one to two orders of magnitude?`));
doc.push(P([`Two things changed between the proposal and the work. First, the unit of retrieval is an `, { t: "annotated activity instance", b: true }, ` (a clip with the actors' boxes and a start/end time), which is what an alert pipeline produces; free temporal localisation inside long video was out of scope. Second, the measured answer turned out to need two reference points, not one: the 2B VLM applied to every candidate was itself worse than the cheap stages, so the cost curve is reported against both the exhaustive VLM and the best zero-call system. The short answer to the research question is that a domain-adapted encoder plus a cheap learned re-ranker keeps all of the quality of the exhaustive 2B VLM at 0 calls, and that larger VLMs are worth about ten calls per query on top of it.`]));

// ---- 2
doc.push(H1("2. Dataset Description & Constraint Compliance"));
doc.push(P([`MEVA (Multiview Extended Video with Activities) was collected by Kitware at the Muscatatuck Urban Training Center for the IARPA DIVA programme [18]: fixed indoor and outdoor cameras, actors performing scripted everyday activities, released under CC BY 4.0 through the AWS Open Data registry with human annotations (actor bounding boxes, activity labels, temporal extents) in a public repository. It is not hosted on Kaggle and is not synthetic. Table 1 lists what was used.`]));
doc.push(TABLE(["Property", "Value"], [
  ["Dataset", "MEVA, 37 activity types; 4,259 five-minute clips (~328 h) released on AWS; 144 h annotated at the 2021 release [18]"],
  ["Used in this work", `${D.instances.toLocaleString()} annotated activity instances whose video was downloaded (~70 GB), ${D.activity.distinct} activity types, ${D.camera.distinct} cameras at ${D.site.distinct} sites`],
  ["Protocol A (official split)", `train ${D.split.counts.train.toLocaleString()} · validation ${D.split.counts.val} · evaluation ${D.split.counts.eval.toLocaleString()} instances; the evaluation cameras also appear in training`],
  ["Protocol B (unseen cameras)", `train ${D.split_b.counts.train.toLocaleString()} · validation ${D.split_b.counts.val} · evaluation ${D.split_b.counts.eval.toLocaleString()} instances; no evaluation camera appears in training`],
  ["Query activities", "22 types with ≥ 20 evaluation instances (10 rare types excluded from the queries, kept in the corpus as distractors)"],
  ["Query set (curated)", "3 natural phrasings per activity = 66 phrasings, written from MEVA's definitions and reviewed blind to any result; frozen by hash before the first evaluation"],
  ["Training captions (curated)", "192 captions over the 22 types for contrastive fine-tuning; disjoint from the evaluation phrasings (no exact match, same-activity text similarity < 0.80)"],
  ["Clip representation", "2 frames/s, at most 16 frames; actor-region crops from the annotated boxes"],
], [0.26, 0.74], { boldFirst: true }));
doc.push(CAP("Table 1. Dataset as used. Counts from the local manifest (report/numbers.json)."));
doc.push(P([{ t: "Size. ", b: true }, `${D.instances.toLocaleString()} visually grounded instances, each paired with several text views, is well above the "thousands of labelled samples" bar, and a 3,726-instance evaluation corpus makes rank-1 retrieval a meaningful test. `, { t: "Balance. ", b: true }, `The activity distribution is long-tailed (447 instances of "enters scene through structure" against 21 of "vehicle reverses" in the A evaluation split). Three measures were applied as proposed: square-root-balanced instance sampling during fine-tuning (head classes capped), macro-averaging of every metric over the 22 activities so head classes cannot hide tail failures, and a stratified query set with exactly three phrasings per activity. `, { t: "Selection rule. ", b: true }, `Every model, epoch, weight and threshold was chosen on the validation split; the evaluation split was read once per result and never used to choose anything.`]));

// ---- 3
doc.push(H1("3. Literature Review — Existing Work and What Was Missing"));
doc.push(P([{ t: "Contrastive image–text encoders. ", b: true }, `Radford et al. [1] trained an image encoder and a text encoder jointly on 400 million web image–caption pairs so that matching pairs score above all other pairs in a batch (CLIP). They showed that the model classifies images zero-shot from text prompts, matching a supervised ResNet-50 on ImageNet without ImageNet labels, and, because the two encoders are independent, that image embeddings can be computed once and searched. Zhai et al. [2] replaced the batch-wide softmax with a per-pair sigmoid loss (SigLIP) that needs no global normalisation; they report that it beats the softmax loss at batch sizes below 16k and that a locked-image variant reaches 84.5 % zero-shot ImageNet accuracy in two days on four TPU-v4 chips. Tschannen et al. [3] extended that recipe with captioning-based pretraining, self-distillation, masked prediction and online data curation (SigLIP 2) and report gains over SigLIP at every model size on zero-shot classification, image–text retrieval and dense prediction, releasing checkpoints from ViT-B (86 M parameters) to g (1 B). This project uses the ViT-B/16 SigLIP 2 checkpoint as its retriever. The limitation the three papers share for our purpose is their training data: web photographs, not fixed elevated cameras with small, distant subjects; §6.1 measures what that costs.`]));
doc.push(P([{ t: "Retrieving moments across a video corpus, and what it costs. ", b: true }, `Escorcia et al. [10] posed moment retrieval over a whole collection rather than inside one pre-selected video, and showed that aligning the query with short clips (their CAL model) retrieves better than moment-level alignment while letting the clip embeddings be indexed in advance. Lei et al. [11] released QVHighlights, about 10,000 YouTube videos with free-form queries annotated with the matching moments and per-clip saliency, together with a detection-style baseline (Moment-DETR). Tran et al. [12] built a retrieval system for hour-long video from an ensemble of a coarse encoder (CLIP) and a fine one (BEiT-3), keyframe de-duplication with TransNetV2, a temporal search with separate start and end queries, and a re-ranking step that uses neighbouring frames; they report higher precision and lower storage on known-item search and question answering, but give no account of model calls per query. Nishimura et al. [13] tile sampled frames into one "super image" so that a VLM can judge a whole video in a single pass, screen candidates with a cheap model and apply the expensive one to the survivors; they show that the accuracy–cost trade-off moves with grid size, model size and resolution, which is the closest precedent for the cascade used here. None of these works evaluates on surveillance footage.`]));
doc.push(P([{ t: "Surveillance video and language. ", b: true }, `Yuan et al. [14] annotated the UCF-Crime surveillance set with 23,542 sentences and event times (UCA, 110.7 hours) and benchmarked temporal grounding, captioning and dense captioning on it; they found that models which are strong on web video perform poorly on surveillance, with grounding recalls a fraction of their usual values, and attribute this to fixed viewpoints, low resolution and small subjects. Wu et al. [15] adapted CLIP to weakly supervised anomaly detection (VadCLIP) with a dual branch and learnable prompts and report state-of-the-art frame-level results on UCF-Crime and XD-Violence; the task is a closed set of anomaly classes, not free text. Owais et al. [16] describe SurveiLens, which answers a free-text description of an event with the matching CCTV segment by sampling frames, captioning them, scoring the captions against the query and localising in time, with the aim of cutting the hours staff spend reviewing recordings; captioning every sampled frame is the dominant expense and is not modelled. Park, Li et al. [17] at Qualcomm propose ForeSea, a three-stage forensic search: a tracker discards footage with nothing of interest, a multimodal embedding indexes the remaining clips, and a video LLM answers the query over the top retrieved clips; on their ForeSeaQA benchmark of image-and-text queries they report 3.5 points more accuracy and 11.0 more temporal IoU than earlier video-RAG systems. ForeSea is the closest system to ours in shape; it reports accuracy and localisation, not what a query costs in model calls.`]));
doc.push(P([{ t: "Adapting and compressing the expensive parts. ", b: true }, `Hu et al. [6] freeze a pretrained model and train low-rank updates to its weight matrices (LoRA); on GPT-3 175B they report quality on par with full fine-tuning while training 10,000 times fewer parameters and using a third of the GPU memory. Hinton et al. [7] formalised knowledge distillation: a small model trained on the softened outputs of a large model or ensemble recovers much of its accuracy, shown on MNIST and on a production speech recogniser. Hofstätter et al. [8] carried this to ranking: a BERT cross-encoder teacher scores query–passage pairs, and cheaper students (dual encoders, ColBERT) are trained with a margin-MSE loss on the teacher's score differences on MS MARCO, keeping most of the cross-encoder's effectiveness at a fraction of its latency. Wang, Lin and Metzler [9] showed with a cascade ranking model that a sequence of increasingly expensive rankers, each pruning the candidates for the next, trades effectiveness against cost explicitly, and learned such cascades on TREC web collections. The proposal's second contribution applied [7], [8] to a vision–language teacher over surveillance clips; §6.5 reports that the teacher's ranking signal was too weak for the transfer to help.`]));
doc.push(P([{ t: "Backbones, dataset and tools. ", b: true }, `Li et al. [4] bridge a frozen image encoder and a frozen LLM with a small query transformer (BLIP-2) and report, among other results, zero-shot VQA above Flamingo-80B with 54 times fewer trainable parameters; it is the cross-encoder the proposal compared against. Bai et al. [5] describe Qwen3-VL, a family of open-weight vision–language models (dense 2B, 4B, 8B and 32B, plus two mixture-of-experts sizes) that take interleaved images, video and text in a 256K-token context; its 2B, 8B and 32B members are the VLMs gated in §6.4. Corona et al. [18] released MEVA, over 9,300 hours of continuous multi-camera video from a training facility, 144 hours of it annotated for 37 activity types with actor boxes at release, built for the activity-detection challenges of TRECVID ActEV. Zhang et al. [19] showed that associating low-confidence detection boxes as well as high-confidence ones recovers occluded objects (ByteTrack), reporting 80.3 MOTA on MOT17 at 30 frames per second; with the Ultralytics YOLO detector [20] it produces the alerts of §5.7.`]));
doc.push(P([{ t: "What was missing. ", b: true }, `Three gaps motivated the proposal and were confirmed again before writing this report. (1) The surveillance retrieval systems [12], [16], [17] report accuracy only; none reports how many expensive model calls a query costs or what happens to accuracy as that budget shrinks. (2) MEVA, the largest annotated fixed-camera activity corpus with a permissive licence, had never been used for natural-language retrieval; its only published use is activity detection. (3) The domain shift from web photographs to fixed surveillance cameras is acknowledged in [1], [14] but had not been measured for a dual encoder on a controlled split, with and without adaptation, on seen and unseen cameras.`]));

// ---- 4
doc.push(H1("4. What Was Proposed and What Was Delivered"));
doc.push(P(`The proposal claimed three contributions: (i) a VLM re-ranker distilled into a deployable student, (ii) domain adaptation of the retrieval encoder, and (iii) an explicit accuracy-versus-cost frontier. Table 2 answers each proposal item with what was built and measured. Two of the three contributions hold as stated; the first does not, and what replaced it is the project's main result.`));
doc.push(TABLE(["Proposal item", "Outcome", "Evidence"], [
  ["LoRA domain adaptation improves first-stage recall (hypothesis 1)", { t: "Confirmed.", b: true }, `Q1 mAP ${f(base("A", "siglip2-tight", "q1", "ap"))} → ${f(lora("A-middle", "q1", "ap"))} (A), ${f(base("B", "siglip2-tight", "q1", "ap"))} → ${f(lora("B-uniform", "q1", "ap"))} (B, unseen cameras); every metric improves (§6.2)`],
  ["A VLM teacher distilled into a student keeps most of its quality (hypothesis 2)", { t: "Not as claimed.", b: true }, "No 2B–8B VLM passed the information gate; distillation never beat training the same student on the human labels, with the 2B or the 32B teacher. A temporal student trained on the labels is the largest single gain (§6.3, §6.5)"],
  ["Accuracy-versus-cost frontier (hypothesis 3: strongly concave)", { t: "Delivered, different shape.", b: true }, "Flat to negative for the 2B VLM (exhaustive 2B < LoRA alone); concave only after a good short list; 8B/32B add +0.06 Success@1 at 10 calls; 32B ≈ 8B at 13× the time (§6.3, Fig. 2)"],
  ["Cross-encoder/VLM invoked on few candidates only", { t: "Delivered.", b: true }, "VLM on the top 5–20 of the student's list, measured for 2B, 8B and 32B"],
  ["Cost as invocations, tokens, latency and money", { t: "Partly.", b: true }, "VLM calls and seconds per query are reported; tokens and money are not, because every model runs locally"],
  ["ChromaDB vector index + SQLite metadata", { t: "Changed.", b: true }, "Exact search over the clip vectors (3,726 clips in 0.04 s) + SQLite; an approximate index would add error at no gain at this size. ChromaDB is installed but unused"],
  ["torch.compile, Weights & Biases, Optuna, ONNX/quantised student", { t: "Not used.", b: true }, "Runs are short (≤ 2,700 steps, ≤ 1.5 h for all four LoRA runs), logged to CSV; grids were fixed in written specifications instead of automated search; nothing was deployed to an edge device"],
  ["Alert ingestion and a query interface", { t: "Delivered.", b: true }, "YOLOv8 + ByteTrack alerts from any video, 5-second clips through the same embedding path, dashboard with Detect and Search tabs (§7)"],
  ["Release of the curated query set", { t: "Delivered.", b: true }, "queries/phrasings.csv and queries/train_captions.csv in the repository, with hashes"],
  ["(Added) image + text LoRA", { t: "Measured, not adopted.", b: true }, `+${f(N.e8.verdict.mean_diff["B-uniform"])} validation mAP on unseen cameras but ${f(N.e8.verdict.mean_diff["A-middle"])} on seen ones; fails a text-drift guard (§6.6)`],
], [0.3, 0.17, 0.53], { size: 16 }));
doc.push(CAP("Table 2. The proposal, item by item."));

// ---- 5
doc.push(H1("5. Technique as Built"));
doc.push(IMG("fig1_pipeline.png", 6.5));
doc.push(CAP("Figure 1. The pipeline as built. Shaded boxes are trained in this project; the query path has three cost levels."));
doc.push(H2("5.1 Clip representation and frozen retriever"));
doc.push(P(`Each annotated instance is sampled at 2 frames per second (at most 16 frames). From the annotated actor boxes, each frame gets a square tight crop at native resolution around the actors and a context crop (≤ 448 px) with surroundings. The crops are embedded by SigLIP 2 base (ViT-B/16, 224 px) [3]; a clip's score for a text query is the mean of its three highest frame similarities, which tolerates frames in which the action is not yet visible. On the validation split, cropping doubled top-10 precision over full frames (0.22 vs 0.10) and SigLIP 2 beat SigLIP once cropped, so the tight view and SigLIP 2 were fixed before any evaluation number was read. Clip vectors are stored once; a query costs one text-encoder call and an exact dot-product search over every clip.`));
doc.push(H2("5.2 Domain adaptation with LoRA"));
doc.push(P(`LoRA adapters of rank 16 are added to the attention projections of the vision tower (48 modules, 1.18 M trainable parameters, 1.25 % of the encoder) [6]; the text tower stays frozen, so query encoding is unchanged and general language is untouched. Training uses the sigmoid image–text loss of SigLIP over the 192-caption bank, made label-aware: every caption of a crop's activity is a positive. Instances are sampled with square-root class balancing; augmentation covers zoom, stretch and resolution degradation but not flips (left/right turns are activities). Learning rate warm-up with reduce-on-plateau, early stopping after five epochs without a validation gain, a 100-epoch cap never reached. Four runs were trained (protocol A/B × uniform or middle-weighted frame sampling); per protocol the run with the best validation Q1 mAP was selected before the evaluation split was read.`));
doc.push(H2("5.3 Gates: when is a VLM allowed into the cascade?"));
doc.push(P(`A VLM scores a clip by the log-odds of "yes" over "no" to "Does this video show the following: {query}?", given 8 evenly spaced frames of the tight view at 448 px and the real frame times. Two gates, fixed in advance, decide whether a model is used. G0 (fit): peak memory within the budget of the machine, deterministic output, and a cached-prefix shortcut (one clip, many queries) that agrees with full scoring within the bf16 noise floor. G1 (information): re-rank the adapted retriever's top 50 of every validation query by z(VLM) + w·z(LoRA) with w chosen on validation; the model passes if it adds at least +0.02 mAP on one protocol. G1 was later repeated over the whole validation set rather than the top 50, to check that the verdict does not depend on the list depth.`));
doc.push(H2("5.4 Temporal student re-ranker"));
doc.push(P(`The student is a small transformer that reads, per frame, the LoRA tight embedding, the frozen context embedding and 12 box-motion features (position, size, displacement of the actor boxes), together with the query text embedding, and outputs one relevance score for the clip. It is trained on pairs from the training split: each activity's positives plus 100 hard negatives (highest-scoring wrong clips under LoRA) and 100 random negatives, about 12,200 pairs per protocol. Three targets were compared: GT, the human label; KD, the teacher's soft score; GT+KD, their average (the proposal's distillation route). Its score is fused with the retriever as z(student) + β·z(LoRA); β and the target variant were chosen on validation (β = 1, GT). At query time the student scores the whole candidate list in about ${f(studentS, 2)} s on the laptop GPU.`));
doc.push(H2("5.5 VLM on the short list"));
doc.push(P(`Where a VLM is used, it re-orders the top k of the previous stage by z(VLM) + γ·z(previous), with γ chosen on validation (γ = 2, the largest value in the grid, every time: the earlier stage is trusted more than the VLM). Cost is counted in VLM calls per query; latency is k × the measured seconds per call plus the student.`));
doc.push(H2("5.6 Evaluation protocol"));
doc.push(P(`Two query types. Q1 (category search): each of the 66 phrasings searches the whole evaluation corpus; mean average precision (mAP) and precision at 10. Q2 (forensic search): "find it on camera C in hour H", one query per annotated event and phrasing (1,140 queries per protocol), searched inside that camera-hour window (median 56 candidates, 3 relevant); Success@K (at least one correct clip in the top K), MRR and Recall@K. All metrics are macro-averaged over the 22 activities; 95 % confidence intervals come from a paired bootstrap over activities (10,000 resamples). Protocol B repeats everything with cameras held out of training, so the gap between A and B measures how much a model leans on having seen a camera.`));
doc.push(H2("5.7 Alert ingestion and dashboard"));
doc.push(P(`The same path indexes new video: YOLOv8 [20] with ByteTrack [19] detects people, vehicles and backpacks; an object raises one alert once it has been seen for 0.5 s with confidence ≥ 0.5, and the alert is held until the object has been gone for 2 s (a new track appearing where one was just lost is stitched to it). Each alert keeps 11 frames at 2 fps, exactly 5 seconds, which go through the same crops, encoders and student features as the MEVA clips, so alerts and MEVA clips share one score scale. A FastAPI back end and a React front end expose three cost levels (Fast: retriever; Standard: + student; Verified: + 2B VLM on the top k) and report the calls and seconds of every query.`));

// ---- 6
doc.push(H1("6. Experiments and Results"));
doc.push(P(`Results are given in the order the work was done. Every number below is on the evaluation split unless marked validation; nothing was selected on evaluation numbers. Raw outputs, per-query tables and the scripts that produce each table are in the repository (docs/results/raw).`));
doc.push(H2("6.1 Zero-shot encoder and frozen baseline"));
doc.push(P(`With no training, the tight view and SigLIP 2 gave the best validation separation (AUC 0.697, P@10 0.22) and this held on evaluation: Q1 mAP ${f(base("A", "siglip2-tight", "q1", "ap"))} with tight crops against ${f(base("A", "siglip2-full", "q1", "ap"))} with full frames (random order: ${f(base("A", "random", "q1", "ap"))}). Protocol B gave ${f(base("B", "siglip2-tight", "q1", "ap"))}, equal to A, as expected for a model that has never seen any of the cameras. Per activity, the frozen encoder finds phone use and large vehicle/door events well but is at chance on activities defined by motion: picking up, putting down, standing up, transferring an object. A single frame of someone standing up looks like someone seated; per-frame embeddings carry no motion. This is the gap the trained components target.`));
doc.push(H2("6.2 Domain adaptation (LoRA)"));
doc.push(TABLE(["Protocol", "System", "Q1 mAP", "Q1 P@10", "Q2 S@1", "Q2 S@5", "Q2 S@10", "Q2 MRR", "Q2 R@10"], ["A", "B"].flatMap((Pp) => [
  [Pp, "frozen SigLIP 2", f(base(Pp, "siglip2-tight", "q1", "ap")), f(base(Pp, "siglip2-tight", "q1", "p10")), f(cas("2b", Pp, "q2", "frozen", "s1")), f(cas("2b", Pp, "q2", "frozen", "s5")), f(cas("2b", Pp, "q2", "frozen", "s10")), f(base(Pp, "siglip2-tight", "q2", "rr")), f(base(Pp, "siglip2-tight", "q2", "r10"))],
  [Pp, { t: `LoRA (${SEL[Pp]}, selected)`, b: true }, f(lora(SEL[Pp], "q1", "ap")), f(lora(SEL[Pp], "q1", "p10")), f(lora(SEL[Pp], "q2", "s1")), f(lora(SEL[Pp], "q2", "s5")), f(lora(SEL[Pp], "q2", "s10")), f(lora(SEL[Pp], "q2", "rr")), f(lora(SEL[Pp], "q2", "r10"))],
  [Pp, `LoRA (${Pp === "A" ? "A-uniform" : "B-middle"}, ablation)`, f(lora(Pp === "A" ? "A-uniform" : "B-middle", "q1", "ap")), f(lora(Pp === "A" ? "A-uniform" : "B-middle", "q1", "p10")), f(lora(Pp === "A" ? "A-uniform" : "B-middle", "q2", "s1")), f(lora(Pp === "A" ? "A-uniform" : "B-middle", "q2", "s5")), f(lora(Pp === "A" ? "A-uniform" : "B-middle", "q2", "s10")), f(lora(Pp === "A" ? "A-uniform" : "B-middle", "q2", "rr")), f(lora(Pp === "A" ? "A-uniform" : "B-middle", "q2", "r10"))],
]), [0.09, 0.27, 0.09, 0.1, 0.09, 0.09, 0.09, 0.09, 0.09], { numeric: true, textCols: 2 }));
doc.push(CAP("Table 3. Frozen encoder against the LoRA-adapted encoder (evaluation split, 0 VLM calls). S@K = Success@K on forensic queries."));
doc.push(P([`Adaptation improves every metric on both protocols. Q1 mAP gains ${CI.loraA_map} (A) and ${CI.loraB_map} (B); forensic Success@1 gains ${CI.loraA_s1} and ${CI.loraB_s1} (95 % CIs over activities). An investigator now finds a correct event at rank 1 for ${Math.round(100 * lora("A-middle", "q2", "s1"))} % of forensic queries (was ${Math.round(100 * cas("2b", "A", "q2", "frozen", "s1"))} %) and within the top 10 for ${Math.round(100 * lora("A-middle", "q2", "s10"))} % (was ${Math.round(100 * cas("2b", "A", "q2", "frozen", "s10"))} %). On A all 22 activities improved; the largest gains are on interactions the frozen model barely separated: "person talks to person" AP ${f(pa("person_talks_to_person").A.ap_base)} → ${f(pa("person_talks_to_person").A.ap_lora)}, "carries heavy object" ${f(pa("person_carries_heavy_object").A.ap_base)} → ${f(pa("person_carries_heavy_object").A.ap_lora)}, "texts on phone" ${f(pa("person_texts_on_phone").A.ap_base)} → ${f(pa("person_texts_on_phone").A.ap_lora)}. The same activities gain on unseen cameras (B), so this is not camera memorisation, and the training captions are disjoint from the evaluation phrasings, so it is not query leakage. About 80 % of the A gain transfers to cameras never seen in training (B), which is the main reading of the A–B comparison; with one seed the remaining gap cannot be attributed to camera familiarity alone. Motion-defined activities rose off chance but stayed the weakest (AP ≈ 0.2 for pick up / put down), which is why a temporal component was still needed. Frame-sampling variants (uniform vs middle-weighted) tie on evaluation. Training is cheap: best epochs ${N.lora.runs["B-uniform"].training.best_epoch}–${N.lora.runs["A-uniform"].training.best_epoch}, about 57 s per epoch on an RTX 4060 laptop GPU, 1.5 h for all four runs.`]));

doc.push(H2("6.3 The re-ranking cascade and the recall-versus-cost curve"));
const fc = (c) => Number(c) === 0 ? "0" : f(c, 1);
const rowsT5 = (Pp) => {
  const r = (label, blk, sys, q1sys) => [Pp, label, fc(cas(blk, Pp, "q2", sys, "cost")), f(cas(blk, Pp, "q2", sys, "s1")), f(cas(blk, Pp, "q2", sys, "s5")), f(cas(blk, Pp, "q2", sys, "rr")), f(cas(blk, Pp, "q1", q1sys || sys, "ap"))];
  return [
    r("frozen SigLIP 2", "2b", "frozen"), r("LoRA", "2b", "lora"),
    [Pp, { t: "LoRA + temporal student (GT, selected)", b: true }, "0", { t: f(cas("2b", Pp, "q2", "student", "s1")), b: true }, f(cas("2b", Pp, "q2", "student", "s5")), f(cas("2b", Pp, "q2", "student", "rr")), f(cas("2b", Pp, "q1", "student", "ap"))],
    r("  student with GT+KD targets (ablation)", "2b", "student[GTKD]"), r("  student with KD targets only (ablation)", "2b", "student[KD]"),
    r("  student alone, no fusion with LoRA", "2b", "student-only"),
    r("LoRA → 2B VLM on top 10 (no student)", "2b", "lora>instruct@10"),
    r("LoRA + student → 2B on top 5", "2b", "student>instruct@5"), r("LoRA + student → 2B on top 10", "2b", "student>instruct@10"),
    r("LoRA + student → 8B on top 10", "8b", "student>instruct@10"),
    [Pp, { t: "LoRA + student → 32B on top 10", b: true }, f(cas("32b", Pp, "q2", "student>instruct@10", "cost"), 1), { t: f(cas("32b", Pp, "q2", "student>instruct@10", "s1")), b: true }, f(cas("32b", Pp, "q2", "student>instruct@10", "s5")), f(cas("32b", Pp, "q2", "student>instruct@10", "rr")), f(cas("32b", Pp, "q1", "student>instruct@10", "ap"))],
    r("LoRA + student → 32B on top 20", "32b", "student>instruct@20"),
    [Pp, "2B VLM on every candidate", f(cas("2b", Pp, "q2", "instruct-all", "cost"), 1), f(cas("2b", Pp, "q2", "instruct-all", "s1")), f(cas("2b", Pp, "q2", "instruct-all", "s5")), f(cas("2b", Pp, "q2", "instruct-all", "rr")), f(cas("q1all", Pp, "q1", "instruct-all", "ap")) + " †"],
    [Pp, "2B on every candidate + LoRA", f(cas("2b", Pp, "q2", "instruct-all+lora", "cost"), 1), f(cas("2b", Pp, "q2", "instruct-all+lora", "s1")), f(cas("2b", Pp, "q2", "instruct-all+lora", "s5")), f(cas("2b", Pp, "q2", "instruct-all+lora", "rr")), f(cas("q1all", Pp, "q1", "instruct-all+lora", "ap")) + " †"],
  ];
};
doc.push(TABLE(["Prot.", "System", "VLM calls / Q2 query", "Q2 S@1", "Q2 S@5", "Q2 MRR", "Q1 mAP"], [...rowsT5("A"), ...rowsT5("B")],
  [0.07, 0.43, 0.12, 0.095, 0.095, 0.095, 0.095], { numeric: true, textCols: 2, shade: (r) => r[0] === "B" }));
doc.push(CAP(`Table 4. The cost curve on the evaluation split. Q1 columns of the VLM rows use the same budget k on the whole-corpus ranking; † = the VLM scored every Q1 candidate (${cas("q1all", "A", "q1", "instruct-all", "cost").toLocaleString()} calls per query on A).`));
doc.push(IMG("fig2_cost_curve.png", 6.5));
doc.push(CAP("Figure 2. Forensic Success@1 against VLM calls per query. The temporal student at 0 calls captures most of the gain; larger VLMs add to it on a short list; the 2B VLM on every candidate is below LoRA alone."));
doc.push(P([{ t: "The student is the main result. ", b: true }, `At 0 VLM calls, fusing the temporal student with the adapted retriever lifts forensic Success@1 from ${f(cas("2b", "A", "q2", "lora", "s1"))} to ${f(cas("2b", "A", "q2", "student", "s1"))} on A and ${f(cas("2b", "B", "q2", "lora", "s1"))} to ${f(cas("2b", "B", "q2", "student", "s1"))} on B (${CI.studA_s1} and ${CI.studB_s1}); Q1 mAP gains ${CI.studA_map} and ${CI.studB_map}. The gain is where single frames fail: on the motion-defined activities (pick up, put down, stand up) Q1 AP rises from ${f2(motionA.lora)} to ${f2(motionA.student)} on average, against ${f2(restA.lora)} → ${f2(restA.student)} for the other ${restA.n} (Fig. 3). The fused score beats the student alone (${f(cas("2b", "A", "q2", "student-only", "s1"))} on A): appearance retriever and temporal student are complementary. The student costs about ${f(studentS, 2)} s per query for the whole corpus.`]));
doc.push(P([{ t: "What a VLM adds, and at what cost. ", b: true }, `The 2B VLM on the student's top 10 adds ${CI.vlm2bA_s1} Success@1 on A and ${CI.vlm2bB_s1} on B: at most +0.03, for ~10 calls (≈ 1.8 s). The 8B adds ${bs("8b", "A: 8B@10 - student")} on A and ${bs("8b", "B: 8B@10 - student")} on B; the 32B ${bs("32b", "A: 32B@10 - student")} on A and ${bs("32b", "B: 32B@10 - student")} on B, the first gain that is significant on unseen cameras. The 32B is not significantly better than the 8B (${bs("32b", "A: 32B@10 - 8B@10")} on A) while costing ${f(vlmS["32b"], 2)} s per call against ${f(vlmS["8b"], 2)} s (${Math.round(vlmS["32b"] / vlmS["8b"])}×). Running the 2B VLM on every candidate is worse than the adapted retriever alone (Q2 S@1 ${f(cas("2b", "A", "q2", "instruct-all", "s1"))} vs ${f(cas("2b", "A", "q2", "lora", "s1"))} on A; Q1 mAP ${f(cas("q1all", "A", "q1", "instruct-all", "ap"))} at ~10 minutes per query against ${f(cas("2b", "A", "q1", "lora", "ap"))} in 40 ms). Fusing it with LoRA only recovers LoRA. The curve of Fig. 2 therefore does not have the strongly concave shape the proposal expected: it starts flat (the VLM is not a better ranker than the adapted encoder) and becomes useful only as a final check on a short, already good list, with 5–10 calls as the knee. The proposal's fourth hypothesis, that the cascade reaches ≥ 90 % of "VLM on every candidate" at ≤ 10 % of the calls, is met trivially (152 % at 4 % of the calls on A) because the comparison point is weak.`]));
doc.push(IMG("fig3_per_activity.png", 4.1));
doc.push(CAP("Figure 3. Q1 average precision per activity on protocol A, frozen → LoRA → LoRA + student, sorted by the frozen score. The largest student gains are on activities defined by motion."));

doc.push(H2("6.4 Which VLMs were tried, and what the gates said"));
const vrow = (name, params, g0k, cfgKey, g1k, extra, verdict) => [name, params, g0k ? f(g0(g0k, "peak_vram_gb"), 1) : "—", g0k ? f2(g0(g0k, "seconds_per_pair_uncached")) : "—",
  g0k ? (N.g0[g0k].pass ? "pass" : "fail") : "—", `${g1(g1k, cfgKey, "A") >= 0 ? "+" : ""}${f(g1(g1k, cfgKey, "A"))} / ${g1(g1k, cfgKey, "B") >= 0 ? "+" : ""}${f(g1(g1k, cfgKey, "B"))}`, extra, verdict];
doc.push(TABLE(["VLM", "Params", "Peak VRAM (GB)", "s / call", "G0 fit", "G1 gain A / B (top 50)", "Whole validation", "Verdict"], [
  vrow("Qwen3-VL-2B-Instruct [5]", "2B", "qwen3vl_2b_instruct", "instruct-tight", "2b_top50", `${f(g1("2b_wholeval", "instruct-tight", "A"))} / ${f(g1("2b_wholeval", "instruct-tight", "B"))}`, "fail; bf16 shortcut noise flagged"),
  vrow("Qwen3-VL-Reranker-2B", "2B", "qwen3vl_2b_reranker", "reranker-tight", "2b_top50", `${f(g1("2b_wholeval", "reranker-tight", "A"))} / ${f(g1("2b_wholeval", "reranker-tight", "B"))}`, "fail; best mean gain → used as the 2B teacher by the pre-set rule"),
  ["Qwen3-VL-2B, 16 frames instead of 8", "2B", "—", "—", "over budget", `+${f(g1("2b_16frames", "instruct-tight", "A"))} / +${f(g1("2b_16frames", "instruct-tight", "B"))}`, "—", "fail; more frames do not help"],
  vrow("Gemma 4 E4B-it", "4B eff.", "gemma4_e4b", "gemma-e4b-tight", "gemma_top50", "—", "fail; weaker than Qwen 2B on A"),
  vrow("Qwen2.5-VL-3B-Instruct", "3B", "qwen25vl_3b", "q25-3b-tight", "qwen25_3b_top50", "—", "fail"),
  vrow("Qwen2.5-VL-7B-Instruct (FP8)", "7B", "qwen25vl_7b_fp8", "q25-7b-fp8-tight", "qwen25_7b_top50", "—", "fail"),
  vrow("Qwen3-VL-8B-Instruct (FP8, office server)", "8B", "qwen3vl_8b_fp8", "instruct-tight", "8b_top50", "—", "fail by 0.001; kept as a short-list stage for comparison"),
  vrow("Qwen3-VL-32B-Instruct (FP8, CPU/GPU split)", "32B", "qwen3vl_32b_fp8", "instruct-tight", "32b_top50", `+${f(g1("32b_wholeval", "instruct-tight", "A"))} / +${f(g1("32b_wholeval", "instruct-tight", "B"))}`, { t: "pass (both depths); G0 fails on speed only", b: true }),
], [0.22, 0.07, 0.09, 0.08, 0.08, 0.14, 0.12, 0.2], { size: 15 }));
doc.push(CAP("Table 5. Every VLM tested as a re-ranker of the adapted retriever's list, with the two gates (validation split). G1 gain = validation Q1 mAP added by fusing the VLM into the top 50; the pass bar is +0.02 on one protocol."));
doc.push(P([`Seven models were gated and six failed. The reason is consistent across models: inside the retriever's own top 50, the VLMs rank right clips above wrong ones with an AUC of ${f2(N.g1_auc["2b"].A["auc_instruct-tight"])} (2B Instruct), ${f2(N.g1_auc.gemma.A["Gemma4-E4B"])} (Gemma) and ${f2(N.g1_auc["8b"].A.Qwen8B)} (8B) on A, while the adapted retriever itself reaches ${f2(N.g1_auc["8b"].A.LoRA)}. The VLM signal is real (ranking the top 50 by the 2B VLM alone gives ${f2(N.g1["2b_top50"].configs["instruct-tight"].A.vlm_only_val_map)} mAP against ${f2(N.g1_auc["2b"].A.random_order_map)} for a random order) but weaker than the retriever it is asked to improve, so the validation always chose the largest retriever weight. Only the 32B model added information (${f(g1("32b_top50", "instruct-tight", "A"))} / ${f(g1("32b_top50", "instruct-tight", "B"))} mAP on the top 50 and ${f(g1("32b_wholeval", "instruct-tight", "A"))} / ${f(g1("32b_wholeval", "instruct-tight", "B"))} over the whole validation set), and even that gain, carried through the cascade, equals the 8B's (§6.3). Two secondary findings: the tight actor view beats the context view for every VLM, repeating the encoder result; and no VLM, and no student either, can tell a left turn from a right turn (per-query AUC 0.49–0.51 inside windows that contain both), a known weakness of image-plane features that a merged "vehicle turns" class resolves (LoRA AP ${f(Number(turns("lora").q1_ap))} on A and ${f(Number(turnsB("lora").q1_ap))} on B for the merged class, against 0.26–0.35 per direction).`]));

doc.push(H2("6.5 Distillation did not help"));
doc.push(TABLE(["Teacher", "Protocol", "GT (human labels)", "GT + KD", "KD only", "Selected"], [
  ["Qwen3-VL-Reranker-2B", "A", f(N.students.A.val_map.GT), f(N.students.A.val_map.GTKD), f(N.students.A.val_map.KD), "GT"],
  ["Qwen3-VL-Reranker-2B", "B", f(N.students.B.val_map.GT), f(N.students.B.val_map.GTKD), f(N.students.B.val_map.KD), "GT"],
  ["Qwen3-VL-32B (retrained on the office server)", "A", f(S32.A.GT), f(S32.A.GTKD), f(S32.A.KD), "GT"],
  ["Qwen3-VL-32B (retrained on the office server)", "B", f(S32.B.GT), f(S32.B.GTKD), f(S32.B.KD), "GT"],
], [0.36, 0.1, 0.15, 0.13, 0.13, 0.13], { numeric: true }));
doc.push(CAP("Table 6. Validation Q1 mAP of the student by training target. KD = teacher soft scores; GT+KD = their average with the human label."));
doc.push(P(`With either teacher, adding its soft scores to the human labels changes nothing measurable (GT+KD − GT: ${CI.gtkdA} on A for the 2B; −0.03 for the 32B), and training on the teacher alone loses badly (KD − GT: ${CI.kdA}). The teachers are weak on the training pairs themselves: AUC 0.62 (2B) and 0.69 (32B) for right against wrong clips, with soft targets that barely separate positives (0.46) from negatives (0.41). A teacher that is less accurate than the labels cannot teach the student more than the labels do. The proposal's second contribution therefore fails as stated, and the honest replacement is simpler: a cheap temporal model trained directly on the annotations.`));

doc.push(H2("6.6 Image + text LoRA (added experiment)"));
doc.push(P(`After the main results, one more question was tested: does adapting the text tower as well help? Four runs (two seeds × two protocols) with LoRA on both towers were compared with image-only runs on validation under a rule fixed in advance (adopt if the mean gain ≥ +0.010 and no protocol drops by more than 0.005). The two-seed means were ${f(N.e8.verdict.mean_diff["A-middle"])} on seen cameras (A, ${f((N.e8.runs["lora_io/A-middle"].best_val_map + N.e8.runs["lora_io_s1/A-middle"].best_val_map) / 2)} → ${f((N.e8.runs["lora_it/A-middle"].best_val_map + N.e8.runs["lora_it_s1/A-middle"].best_val_map) / 2)}) and +${f(N.e8.verdict.mean_diff["B-uniform"])} on unseen cameras (B, ${f((N.e8.runs["lora_io/B-uniform"].best_val_map + N.e8.runs["lora_io_s1/B-uniform"].best_val_map) / 2)} → ${f((N.e8.runs["lora_it/B-uniform"].best_val_map + N.e8.runs["lora_it_s1/B-uniform"].best_val_map) / 2)}). The B gain repeats across seeds and is about three times the seed noise (±0.017); the A loss fails the rule, so the image-only model stays. A text-drift guard confirmed the choice: on 40 appearance sentences never used in training, the adapted text towers reorder general wordings (Spearman ρ ${f2(N.drift.results.B0.sentences.rho)}–${f2(N.drift.results.A1.sentences.rho)} against the frozen tower, bar 0.90). Text-side adaptation is a promising route for unseen cameras but needs a retention objective first.`));

doc.push(H2("6.7 Hypotheses scorecard"));
doc.push(TABLE(["Hypothesis (fixed before the result)", "Verdict", "Basis"], [
  ["P1 (proposal): LoRA adaptation beats the frozen encoder on Recall@10 / mAP", "Confirmed", `Q1 mAP ${f(base("A", "siglip2-tight", "q1", "ap"))} → ${f(lora("A-middle", "q1", "ap"))}, Q2 R@10 ${f(base("A", "siglip2-tight", "q2", "r10"))} → ${f(lora("A-middle", "q2", "r10"))} (A); all CIs exclude zero`],
  ["P2 (proposal): a distilled student keeps most of the teacher's quality", "Rejected", "No 2B–8B teacher passes G1; KD ≪ GT, GT+KD ≈ GT for the 2B and the 32B teacher"],
  ["P3 (proposal): the accuracy-vs-cost curve is strongly concave", "Partly", "Concave after the student (knee at 5–10 calls); flat/negative for the exhaustive 2B VLM"],
  ["H1: the student beats LoRA at 0 calls, most on motion-defined activities", "Confirmed", `Motion group AP +${f2(motionA.student - motionA.lora)} vs +${f2(restA.student - restA.lora)} for the rest; every CI excludes zero`],
  ["H2: LoRA → VLM top-10 beats LoRA on Q2 S@1", "Not tested (G1 failed); +0.02–0.03 measured", "CIs touch zero for the 2B"],
  ["H3: GT+KD student ≥ GT student", "Rejected", `${CI.gtkdA} (A)`],
  ["H4: student → VLM top-5 reaches ≥ 90 % of exhaustive-VLM S@1 at ≤ 10 % of the calls", "Confirmed, trivially", "152 % at 4 % of the calls (A): the exhaustive 2B is a weak reference"],
  ["H5: the VLM is at chance on left vs right turns, the student is not", "Half: both at chance", "AUC 0.49–0.51 for VLM, student, LoRA"],
], [0.42, 0.18, 0.4], { size: 16 }));
doc.push(CAP("Table 7. The proposal's three hypotheses (P1–P3) and the five cascade hypotheses (H1–H5)."));

// ---- 7
doc.push(H1("7. System Demonstration"));
doc.push(P(`The retrieval pipeline runs end to end as a local dashboard (Fig. 4). The Search tab answers free text over the 3,726 MEVA evaluation clips, over alerts from uploaded videos, or both, at the three cost levels, and shows for each query the clips searched, the seconds per stage, the VLM calls made and what a VLM on every clip would have cost (3,726 calls ≈ 10 minutes). For an evaluation phrasing it marks which results carry the label; "a person opening a car door" at the Standard level returns ${"6"} of the top 10 with that label in 0.6 s and 0 VLM calls, against 4 at the Fast level. A test confirms that the dashboard reproduces the evaluated system: Success@1 and Success@10 are identical to the evaluation on every one of the 1,206 Q1 and Q2 queries. The Detect tab turns any video into searchable alerts: a 2.5-minute CCTV clip of a parking garage produced 22 person and vehicle alerts, each a 5-second clip with the boxes drawn, searchable seconds after the upload finished (a 10 s 720p video was searchable 22 s after upload, including loading the detector). The 3D clip map projects every clip's embedding (PCA of the LoRA vectors) and lights up the results around the query; it is an illustration, not a measurement. Confidence bands (High / Medium / Low) place a clip's score among the MEVA scores for the same query; they are relative, not probabilities. Footage unlike MEVA (indoor, close-up) is found but typically rates Low and sits in its own region of the map: a demonstration, not a measured result, since no labels exist for it.`));
doc.push(IMG("fig4a_search.png", 5.6));
doc.push(CAP("Figure 4a. Search tab: an evaluation phrasing at the Standard level with labels shown (6 of the top 10 carry the label), and the cost line."));
doc.push(new Table({ columnWidths: [Math.round(TEXT_W / 2), Math.round(TEXT_W / 2)], width: { size: TEXT_W, type: WidthType.DXA }, layout: TableLayoutType.FIXED,
  borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE }, left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE }, insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE } },
  rows: [new TableRow({ children: ["fig4b_detect.png", "fig4c_map.png"].map((nm) => {
    const buf = fs.readFileSync(FIG(nm)); const s = pngSize(buf); const w = 272, h = Math.round(w * s.h / s.w);
    return new TableCell({ width: { size: Math.round(TEXT_W / 2), type: WidthType.DXA }, margins: { top: 0, bottom: 0, left: 40, right: 40 },
      children: [new Paragraph({ children: [new ImageRun({ type: "png", data: buf, transformation: { width: w, height: h } })], alignment: AlignmentType.CENTER, spacing: { after: 0 } })] });
  }) })] }));
doc.push(CAP("Figure 4b–c. Detect tab with alerts from an uploaded CCTV clip (left); the 3D clip map with the query's top results lit (right)."));

// ---- 8
doc.push(H1("8. Limitations and Threats to Validity"));
for (const t of [
  [{ t: "Scripted data. ", b: true }, "MEVA activities are performed by actors at a training facility, so absolute numbers may be optimistic for uncontrolled footage; comparisons between arms, which share the data, remain valid."],
  [{ t: "Single training seed. ", b: true }, "Every selected model was trained once; the added experiment measured the seed-to-seed spread of the same set-up at ±0.017 validation mAP, so differences below ~0.02 that rest on one seed should be read with care. Confidence intervals are over activities, not seeds."],
  [{ t: "Small validation sets. ", b: true }, `${D.split.counts.val} and ${D.split_b.counts.val} validation instances (15–17 activities with ≥ 5), so the chosen epoch and sampling variant are noisy picks; on evaluation the variants tie.`],
  [{ t: "VLM set-up. ", b: true }, "The gates used 8 frames, the tight view and one default prompt; prompt or frame-layout changes might shift the 2B–8B verdicts, though 16 frames did not. The 32B was run with a CPU/GPU split, which is why it fails G0 on speed."],
  [{ t: "Turn direction. ", b: true }, "Left and right turns are not separable by any component; the box features are image-plane displacements, and heading relative to the camera is not encoded."],
  [{ t: "Cost accounting. ", b: true }, "Calls and seconds, on one laptop GPU and one office GPU; no token or monetary costs since no paid API was used."],
  [{ t: "Alerts from new video. ", b: true }, "No labels exist for them, so their retrieval quality is not measured; the confidence band is relative to MEVA."],
]) doc.push(UL(t));

// ---- 9
doc.push(H1("9. Benefits and Stakeholders"));
doc.push(P(`What the results support: post-incident review by a plain-language query instead of scrubbing by camera and time; a zero-VLM operating point (retriever + student, ≈ 0.2 s per query on a laptop GPU) that runs on modest on-premise hardware, so footage never leaves the operator's infrastructure; a VLM check on the top 10 as an optional confirmation step at about 1.5–2 s; and a reusable evaluation asset, the MEVA query set with its protocol. What they do not support: identification of people (the system retrieves events, not persons, by design) or automated determination of fact, since the best operating point still misses the right clip at rank 1 for four forensic queries in ten. Stakeholders are as in the proposal: security operators and investigators (time to evidence), facility and transport operators (review cost, no recurring API spend), the legal process (retrieval as an aid whose output is human-verified), members of the public under surveillance (a more capable archive is a privacy consideration; local inference and event-level retrieval are partial mitigations), and researchers (a cost-aware protocol on a CC BY 4.0 corpus).`));

// ---- 10
doc.push(H1("10. Conclusion and Future Work"));
doc.push(P(`Three findings. First, domain adaptation of a dual encoder with a small LoRA on 192 captions is the cheapest large gain available: it raises category-search mAP by 75 % and transfers most of that to cameras never seen in training. Second, the expensive component of the proposal, a VLM teacher, turned out to be the weakest ranker in the cascade: six of seven VLMs failed to add information to the adapted retriever, distillation from any of them did not beat the human labels, and the exhaustive 2B VLM is worse than the cheap stages. What replaced it, a temporal re-ranker over frame embeddings and box motion trained directly on the annotations, is the largest single step on the curve at zero VLM calls. Third, VLMs still earn their place as a final check on a short list: an 8B or 32B model on the top 10 adds a significant +0.06 Success@1 on both protocols, and beyond 10–20 calls nothing is gained. The cost curve is therefore measured, and its shape is the finding: adapt, re-rank cheaply, and spend the VLM calls at the end, not at the start.`));
doc.push(P(`Future work, in order of expected value: heading-relative motion features for turn direction; a retention objective so that text-side adaptation keeps its unseen-camera gain without drifting on general language; a second seed for the selected models; a prompt and frame-layout study for the VLM gate; and labelled evaluation footage from outside MEVA to turn the confidence bands into calibrated probabilities.`));

// ---- References
doc.push(H1("References"));
const refs = [
  "A. Radford et al., \"Learning transferable visual models from natural language supervision,\" in Proc. 38th Int. Conf. Machine Learning (ICML), 2021, pp. 8748–8763.",
  "X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, \"Sigmoid loss for language image pre-training,\" in Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), 2023; arXiv:2303.15343.",
  "M. Tschannen et al., \"SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features,\" arXiv:2502.14786, Feb. 2025.",
  "J. Li, D. Li, S. Savarese, and S. Hoi, \"BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,\" in Proc. 40th Int. Conf. Machine Learning (ICML), 2023.",
  "S. Bai et al. (Qwen Team), \"Qwen3-VL technical report,\" arXiv:2511.21631, Nov. 2025.",
  "E. J. Hu et al., \"LoRA: Low-rank adaptation of large language models,\" in Proc. Int. Conf. Learning Representations (ICLR), 2022; arXiv:2106.09685.",
  "G. Hinton, O. Vinyals, and J. Dean, \"Distilling the knowledge in a neural network,\" arXiv:1503.02531, 2015.",
  "S. Hofstätter, S. Althammer, M. Schröder, M. Sertkan, and A. Hanbury, \"Improving efficient neural ranking models with cross-architecture knowledge distillation,\" arXiv:2010.02666, 2020.",
  "L. Wang, J. Lin, and D. Metzler, \"A cascade ranking model for efficient ranked retrieval,\" in Proc. 34th Int. ACM SIGIR Conf., 2011, pp. 105–114.",
  "V. Escorcia, M. Soldan, J. Sivic, B. Ghanem, and B. Russell, \"Temporal localization of moments in video collections with natural language,\" arXiv:1907.12763, 2019.",
  "J. Lei, T. L. Berg, and M. Bansal, \"QVHighlights: Detecting moments and highlights in videos via natural language queries,\" in Advances in Neural Information Processing Systems (NeurIPS), 2021.",
  "H.-L. Tran et al., \"Towards efficient and robust moment retrieval system: A unified framework for multi-granularity models and temporal reranking,\" in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), 2025; arXiv:2504.08384.",
  "T. Nishimura, S. Nakada, and M. Kondo, \"Vision-language models learn super images for efficient partially relevant video retrieval,\" arXiv:2312.00414, 2023 (v2 2024).",
  "T. Yuan et al., \"Towards surveillance video-and-language understanding: New dataset, baselines, and challenges,\" in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2024; arXiv:2309.13925.",
  "P. Wu et al., \"VadCLIP: Adapting vision-language models for weakly supervised video anomaly detection,\" in Proc. AAAI Conf. Artificial Intelligence, 2024.",
  "M. Owais, A. Siddiqui, A. Alam, and S. Ali, \"SurveiLens: Natural language-based moment retrieval for surveillance video analysis,\" IEEE Xplore document 11400480, 2025–2026 (preprint Sept. 2025).",
  "H. Park, Y. Li, et al., \"ForeSea: AI forensic search with multi-modal queries for video surveillance,\" arXiv:2603.22872, Mar. 2026.",
  "K. Corona, K. Osterdahl, R. Collins, and A. Hoogs, \"MEVA: A large-scale multiview, multimodal video dataset for activity detection,\" in Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV), 2021, pp. 1060–1068.",
  "Y. Zhang et al., \"ByteTrack: Multi-object tracking by associating every detection box,\" in Proc. European Conf. Computer Vision (ECCV), 2022; arXiv:2110.06864.",
  "G. Jocher, A. Chaurasia, and J. Qiu, \"Ultralytics YOLO (version 8),\" software, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics",
];
refs.forEach((r, i) => doc.push(new Paragraph({ children: runs(`[${i + 1}] ${r}`, 18), spacing: { after: 40, line: 240 }, indent: { left: 400, hanging: 400 }, alignment: AlignmentType.LEFT })));

// ---- Appendix
doc.push(H1("Appendix A. Reproducibility"));
doc.push(P(`The repository holds the code (src/sfs: data, embed, train, teacher, rerank, eval, alerts, demo, web), the specifications written before each stage (docs/planning), the Result files with their raw outputs (docs/results, docs/results/raw) and an append-only experiment log (docs/EXPERIMENT_LOG.md, decisions D-001 to D-064). Every number in this report is read from report/numbers.json, generated from the raw outputs by scripts/report_numbers.py and checked by tests/test_report_numbers.py against the Result files; the confidence intervals of §6.2–6.5 and the 32B student table are quoted from Results 03–05. Figures 1–3 are drawn by scripts/report_figures.py from the same file.`));
doc.push(TABLE(["Stage", "Command (conda env dlproj, PYTHONPATH=src)", "Output"], [
  ["Embeddings + crop store", "python -m sfs.embed.run; python -m sfs.data.extract_crops", "data/crops_v2, per-video shards"],
  ["Frozen baseline", "python -m sfs.eval.run --protocol A|B", "Result 02"],
  ["LoRA (4 runs, selection, re-embed, eval)", "python -m sfs.train.run; python -m sfs.embed.reembed; scripts/lora_report.py", "Result 03, runs/lora"],
  ["Gates, teacher labels, students, cascade", "python -m sfs.teacher.run; python -m sfs.rerank.train; python -m sfs.eval.cascade_run report", "Results 04–05, runs/cascade, runs/student"],
  ["Image + text LoRA, drift guard", "python -m sfs.train.run --text-lora; scripts/text_drift_check.py", "Result 06"],
  ["Dashboard", "python -m sfs.demo.index --protocol A; python -m sfs.web.server (or docker compose up)", "http://127.0.0.1:8000"],
  ["Tests", "pytest (the unit suite, fixture data, no GPU); pytest -m slow (real-model parity tests)", "—"],
], [0.28, 0.5, 0.22], { size: 16 }));

// ------------------------------------------------------------------ assemble
const document = new Document({
  creator: "Wajahat Ahmed", title, description: "Final report, CS-5138 Deep Learning",
  styles: {
    default: { document: { run: { font: FONT, size: BODY }, paragraph: { spacing: { after: 80, line: 252 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 26, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 220, after: 80 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 23, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 140, after: 60 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 21, bold: true, italics: true }, paragraph: { spacing: { before: 100, after: 40 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Wajahat Ahmed · CS-5138 Deep Learning · Final Report · page ", PageNumber.CURRENT], size: 16, font: FONT, color: "666666" })] })] }) },
    children: doc,
  }],
});

Packer.toBuffer(document).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length, "bytes"); });
