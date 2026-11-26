// Builds the final report in the course template as Word: report/Ahmed_Wajahat_DLReport.docx (D-080).
// The text, tables and numbers are the same as report/latex/main.tex (the LaTeX version of this report, whose numbers
// come from report/numbers.json); change both together. Equations are native Word equations; the contents and the
// lists of figures and tables are inserted and updated by scripts/report_word_finish.ps1, which also exports the PDF.
//
//   cd report/build && node ../../scripts/report_template_build.js
//   powershell -File ../../scripts/report_word_finish.ps1 -Docx ..\Ahmed_Wajahat_DLReport.docx -Pdf ..\Ahmed_Wajahat_DLReport.pdf
//
"use strict";
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "..");
const lib = (name) => require(require.resolve(name, { paths: [path.join(ROOT, "report", "build")] }));
const {
  AlignmentType, BorderStyle, Document, ExternalHyperlink, Footer, Header, HeadingLevel, ImageRun, LevelFormat,
  Math: OMath, MathFraction, MathFunction, MathRoundBrackets, MathRun, MathSquareBrackets, MathSubScript, MathSubSuperScript,
  MathSum, MathSuperScript, NumberFormat, PageBreak, PageNumber, Packer, Paragraph, SequentialIdentifier, ShadingType,
  SimpleField, Tab, TabStopType, Table, TableCell, TableLayoutType, TableRow, TextRun, VerticalAlign, WidthType,
} = lib("docx");
const JSZip = lib("jszip");

const OUT = process.env.REPORT_OUT || path.join(ROOT, "report", "Ahmed_Wajahat_DLReport.docx");
const FIG = (name) => path.join(ROOT, "report", "figures", name);

// ------------------------------------------------------------------ details (placeholders are highlighted in yellow)
const INFO = {
  university: "NED University of Engineering & Technology",
  department: "Department of Computer and Information Systems Engineering (CIS)",
  title: "Cost-Bounded Natural-Language Forensic Search over Surveillance Video",
  student: "Wajahat Ahmed", id: "CS-141/2026", supervisor: "Dr Sharaf Hussain",
  course: "CS-5138 Deep Learning (Graduate)",
  date: "24/11/2026",
  link: "[https://drive.google.com/drive/folders/1tCvC671fWx6psa6T1RzxxCK8stBeic8U?usp=sharing](https://drive.google.com/drive/folders/1tCvC671fWx6psa6T1RzxxCK8stBeic8U?usp=sharing)",
};

// ------------------------------------------------------------------ page and type
const FONT = "Times New Roman";
const BODY = 24;                                   // 12 pt
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1417; // A4, 2.5 cm
const TEXT_W = PAGE_W - 2 * MARGIN;               // 9072 dxa = 6.3 in
const PX = (dxa) => Math.round(dxa / 1440 * 96);

// ------------------------------------------------------------------ inline markup
// **bold**  *italic*  `mono`  _{sub}  ^{sup}  ==highlight==  [text](url)
const TOK = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|_\{[^}]+\}|\^\{[^}]+\}|==[^=]+==|\[[^\]]+\]\(https?:[^)]+\))/;
function runs(s, o = {}) {
  const size = o.size || BODY;
  const base = { font: o.font || FONT, size, bold: o.bold, italics: o.italics, color: o.color };
  const out = [];
  for (const part of String(s).split(TOK)) {
    if (!part) continue;
    let m;
    if (part.startsWith("**") && part.endsWith("**")) out.push(new TextRun({ ...base, text: part.slice(2, -2), bold: true }));
    else if (/^\*[^*]/.test(part) && part.endsWith("*") && part.length > 2) out.push(new TextRun({ ...base, text: part.slice(1, -1), italics: true }));
    else if (part.startsWith("`")) out.push(new TextRun({ ...base, text: part.slice(1, -1), font: "Consolas", size: size - 3 }));
    else if (part.startsWith("_{")) out.push(new TextRun({ ...base, text: part.slice(2, -1), subScript: true }));
    else if (part.startsWith("^{")) out.push(new TextRun({ ...base, text: part.slice(2, -1), superScript: true }));
    else if (part.startsWith("==")) out.push(new TextRun({ ...base, text: part.slice(2, -2), highlight: "yellow" }));
    else if ((m = part.match(/^\[([^\]]+)\]\((https?:[^)]+)\)$/)))
      out.push(new ExternalHyperlink({ link: m[2], children: [new TextRun({ ...base, text: m[1], color: "0563C1", underline: {} })] }));
    else out.push(new TextRun({ ...base, text: part }));
  }
  return out;
}
const P = (s, o = {}) => new Paragraph({
  children: runs(s, o), alignment: o.align || AlignmentType.JUSTIFIED, keepNext: o.keepNext, keepLines: o.keepLines,
  spacing: { before: o.before ?? 0, after: o.after ?? 120, line: o.line ?? 276 }, indent: o.indent,
});
const H1 = (num, t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(num ? `${num} ${t}` : t)] });
const H2 = (num, t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(`${num} ${t}`)] });
const UL = (s) => new Paragraph({ children: runs(s), numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 276 } });
const OL = (s) => new Paragraph({ children: runs(s), numbering: { reference: "steps", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 276 } });
const BREAK = () => new Paragraph({ children: [new PageBreak()] });

// ------------------------------------------------------------------ captions (SEQ fields: Word builds the lists from them)
const CAP = (label, text) => new Paragraph({ style: "Caption", keepNext: label !== "Figure",
  children: [new TextRun(`${label} `), new SequentialIdentifier(label), new TextRun(`: ${text}`)] });
const CAPNOTE = (s) => new Paragraph({ children: runs(s, { size: 20 }), alignment: AlignmentType.CENTER, spacing: { before: 0, after: 200, line: 252 } });

// ------------------------------------------------------------------ figures
function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
function image(name, o) {
  const buf = fs.readFileSync(FIG(name));
  const s = pngSize(buf);
  const w = o.w ?? Math.round(o.h * s.w / s.h), h = o.h ?? Math.round(o.w * s.h / s.w);
  return new ImageRun({ type: "png", data: buf, transformation: { width: w, height: h },
    altText: { title: o.alt, description: o.alt, name: name } });
}
const FIGURE = (name, frac, alt) => new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
  children: [image(name, { w: Math.round(PX(TEXT_W) * frac), alt })] });

// ------------------------------------------------------------------ booktabs-style tables
const RULE = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: "000000" });
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
// rows: arrays of cells; a cell is a string (markup) or { t, span, rowSpan, align, bold }
// o: { widths (fractions), header (number of header rows), align (per column), size, groups (row indices that start a group) }
function TABLE(rows, o) {
  const widths = o.widths.map((x) => Math.round(TEXT_W * x));
  const nHead = o.header ?? 1, size = o.size || 20, groups = new Set(o.groups || []);
  const last = rows.length - 1;
  const covered = new Array(widths.length).fill(0);   // rows still covered by a rowSpan cell above, per column
  const trs = rows.map((r, i) => {
    let col = 0;
    const skip = () => { while (col < widths.length && covered[col] > 0) { covered[col] -= 1; col += 1; } };
    const cells = r.map((c) => {
      skip();
      const cell = typeof c === "object" ? c : { t: c };
      const span = cell.span || 1;
      if (cell.rowSpan > 1) for (let k = col; k < col + span; k++) covered[k] = cell.rowSpan - 1;
      const w = widths.slice(col, col + span).reduce((a, b) => a + b, 0);
      const align = cell.align || (i < nHead ? (span > 1 ? "center" : (o.align?.[col] || "left")) : (o.align?.[col] || "left"));
      col += span;
      const top = i === 0 ? RULE(12) : (i === nHead || groups.has(i)) ? RULE(6) : NONE;
      const bottom = i === last ? RULE(12) : (cell.rule ? RULE(4) : NONE);
      return new TableCell({
        width: { size: w, type: WidthType.DXA }, columnSpan: span > 1 ? span : undefined, rowSpan: cell.rowSpan,
        verticalAlign: VerticalAlign.CENTER, margins: { top: 40, bottom: 40, left: 80, right: 80 },
        borders: { top, bottom, left: NONE, right: NONE },
        // keep-with-next on every row but the last keeps the table on one page
        children: [new Paragraph({ children: runs(cell.t, { size, bold: i < nHead || cell.bold }), spacing: { after: 0, line: 240 }, keepNext: i < last,
          alignment: { left: AlignmentType.LEFT, center: AlignmentType.CENTER, right: AlignmentType.RIGHT }[align] })],
      });
    });
    return new TableRow({ tableHeader: i < nHead, cantSplit: true, children: cells });
  });
  return new Table({ columnWidths: widths, width: { size: TEXT_W, type: WidthType.DXA }, layout: TableLayoutType.FIXED,
    alignment: AlignmentType.CENTER, borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
    rows: trs });
}
const AFTER_TABLE = () => new Paragraph({ spacing: { after: 120 }, children: [] });
const TNOTE = (s) => new Paragraph({ children: runs(s, { size: 20 }), alignment: AlignmentType.CENTER, spacing: { before: 80, after: 240, line: 252 } });

// ------------------------------------------------------------------ code blocks and the algorithm box
function CODE(text) {
  const lines = text.replace(/^\n/, "").replace(/\n\s*$/, "").split("\n");
  return new Table({ columnWidths: [TEXT_W], width: { size: TEXT_W, type: WidthType.DXA }, layout: TableLayoutType.FIXED,
    borders: { top: RULE(4), bottom: RULE(4), left: RULE(4), right: RULE(4), insideHorizontal: NONE, insideVertical: NONE },
    rows: [new TableRow({ children: [new TableCell({ width: { size: TEXT_W, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "F2F0EA", color: "auto" }, margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: lines.map((l) => new Paragraph({ spacing: { after: 0, line: 228 },
        children: [new TextRun({ text: l.length ? l : " ", font: "Consolas", size: 14 })] })) })], cantSplit: true })] });
}
function ALGORITHM(title, require_, lines) {
  const para = (num, level, s) => new Paragraph({ spacing: { after: 0, line: 228 }, tabStops: [{ type: TabStopType.LEFT, position: 400 }],
    indent: { left: 400 + level * 360, hanging: 400 + level * 360 },
    children: [new TextRun({ text: num ? `${num}` : "", font: FONT, size: 20 }), new TextRun({ children: [new Tab()] }),
      ...(level ? [new TextRun({ text: " ".repeat(level * 4), size: 20 })] : []), ...runs(s, { size: 21 })] });
  const children = [
    new Paragraph({ spacing: { after: 60 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 2 } },
      children: runs(`**Algorithm 1** ${title}`, { size: 22 }) }),
    para("", 0, `**Require:** ${require_}`),
    ...lines.map(([lvl, s], k) => para(k + 1, lvl, s)),
  ];
  return new Table({ columnWidths: [TEXT_W], width: { size: TEXT_W, type: WidthType.DXA }, layout: TableLayoutType.FIXED,
    borders: { top: RULE(12), bottom: RULE(12), left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
    rows: [new TableRow({ cantSplit: false, children: [new TableCell({ width: { size: TEXT_W, type: WidthType.DXA },
      margins: { top: 40, bottom: 80, left: 0, right: 0 }, children })] })] });
}

// ------------------------------------------------------------------ equations (native Word math)
// Runs that start with U+2063 are set upright (m:sty p) in a post-processing step, like \mathrm in LaTeX.
const UP = "⁣";
const m = (s) => new MathRun(s);
const up = (s) => new MathRun(UP + s);
const sub = (b, s) => new MathSubScript({ children: [].concat(b), subScript: [].concat(s) });
const supr = (b, s) => new MathSuperScript({ children: [].concat(b), superScript: [].concat(s) });
const subsup = (b, s, p) => new MathSubSuperScript({ children: [].concat(b), subScript: [].concat(s), superScript: [].concat(p) });
const frac = (n, d) => new MathFraction({ numerator: [].concat(n), denominator: [].concat(d) });
const sum = (lo, hi, body) => new MathSum({ children: [].concat(body), subScript: [].concat(lo), superScript: [].concat(hi) });
const sq = (c) => new MathSquareBrackets({ children: [].concat(c) });
const rnd = (c) => new MathRoundBrackets({ children: [].concat(c) });
const log = (c) => new MathFunction({ name: [up("log")], children: [].concat(c) });
const sigma = (c) => [m("σ"), rnd(c)];
let EQN = 0;
function EQ(children) {
  EQN += 1;
  return new Paragraph({ spacing: { before: 120, after: 120 }, keepLines: true,
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(TEXT_W / 2) }, { type: TabStopType.RIGHT, position: TEXT_W }],
    children: [new TextRun({ children: [new Tab()] }), new OMath({ children }), new TextRun({ children: [new Tab(), `(${EQN})`], font: FONT, size: BODY })] });
}
const bce = (y, s) => sq([y, log(sigma(s)), m("+"), rnd([m("1−"), y]), log(rnd([m("1−"), ...sigma(s)]))]);

// ------------------------------------------------------------------ references (order = LaTeX thebibliography)
const REFS = [
  ["clip", "A. Radford et al., “Learning transferable visual models from natural language supervision,” in *Proc. ICML*, 2021, pp. 8748–8763."],
  ["siglip", "X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in *Proc. ICCV*, 2023."],
  ["siglip2", "M. Tschannen et al., “SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features,” arXiv:2502.14786, 2025."],
  ["qwen3vl", "S. Bai et al. (Qwen Team), “Qwen3-VL technical report,” arXiv:2511.21631, 2025."],
  ["lora", "E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in *Proc. ICLR*, 2022."],
  ["hinton", "G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv:1503.02531, 2015."],
  ["hofstatter", "S. Hofstätter, S. Althammer, M. Schröder, M. Sertkan, and A. Hanbury, “Improving efficient neural ranking models with cross-architecture knowledge distillation,” arXiv:2010.02666, 2020."],
  ["cascade", "L. Wang, J. Lin, and D. Metzler, “A cascade ranking model for efficient ranked retrieval,” in *Proc. ACM SIGIR*, 2011, pp. 105–114."],
  ["escorcia", "V. Escorcia, M. Soldan, J. Sivic, B. Ghanem, and B. Russell, “Temporal localization of moments in video collections with natural language,” arXiv:1907.12763, 2019."],
  ["qvhighlights", "J. Lei, T. L. Berg, and M. Bansal, “QVHighlights: Detecting moments and highlights in videos via natural language queries,” in *Proc. NeurIPS*, 2021."],
  ["tran", "H.-L. Tran et al., “Towards efficient and robust moment retrieval system: A unified framework for multi-granularity models and temporal reranking,” in *Proc. CVPR Workshops*, 2025."],
  ["superimage", "T. Nishimura, S. Nakada, and M. Kondo, “Vision-language models learn super images for efficient partially relevant video retrieval,” arXiv:2312.00414, 2023."],
  ["uca", "T. Yuan et al., “Towards surveillance video-and-language understanding: New dataset, baselines, and challenges,” in *Proc. CVPR*, 2024."],
  ["vadclip", "P. Wu et al., “VadCLIP: Adapting vision-language models for weakly supervised video anomaly detection,” in *Proc. AAAI*, 2024."],
  ["surveilens", "M. Owais, A. Siddiqui, A. Alam, and S. Ali, “SurveiLens: Natural language-based moment retrieval for surveillance video analysis,” IEEE Xplore document 11400480, 2025."],
  ["foresea", "H. Park, Y. Li, et al., “ForeSea: AI forensic search with multi-modal queries for video surveillance,” arXiv:2603.22872, 2026."],
  ["meva", "K. Corona, K. Osterdahl, R. Collins, and A. Hoogs, “MEVA: A large-scale multiview, multimodal video dataset for activity detection,” in *Proc. WACV*, 2021, pp. 1060–1068."],
  ["bytetrack", "Y. Zhang et al., “ByteTrack: Multi-object tracking by associating every detection box,” in *Proc. ECCV*, 2022."],
  ["yolo", "G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLO (version 8),” 2023. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)"],
];
const CITE = Object.fromEntries(REFS.map(([k], i) => [k, i + 1]));
const c = (k) => { if (!CITE[k]) throw new Error(`unknown citation ${k}`); return `[${CITE[k]}]`; };

// ================================================================== title page
const titlePage = [
  P(INFO.university, { align: AlignmentType.CENTER, size: 32, before: 1200, after: 120 }),
  P(INFO.department, { align: AlignmentType.CENTER, size: 26, after: 1400 }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1000 },
    border: { top: { style: BorderStyle.SINGLE, size: 18, color: "000000", space: 10 }, bottom: { style: BorderStyle.SINGLE, size: 18, color: "000000", space: 10 } },
    children: runs(`**${INFO.title}**`, { size: 40 }) }),
  P("*A Deep Learning Project Report*", { align: AlignmentType.CENTER, size: 32, after: 1000 }),
  P("**Submitted by:**", { align: AlignmentType.CENTER, after: 60 }),
  P(`${INFO.student} (${INFO.id})`, { align: AlignmentType.CENTER, after: 360 }),
  P("**Supervised by:**", { align: AlignmentType.CENTER, after: 60 }),
  P(INFO.supervisor, { align: AlignmentType.CENTER, after: 360 }),
  P("**Course:**", { align: AlignmentType.CENTER, after: 60 }),
  P(INFO.course, { align: AlignmentType.CENTER, after: 2000 }),
  P(INFO.date, { align: AlignmentType.CENTER }),
];

// ================================================================== front matter
const centreHead = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 240 }, children: runs(`**${t}**`, { size: 32 }) });
const front = [
  P("**Abstract**", { align: AlignmentType.CENTER, size: 26, after: 160 }),
  P(`This report presents *${INFO.title}*: searching surveillance activity clips with a typed sentence (“a person opening a car door”) while making as few calls as possible to an expensive vision-language model (VLM). We use MEVA, a public multi-camera surveillance dataset, from which 10,678 annotated activity clips were downloaded, and a frozen test set of 66 queries over 22 activity types. Two protocols are evaluated: the official split, and a split whose eight test cameras never appear in training. The system has three stages. The SigLIP 2 image encoder is adapted to the surveillance domain with LoRA (1.18 M trained weights). A small temporal transformer (2.2 M weights) then re-ranks each clip from its frame embeddings and actor-box motion. An optional VLM checks only the top ten clips. Seven VLMs (2B to 32B parameters) were tested as rankers. With no VLM calls, category-search mAP rises from 0.179 (frozen encoder) to 0.313 with LoRA and 0.476 with the temporal transformer; forensic Success@1 rises from 0.27 to 0.53, with similar gains on unseen cameras. A 2B VLM applied to every clip scores below the cheap stages, distilling VLM scores into the student never beat training on the human labels, and an 8B or 32B VLM on the top ten adds a further +0.06 Success@1. The pipeline runs as a dashboard that turns any video into searchable alerts and reports the cost of each query.`,
    { indent: { left: 567, right: 567 } }),
  P("**Keywords:** deep learning, text-to-video retrieval, vision-language models, LoRA, temporal transformer, surveillance video, cost-aware re-ranking", { before: 240 }),
  BREAK(),
  centreHead("Contents"), P("@@TOC@@"),
  BREAK(),
  centreHead("List of Figures"), P("@@LOF@@"),
  new Paragraph({ spacing: { after: 240 }, children: [] }),
  centreHead("List of Tables"), P("@@LOT@@"),
];

// ================================================================== body
const B = [];
// ---------------------------------------------------------------- 1
B.push(H1("1", "Introduction"));
B.push(H2("1.1", "Background and Motivation"));
B.push(P("A surveillance installation produces hours of footage and thousands of automated alerts every day. When an incident is reported after the fact, investigators scrub that footage by camera and time, because nothing indexes what the footage *shows*. Natural-language search would let an investigator type “a person taking a bag from a parked car” and receive the matching clips. Vision-language models (VLMs) can judge such a query well: shown a clip, they answer “does this show …?”. But one VLM call per clip is slow, and over an archive it costs minutes per query (about 10 minutes for 3,726 clips with a 2B model on a laptop GPU) and real money with a paid API. The practical question is therefore not only how accurate the search is, but how much accuracy is kept as the number of expensive calls is cut."));
B.push(H2("1.2", "Problem Statement"));
B.push(P("**Input:** a free-text query *q* and a corpus of surveillance clips 𝒞, each an annotated activity instance (actors’ boxes, start and end time), optionally restricted to a camera and hour. **Output:** a ranking of 𝒞 by relevance to *q*. **Objective:** maximise retrieval quality (mAP, Success@*K*) under a budget of VLM calls per query, and measure the whole quality-versus-cost curve: how much of the accuracy of a VLM applied to every clip can be kept with one to two orders of magnitude fewer calls?"));
B.push(H2("1.3", "Objectives"));
B.push(UL("**Objective 1:** build a retrieval benchmark on MEVA, a public surveillance dataset that had never been used for natural-language retrieval: a frozen query set and two evaluation protocols."));
B.push(UL("**Objective 2:** adapt a pretrained image–text encoder (SigLIP 2) to fixed surveillance cameras with LoRA, and measure the gain on seen and unseen cameras."));
B.push(UL("**Objective 3:** add a cheap learned re-ranker and decide, with fixed gates, whether and where a VLM is worth its cost; report the quality-versus-cost curve."));
B.push(UL("**Objective 4:** run the pipeline end to end on new video: detected alerts, searchable by text, with the cost of each query shown."));
B.push(H2("1.4", "Scope and Limitations"));
B.push(P("**In scope:** retrieval of annotated activity instances (the unit an alert pipeline produces); 22 activity types with at least 20 test instances; models that run on one 8 GB laptop GPU, plus a 16 GB office GPU for the 8B and 32B VLMs; cost counted as VLM calls and seconds. **Out of scope:** finding the start and end of an event inside long untrimmed video; identifying people (the system retrieves events, not persons, by design); paid APIs, so no token or money cost. **Limitations:** MEVA activities are scripted by actors, so absolute numbers may be optimistic for uncontrolled footage; each selected model was trained with one seed; and no labels exist for footage from outside MEVA, so its retrieval quality is shown, not measured."));
B.push(H2("1.5", "Report Organization"));
B.push(P("Section 2 reviews related work. Section 3 describes the dataset, preprocessing and the two splitting protocols. Section 4 presents the architecture, losses, optimisation and training procedure. Section 5 reports the experiments: baselines, the main comparison, qualitative results and ablations. Section 6 discusses the findings and Section 7 concludes. Appendix A links the code and models and explains how to install and run the system; Appendix B holds additional results."));

// ---------------------------------------------------------------- 2
B.push(H1("2", "Related Work"));
B.push(P(`**Contrastive image–text encoders.** CLIP ${c("clip")} trained an image and a text encoder jointly on 400 million web image–caption pairs, so that images can be classified and searched zero-shot from text, and so that image embeddings can be computed once and indexed. SigLIP ${c("siglip")} replaced the batch-wide softmax with a per-pair sigmoid loss that needs no global normalisation, and SigLIP 2 ${c("siglip2")} added captioning-based pretraining, self-distillation and data curation, improving zero-shot classification and retrieval at every model size. This project uses SigLIP 2 ViT-B/16 as its retriever. All three are trained on web photographs, not fixed, elevated cameras with small, distant people.`));
B.push(P(`**Video moment retrieval and its cost.** Escorcia et al. ${c("escorcia")} posed moment retrieval over a whole video collection and showed that clip-level alignment allows the corpus to be indexed in advance. QVHighlights ${c("qvhighlights")} released free-form queries on 10,000 YouTube videos. Tran et al. ${c("tran")} combined coarse and fine encoders with temporal re-ranking for hour-long video, and Nishimura et al. ${c("superimage")} screened candidates with a cheap model before a VLM, showing that the accuracy–cost trade-off moves with model size and resolution; that is the closest precedent for our cascade. None of these evaluates on surveillance footage.`));
B.push(P(`**Surveillance video and language.** UCA ${c("uca")} annotated 110 hours of UCF-Crime with sentences and found that models strong on web video perform poorly on surveillance (fixed viewpoints, low resolution, small subjects). VadCLIP ${c("vadclip")} adapted CLIP to anomaly detection, a closed set of classes. SurveiLens ${c("surveilens")} captions sampled frames and matches the captions to a query; ForeSea ${c("foresea")} filters footage with a tracker, indexes it with a multimodal embedding and answers with a video LLM over the top clips. Both report accuracy, not the number of expensive calls per query.`));
B.push(P(`**Adaptation and distillation.** LoRA ${c("lora")} trains low-rank updates of frozen weight matrices, matching full fine-tuning with far fewer trained parameters. Knowledge distillation ${c("hinton")} trains a small model on the soft outputs of a large one, and Hofstätter et al. ${c("hofstatter")} distilled a cross-encoder ranker into cheaper students. Cascade ranking ${c("cascade")} orders rankers by cost, each pruning candidates for the next.`));
B.push(P(`**How this work differs.** (1) The surveillance retrieval systems above report accuracy only; we report accuracy *as a function of VLM calls*. (2) MEVA ${c("meva")}, the largest annotated fixed-camera activity corpus with a permissive licence, had not been used for natural-language retrieval. (3) We measure the web-to-CCTV domain shift of a dual encoder on seen and unseen cameras, with and without adaptation. (4) We test the proposed distillation route against a student trained on human labels, and report that it does not help.`));

// ---------------------------------------------------------------- 3
B.push(H1("3", "Dataset and Preprocessing"));
B.push(H2("3.1", "Dataset Description"));
B.push(P(`MEVA (Multiview Extended Video with Activities) ${c("meva")} was recorded by Kitware for the IARPA DIVA programme: fixed indoor and outdoor cameras at a training facility, actors performing scripted everyday activities, released under CC BY 4.0 with human annotations (actor bounding boxes, activity labels, start and end times). It is public, not hosted on Kaggle and not synthetic, and is available at [https://mevadata.org](https://mevadata.org) (videos on AWS Open Data, [https://registry.opendata.aws/mevadata/](https://registry.opendata.aws/mevadata/)). We downloaded the video of 10,678 annotated activity instances (about 70 GB): 32 activity types, 24 cameras at 4 sites. The 22 types with at least 20 test instances are the query activities; the 10 rarer types stay in the corpus as distractors. The distribution is long-tailed (447 instances of *enters scene through structure* against 21 of *vehicle reverses* in the test split of protocol A).`));
B.push(P("There were no text queries, so we wrote them: 3 natural phrasings per query activity (66 in total), drafted from MEVA’s activity definitions, reviewed on training-split examples, and frozen by hash before the first evaluation. A separate bank of 192 training captions is used for fine-tuning; it shares no sentence with the test phrasings, and same-activity caption–phrasing similarity is below 0.80."));
B.push(CAP("Table", "Summary of the dataset used in this project (instances = clips)."));
B.push(TABLE([
  ["Protocol", "Split", "# Samples", "# Classes", "Notes"],
  [{ t: "A (official)", rowSpan: 3 }, "Training", "6,213", "32", "192 training captions"],
  ["Validation", "739", "15", "model selection only"],
  ["Test", "3,726", "32", "22 query activities, 66 phrasings"],
  [{ t: "B (unseen cameras)", rowSpan: 3 }, "Training", "6,154", "32", "16 cameras"],
  ["Validation", "586", "15", "same 16 cameras"],
  ["Test", "3,938", "32", "8 cameras never seen in training"],
  [{ t: "Total", span: 2 }, "10,678", "32", "24 cameras, 4 sites"],
], { widths: [0.22, 0.15, 0.14, 0.13, 0.36], align: ["left", "left", "right", "right", "left"], groups: [4, 7] }));
B.push(AFTER_TABLE());
B.push(H2("3.2", "Preprocessing"));
B.push(UL("**Frame sampling:** each instance is sampled at 2 frames per second, at most 16 frames."));
B.push(UL("**Crops:** from the annotated actor boxes, each frame gets a square *tight* crop around the actors at native resolution, and a *context* crop with surroundings (≤448 px). On validation, tight crops doubled top-10 precision over full frames (0.22 against 0.10), so the tight view was fixed before any test number was read."));
B.push(UL("**Image normalisation:** crops are resized to 224×224 (bilinear) and normalised exactly as the SigLIP 2 processor does; the training pipeline reproduces the processor pixel-for-pixel (tested)."));
B.push(UL("**Augmentation (LoRA training only):** zoom in/out, stretch and resolution degradation. No horizontal flips, because left and right turns are different activities."));
B.push(UL("**Text:** SigLIP 2 tokenizer, padded to 64 tokens, as in its pretraining."));
B.push(UL("**Box motion:** per frame, 12 numbers from the actor boxes (cumulative and per-step displacement, log height and width, aspect ratio, number of actors, distance between actors, time, a has-box flag)."));
B.push(UL("**Class balance:** square-root-balanced sampling of instances during fine-tuning, and every metric is macro-averaged over the 22 activities so frequent classes cannot hide rare-class failures."));
B.push(H2("3.3", "Data Splitting Strategy"));
B.push(P("Two protocols are evaluated. **Protocol A** is MEVA’s official split; its test cameras also appear in training. **Protocol B** holds 8 of the 24 cameras out of training and validation entirely, so the gap between A and B measures how much a model relies on having seen a camera. In both, the split is by recording, not by frame, so no clip contributes frames to two splits. **Selection rule:** every model, epoch, weight and threshold was chosen on the validation split; the test split was read once per result and never used to choose anything."));

// ---------------------------------------------------------------- 4
B.push(H1("4", "Methodology"));
B.push(H2("4.1", "Proposed Architecture"));
B.push(P("Figure 1 shows the system. At index time every clip is embedded once; at query time a sentence costs one text-encoder call, a dot product with every frame, and a pass of the small transformer over the candidates."));
B.push(FIGURE("fig1_architecture.png", 1.0, "Architecture: SigLIP 2 image encoder with LoRA, frame tokens, temporal transformer, text encoder, cosine per frame, fusion and optional VLM on the top k."));
B.push(CAP("Figure", "Overview of the proposed model architecture."));
B.push(CAPNOTE("The three search levels of the dashboard are *Fast* (cosine score only), *Standard* (fusion with the temporal transformer, the main system) and *Verified* (Standard, then the VLM on the top *k*)."));
B.push(P(`**Stage 1: adapted retriever.** SigLIP 2 base (ViT-B/16, 224 px) ${c("siglip2")} embeds every tight crop and the query into the same 768-dimensional space; vectors are ℓ_{2}-normalised, so the dot product is the cosine similarity. A clip’s score is the mean of its three highest frame similarities, which tolerates frames in which the action has not started. LoRA ${c("lora")} adapters of rank 16 are added to the query, key, value and output projections of all 12 attention layers of the image encoder (48 adapters, 1.18 M trained weights, 1.25 % of the encoder). The text encoder stays frozen, so query encoding and general language are unchanged.`));
B.push(P("**Stage 2: temporal transformer (“student”).** A single frame cannot show standing up or picking something up. The re-ranker reads one token per frame (1,548 numbers: the LoRA tight embedding, the frozen context embedding and the 12 box-motion features) and one token for the query (768 numbers). Each token is layer-normalised and projected to *d* = 256; frames get a learned position embedding. A 2-layer pre-norm transformer encoder (4 heads, feed-forward 1,024, dropout 0.1) processes the at most 17 tokens; the output at the query token goes through a linear head to one relevance logit *s*_{tf}. Its score is fused with the retriever’s, *s* = *z*(*s*_{tf}) + β *z*(*s*_{lora}), where *z* standardises over the candidates and β is chosen on validation (β = 1)."));
B.push(P(`**Stage 3: VLM on a short list (optional).** A VLM (Qwen3-VL ${c("qwen3vl")}) scores a clip by the log-odds of “yes” over “no” to “Does this video show the following: {query}?”, given 8 evenly spaced tight frames at 448 px with their timestamps. It re-orders only the top *k* of stage 2 by *z*(VLM) + γ *z*(*s*) (γ = 2, chosen on validation). Two gates fixed in advance decide whether a VLM is used: G0 (fits in memory, deterministic) and G1 (adds at least +0.02 validation mAP when fused into the adapted retriever’s top 50).`));
B.push(H2("4.2", "Loss Function"));
B.push(P("**LoRA: label-aware sigmoid loss.** For a batch of *N* crop embeddings **x**_{i} and *M* caption embeddings **t**_{j}, with the pretrained temperature τ and bias *b* of SigLIP (also trained),", { keepNext: true }));
B.push(EQ([sub(m("ℒ"), up("LoRA")), rnd(m("θ")), m("=−"), frac(m("1"), m("N")),
  sum(m("i=1"), m("N"), sum(m("j=1"), m("M"), bce(sub(m("y"), m("ij")), sub(m("ℓ"), m("ij")))))]));
B.push(P("where ℓ_{ij} = τ **x**_{i}^{⊤}**t**_{j} + *b* is the logit of the pair and *y*_{ij} = 1 when caption *j* describes the activity of crop *i*. It differs from SigLIP’s loss in one point: every caption of the crop’s activity is a positive, not only its own caption."));
B.push(P("**Temporal transformer: binary cross-entropy.** For a (clip, caption) pair *p* with logit *s*_{p}, label *y*_{p} ∈ {0, 1} and, when a teacher is used, the teacher’s VLM margin *m*_{p},", { keepNext: true }));
B.push(EQ([sub(m("ℒ"), up("tf")), m("=−"), frac(m("1"), m("P")),
  sum(m("p=1"), m("P"), bce(sub(m("ỹ"), m("p")), sub(m("s"), m("p"))))]));
B.push(EQ([sub(m("ỹ"), m("p")), m("=α"), sub(m("y"), m("p")), m("+"), rnd(m("1−α")), m("σ"), rnd(sub(m("m"), m("p")))]));
B.push(P("where ỹ_{p} is the training target. α = 1 trains on the human labels (GT), α = 0.5 mixes in the teacher (GT+KD, the distillation route), and α = 0 trains on the teacher alone (KD)."));
B.push(H2("4.3", "Optimization"));
B.push(P("Both trained components use AdamW. **LoRA:** learning rate 10^{−4}, batch 64, linear warm-up over 2 epochs, then the rate is halved after every 2 epochs without a validation gain; at most 100 epochs, early stopping after 5 epochs without a gain (best epochs 10–28; the cap was never reached). **Temporal transformer:** learning rate 3×10^{−4}, batch 256, at most 50 epochs, early stopping after 5 epochs without a gain. The model of the best validation epoch is kept. The validation metric is Q1 mAP on the validation split."));
B.push(H2("4.4", "Regularization and Hyperparameters"));
B.push(CAP("Table", "Hyperparameter configuration."));
B.push(TABLE([
  ["Hyperparameter", "LoRA (image encoder)", "Temporal transformer"],
  ["Optimizer", "AdamW", "AdamW"],
  ["Learning rate", "1×10^{−4}, warm-up 2 epochs", "3×10^{−4}"],
  ["LR schedule", "×0.5 after 2 epochs without gain", "constant"],
  ["Batch size", "64 crops", "256 pairs"],
  ["Epochs", "≤100, early stop patience 5", "≤50, early stop patience 5"],
  ["Weight decay", "0.01 (adapters only)", "0.01"],
  ["Dropout", "0.1 (LoRA dropout)", "0.1"],
  ["Size", "rank 16, α = 32, 48 adapters, 1.18 M", "2 layers, *d* = 256, 4 heads, 2.2 M"],
  ["Training data", "6,213 instances × 192 captions", "≈12,200 pairs per protocol"],
  ["Fusion weight", "–", "β = 1 (grid on validation)"],
], { widths: [0.22, 0.39, 0.39] }));
B.push(AFTER_TABLE());
B.push(P("Other regularisation: square-root class balancing, crop augmentation (Section 3.2), a frozen text encoder, and the restriction of LoRA to the attention projections. The rank, α and the transformer’s size were fixed in the plan before training and not tuned."));
B.push(H2("4.5", "Training Procedure"));
B.push(P("Four LoRA runs were trained (protocol A or B × uniform or middle-weighted frame sampling) and one was selected per protocol on validation. The training pairs of the temporal transformer are, for each activity, its positive clips plus 100 hard negatives (the highest-scoring wrong clips under LoRA) and 100 random negatives."));
B.push(ALGORITHM("Training procedure (LoRA retriever, then temporal transformer)",
  "training clips 𝒟_{train}, captions 𝒯, validation set 𝒟_{val}", [
  [0, "Load SigLIP 2; freeze all weights; insert LoRA adapters θ (rank 16) into the image encoder"],
  [0, "**for** epoch *e* = 1 **to** 100 **do**"],
  [1, "**for** each mini-batch ℬ of crops (square-root class-balanced, augmented) **do**"],
  [2, "**x** ← image encoder_{θ}(ℬ);  **t** ← text encoder(𝒯)"],
  [2, "compute ℒ_{LoRA} (Eq. 1);  θ ← AdamW(θ, ∇_{θ}ℒ)"],
  [1, "**end for**"],
  [1, "evaluate validation Q1 mAP; keep the best θ; halve the LR after 2 epochs, stop after 5, without gain"],
  [0, "**end for**"],
  [0, "Re-embed every clip with the best adapters; build frame tokens (LoRA + context + box motion)"],
  [0, "Build pairs: positives + 100 hard + 100 random negatives per activity"],
  [0, "**for** epoch *e* = 1 **to** 50 **do**"],
  [1, "**for** each mini-batch of 256 (clip, caption) pairs **do**"],
  [2, "*s* ← transformer_{φ}(query token, frame tokens);  φ ← AdamW(φ, ∇_{φ}ℒ_{tf}) (Eq. 2)"],
  [1, "**end for**"],
  [1, "choose β on validation; keep the best (φ, β); stop after 5 epochs without gain"],
  [0, "**end for**"],
  [0, "**return** adapters θ, transformer φ, fusion weight β"],
]));
B.push(AFTER_TABLE());

// ---------------------------------------------------------------- 5
B.push(H1("5", "Experiments and Results"));
B.push(H2("5.1", "Experimental Setup"));
B.push(P("**Hardware:** a laptop with an NVIDIA RTX 4060 Laptop GPU (8 GB) and 32 GB RAM for everything except the 8B and 32B VLMs, which ran on an office server with an RTX 4070 Ti SUPER (16 GB; the 32B with weights streamed between CPU and GPU). **Software:** Python 3.12, PyTorch 2.14 (CUDA 12.6), Hugging Face Transformers 5.20, PEFT 0.21, Ultralytics YOLOv8 8.4. **Runtime:** one LoRA epoch takes about 57 s; all four LoRA runs 1.5 h; a temporal transformer run a few minutes. At query time the Standard level takes about 0.26 s for the 3,726 test clips, of which the transformer is 0.18 s. Appendix A explains how to install and run the system."));
B.push(H2("5.2", "Evaluation Metrics"));
B.push(P("Two query types are evaluated. **Q1, category search:** each of the 66 phrasings ranks the whole test corpus. **Q2, forensic search:** “find it on camera *C* in hour *H*”, one query per annotated event and phrasing (1,140 per protocol), ranked inside that camera-hour (median 56 candidates). For a ranking with relevant set *R*,", { keepNext: true }));
B.push(EQ([up("AP"), m("="), frac(m("1"), m("|R|")), sum(m("k=1"), m("n"), [m("P@k⋅"), up("rel"), rnd(m("k"))]),
  m(",   "), up("Success@"), m("K="), m("𝟙"), sq([m("∃ k≤K: "), up("rel"), rnd(m("k")), m("=1")])]));
B.push(EQ([up("RR"), m("="), frac(m("1"), up("rank of the first relevant clip")), m(",   "), up("MRR"), m("="), up("mean of RR over queries")]));
B.push(P("with *P*@*k* the precision of the top *k*. mAP is the mean AP over queries; P@10 and Recall@10 are also reported. All metrics are *macro-averaged over the 22 activities*. 95 % confidence intervals come from a paired bootstrap over activities (10,000 resamples). Cost is the number of VLM calls and the seconds per query."));
B.push(H2("5.3", "Quantitative Results"));
B.push(CAP("Table", "Comparison of models on the test set."));
const MAIN = [
  ["A", "Zero-shot CLIP ViT-B/16 (baseline)", "0", "0.142", "0.189", "0.237", "0.558", "0.391"],
  ["A", "Zero-shot SigLIP 2, full frame", "0", "0.096", "0.127", "–", "–", "0.267"],
  ["A", "Zero-shot SigLIP 2, tight crops", "0", "0.179", "0.259", "0.273", "0.655", "0.447"],
  ["A", "+ LoRA", "0", "0.313", "0.408", "0.389", "0.797", "0.558"],
  ["A", "+ LoRA + temporal transformer (proposed)", "0", "0.476", "0.527", "0.531", "0.900", "0.690", true],
  ["A", " + 8B VLM on top 10", "9.8", "0.477", "0.527", "0.589", "0.914", "0.728"],
  ["A", " + 32B VLM on top 10", "9.8", "0.478", "0.527", "0.597", "0.912", "0.732"],
  ["A", "2B VLM on every candidate", "124.5", "0.207", "0.312", "0.360", "0.729", "0.523"],
  ["B", "Zero-shot CLIP ViT-B/16 (baseline)", "0", "0.138", "0.192", "0.261", "0.584", "0.409"],
  ["B", "Zero-shot SigLIP 2, tight crops", "0", "0.174", "0.274", "0.275", "0.703", "0.460"],
  ["B", "+ LoRA", "0", "0.280", "0.362", "0.369", "0.723", "0.528"],
  ["B", "+ LoRA + temporal transformer (proposed)", "0", "0.442", "0.521", "0.519", "0.848", "0.663", true],
  ["B", " + 8B VLM on top 10", "9.7", "0.443", "0.521", "0.554", "0.859", "0.692"],
  ["B", " + 32B VLM on top 10", "9.7", "0.444", "0.521", "0.564", "0.867", "0.692"],
  ["B", "2B VLM on every candidate", "307.8", "0.212", "0.312", "0.367", "0.729", "0.531"],
].map((r) => r.slice(0, 8).map((x) => ({ t: x, bold: r[8] })));
B.push(TABLE([
  ["", "", "VLM", { t: "Q1", span: 2, rule: true }, { t: "Q2", span: 3, rule: true }],
  ["Prot.", "Model", "calls", "mAP", "P@10", "S@1", "S@5", "MRR"],
  ...MAIN,
], { widths: [0.07, 0.41, 0.08, 0.088, 0.088, 0.088, 0.088, 0.088], header: 2, size: 18,
  align: ["left", "left", "right", "center", "center", "center", "center", "center"], groups: [10] }));
B.push(TNOTE("Q1 = category search, Q2 = forensic search; VLM calls are per Q2 query."));
B.push(P("Random ordering gives Q1 mAP 0.047. Every stage improves every metric on both protocols:", { keepNext: true }));
B.push(UL("**LoRA:** Q1 mAP +0.134 [+0.085, +0.194] on A and +0.107 [+0.062, +0.158] on B; Q2 Success@1 +0.115 and +0.094. About 80 % of the gain carries over to cameras never seen in training."));
B.push(UL("**Temporal transformer:** Success@1 +0.143 [+0.058, +0.228] on A and +0.150 [+0.082, +0.220] on B, at zero VLM calls; Q1 mAP +0.163 and +0.162. This is the largest single step."));
B.push(UL("**VLMs:** an 8B or 32B VLM on the top 10 adds +0.058 to +0.065 Success@1 on A (32B: [+0.040, +0.095]); on B only the 32B gain is significant (+0.045 [+0.018, +0.073]). The 32B is not significantly better than the 8B (+0.007 [−0.020, +0.034]) at 13× the time per call (1.74 s against 0.13 s)."));
B.push(UL("**VLM on everything:** the 2B VLM on every candidate is *worse* than LoRA alone (Q1 mAP 0.207 against 0.313) and takes about 10 minutes per Q1 query against 0.04 s."));
B.push(UL("**CLIP vs SigLIP 2:** CLIP has a higher validation AUC (0.726 against 0.697, entirely from one activity, *person stands up*) but is below SigLIP 2 on every test metric of both protocols."));
B.push(FIGURE("fig2_cost_curve.png", 0.82, "Quality-versus-cost curve: forensic Success@1 against VLM calls per query."));
B.push(CAP("Figure", "Quality-versus-cost curve (Success@1 against VLM calls)."));
B.push(CAPNOTE("Forensic Success@1 against VLM calls per query. The temporal transformer at 0 calls captures most of the gain; larger VLMs add to it on a short list; the 2B VLM on every candidate is below LoRA alone."));
B.push(H2("5.4", "Qualitative Results"));
B.push(P("Figure 3 follows one test clip through the system for the query “a person picking up an object”. The frozen encoder ranks the correct clip 286th of 3,726; LoRA moves it to 126th; the temporal transformer, which sees the person bend down and then straighten up with the item, ranks it first. Figure 4 shows the same effect over all activities: the largest gains of the transformer are on activities defined by motion."));
{
  const wImg = Math.round(TEXT_W * 0.25), wTxt = TEXT_W - 2 * wImg;
  const imgCell = (name) => new TableCell({ width: { size: wImg, type: WidthType.DXA }, borders: { top: NONE, bottom: NONE, left: NONE, right: NONE },
    verticalAlign: VerticalAlign.CENTER, margins: { left: 40, right: 40 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [image(name, { h: 120, alt: "MEVA test clip 9850, a person picking up an object" })] })] });
  const rank = new TableCell({ width: { size: wTxt, type: WidthType.DXA }, borders: { top: NONE, bottom: NONE, left: NONE, right: NONE },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
      children: [...runs("**frozen**  rank 286   →   **+ LoRA**  rank 126   →   ", { size: 22 }), ...runs("**+ transformer**  rank 1", { size: 22, color: "B35900" })] })] });
  B.push(new Table({ columnWidths: [wImg, wImg, wTxt], width: { size: TEXT_W, type: WidthType.DXA }, layout: TableLayoutType.FIXED,
    borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
    rows: [new TableRow({ cantSplit: true, children: [imgCell("arch_frame_0.png"), imgCell("arch_frame_1.png"), rank] })] }));
}
B.push(CAP("Figure", "Sample output: one test clip through the three stages."));
B.push(CAPNOTE("Two frames (1 s apart) of MEVA test clip 9850, “a person picking up an object”, and its rank among 3,726 test clips at each stage."));
B.push(FIGURE("fig3_per_activity.png", 1.0, "Average precision per activity on protocol A for frozen, LoRA and LoRA plus temporal transformer."));
B.push(CAP("Figure", "Average precision per activity (protocol A)."));
B.push(CAPNOTE("Frozen → LoRA → LoRA + temporal transformer, sorted by the frozen score."));
B.push(P("Notable per-activity changes on A: *person talks to person* AP 0.238 → 0.744 (LoRA) → 0.932 (transformer); the motion-defined activities (pick up, put down, stand up, transfer an object) from 0.19–0.21 to 0.56–0.57. The search dashboard (Figure 5) answers “a person opening a car door” with 6 correct clips in the top 10 at the Standard level in 0.6 s, against 4 at the Fast level."));
B.push(FIGURE("fig4a_search.png", 0.95, "The search dashboard showing results for a test phrasing at the Standard level."));
B.push(CAP("Figure", "The search dashboard."));
B.push(CAPNOTE("A test phrasing at the Standard level, correct results marked, with the cost of the query (0 VLM calls) against a VLM on every clip."));
B.push(H2("5.5", "Ablation Study"));
B.push(CAP("Table", "Ablation study."));
B.push(TABLE([
  ["Change", "Result", "Conclusion"],
  ["Full frame instead of tight crops (frozen, test A)", "Q1 mAP 0.096 vs 0.179", "crop to the actors"],
  ["LoRA frame sampling: middle-weighted vs uniform (test)", "A 0.313 vs 0.315; B 0.275 vs 0.280", "no difference"],
  ["Transformer alone, no fusion with LoRA (test A)", "S@1 0.460 vs 0.531", "the two scores are complementary"],
  ["Transformer target GT+KD, 2B teacher (val A / B)", "0.594 vs 0.600 / 0.539 vs 0.566", "distillation does not help"],
  ["Transformer target KD only, 2B teacher (val A / B)", "0.424 / 0.377", "teacher weaker than labels"],
  ["Same with the 32B teacher, GT / GT+KD / KD (val A)", "0.606 / 0.573 / 0.447", "still no gain"],
  ["VLM with 16 frames instead of 8 (G1 gain, val)", "+0.006 / +0.009", "more frames do not help"],
  ["LoRA on image + text towers, 2 seeds (val A / B)", "−0.019 / +0.047", "not adopted (rule + drift check)"],
  ["2B VLM on LoRA’s top 10, no transformer (test A)", "S@1 0.421 vs 0.389", "small; the transformer is worth more"],
], { widths: [0.5, 0.27, 0.23], size: 19 }));
B.push(TNOTE("Validation (val) numbers were used for the choices; test numbers are reported once."));
B.push(P("Seven VLMs (Qwen3-VL 2B, 2B-Reranker, 8B and 32B; Qwen2.5-VL 3B and 7B; Gemma 4 E4B) were tested with the G1 gate (Table 6 in Appendix B). Only the 32B passed. Inside the retriever’s own top 50, the VLMs separate right from wrong clips with an AUC of 0.54–0.66, below the adapted retriever’s 0.67, which is why they add little."));

// ---------------------------------------------------------------- 6
B.push(H1("6", "Discussion"));
B.push(P("**Why it works.** The frozen encoder fails for two reasons, and each trained component fixes one. *Domain:* SigLIP 2 was trained on web photographs; CCTV shows small, distant, often low-resolution people from above. A small LoRA on 192 captions moves the image embeddings towards that domain (+75 % Q1 mAP), and since the same activities improve on unseen cameras, this is not camera memorisation. *Time:* per-frame embeddings carry no motion, so the frozen and adapted encoders are near chance on standing up or picking up. The temporal transformer sees the frames in order together with how the boxes move, and these activities gain the most."));
B.push(P("**Why the VLMs disappointed.** The proposal expected a VLM teacher to be the strongest ranker, to be distilled into a cheap student. Measured on our data, a 2B VLM is a weaker ranker than the adapted encoder: it sees 8 small frames and answers a yes/no question, while the encoder was adapted to this footage. A teacher that is less accurate than the labels cannot teach the student more than the labels do, so distillation never helped. The large VLMs are useful in a different role: as a final check on a short, already good list."));
B.push(P("**Failure modes.** (1) Left versus right turns: no component separates them (AUC 0.49–0.51), because the box features are image-plane displacements and heading relative to the camera is not encoded; a merged “vehicle turns” class is found well (AP 0.57–0.62). (2) Footage unlike MEVA (indoor, close-up) is found but rates low relative to MEVA scores. (3) The best system still misses the right clip at rank 1 for about four forensic queries in ten, so it is an aid for an investigator, not an automatic decision."));
B.push(P("**What I learned.** Measuring before building changes the plan: fixing gates and selection rules in advance turned a failed hypothesis (distillation) into a clear negative result instead of a tuned-up positive one. Cheap, domain-specific components beat a large general model used everywhere, and cost has to be measured as carefully as accuracy."));

// ---------------------------------------------------------------- 7
B.push(H1("7", "Conclusion and Future Work"));
B.push(H2("7.1", "Conclusion"));
B.push(P("We built natural-language search over surveillance video on MEVA and measured its quality as a function of VLM calls. Three findings: (1) adapting the image encoder with LoRA (1.18 M weights, 192 captions) is the cheapest large gain and carries over to unseen cameras; (2) a small temporal transformer trained on the human labels is the largest single step, reaching Q1 mAP 0.476 and Success@1 0.53 at zero VLM calls, while VLM distillation did not help; (3) VLMs earn their place only at the end, as a check on the top 10, where an 8B or 32B model adds +0.06 Success@1. The resulting system answers a query over 3,726 clips in about a quarter of a second on a laptop GPU, against about 10 minutes for a 2B VLM on every clip, and it runs end to end on new video as a dashboard."));
B.push(H2("7.2", "Future Work"));
B.push(UL("Heading-relative motion features (or a calibrated ground plane) to separate left and right turns."));
B.push(UL("A retention objective so that adapting the text encoder keeps its gain on unseen cameras (+0.047) without drifting on general language."));
B.push(UL("Several training seeds for the selected models, and a prompt and frame-layout study for the VLM gate."));
B.push(UL("Labelled footage from outside MEVA, to measure retrieval on real (unscripted) CCTV and calibrate the dashboard’s confidence bands."));
B.push(UL("Temporal localisation inside long videos, and a quantised student for edge deployment."));

// ---------------------------------------------------------------- references
B.push(H1("", "References"));
REFS.forEach(([, text], i) => B.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 80, line: 252 },
  tabStops: [{ type: TabStopType.LEFT, position: 567 }], indent: { left: 567, hanging: 567 },
  children: [new TextRun({ text: `[${i + 1}]`, font: FONT, size: 22 }), new TextRun({ children: [new Tab()] }), ...runs(text, { size: 22 })] })));

// ---------------------------------------------------------------- appendix A
B.push(H1("A", "Code, Models and How to Run the System"));
B.push(H2("A.1", "Shared Folder"));
B.push(P(`The complete code and the trained models are shared here: ${INFO.link}. The MEVA dataset is not redistributed (Appendix A.5).`));
B.push(CAP("Table", "Contents of the shared folder."));
B.push(TABLE([
  ["Folder", "Contents", "Size"],
  ["`code/`", "Python package `sfs` (training, evaluation, dashboard), built web page, run scripts, Docker files", "< 0.1 GB"],
  ["`models/siglip2/`", "SigLIP 2 base (image and text encoder), unchanged", "1.4 GB"],
  ["`models/lora_adapter/`", "**trained** LoRA adapter of the image encoder (1.18 M weights)", "< 0.1 GB"],
  ["`models/index/`", "**trained** temporal transformer (`student.pt`); embeddings of the 3,726 MEVA test clips", "0.3 GB"],
  ["`models/yolo/`", "YOLOv8m detector for uploaded videos", "< 0.1 GB"],
  ["`models/qwen3-vl-2b/`", "Qwen3-VL-2B, used only by the Verified level", "4.0 GB"],
  ["`test_videos/`, `my_alerts/`", "three CCTV test clips, and the same clips already processed (123 alerts)", "0.9 GB"],
], { widths: [0.27, 0.6, 0.13], align: ["left", "left", "right"] }));
B.push(AFTER_TABLE());
B.push(H2("A.2", "Requirements"));
B.push(UL("Python 3.12 (the package versions are pinned for it)."));
B.push(UL("Recommended: an NVIDIA GPU with at least 8 GB and a driver that supports CUDA 12.6 (version 560 or newer); tested on an RTX 4060 Laptop GPU under Windows 11. Without a GPU the system runs on the processor, more slowly and without the Verified level."));
B.push(UL("About 6 GB of disk for the Python packages. No internet connection is needed at run time."));
B.push(H2("A.3", "Installation"));
B.push(P("Open a terminal in the `code` folder and create an environment (Windows shown; on Linux use `python3.12` and `. .venv/bin/activate`):", { keepNext: true }));
B.push(CODE(`
py -3.12 -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt        (without a GPU: requirements-cpu.txt)
`));
B.push(AFTER_TABLE());
B.push(H2("A.4", "Running the Dashboard"));
B.push(OL("In the `code` folder, with the environment active, run `run.bat` (Windows) or `./run.sh` (Linux). The message “MEVA media not found …” is expected: the dataset is not in the folder."));
B.push(OL("Wait for `open http://127.0.0.1:8000` (about a minute while the models load) and open that address in a browser. Stop the server with Ctrl+C; another port: `run.bat --port 8010`."));
B.push(OL("**Detect tab:** upload a video (one of the clips in `test_videos/` or any MP4, MOV, AVI, MKV or WebM). YOLOv8 finds people, vehicles and backpacks; each object present for 0.5 s becomes an alert with a 5-second clip, embedded with the trained models. Processing takes about four times the video’s length on a laptop GPU. Alerts are kept in `my_alerts/`; the three test clips are already processed."));
B.push(OL("**Search tab:** type what happened, e.g. “a person walking past a parked car”, and choose a level: *Fast* (LoRA-tuned SigLIP 2 only), *Standard* (with the temporal transformer; the main result) or *Verified* (Standard, then the VLM on the top *k*). The bar under the search shows the time of each stage and what a VLM on every clip would have cost; *Map* shows every clip in 3D."));
B.push(P("**Docker (NVIDIA GPU):** in `code`, create an empty folder `../meva` and run `docker compose up`; the first run builds a 12 GB image, then the dashboard is at `http://127.0.0.1:8000`. The folder’s `README.md` has the full guide and a troubleshooting table.", { before: 120 }));
B.push(H2("A.5", "Dataset and Reproducing the Experiments"));
B.push(P("MEVA is public (CC BY 4.0) at [https://mevadata.org](https://mevadata.org); the videos are on AWS Open Data ([https://registry.opendata.aws/mevadata/](https://registry.opendata.aws/mevadata/)) and the annotations in the MEVA data repository linked from the site. The experiments run as modules of the `sfs` package, in this order: `sfs.data.manifest` (clip list, protocols A and B) → `sfs.data.extract_crops` (crop store) → `sfs.embed.run` (frozen embeddings) → `sfs.train.run` (LoRA) → `sfs.embed.reembed` → `sfs.rerank.features` and `sfs.rerank.train` (temporal transformer) → `sfs.teacher.run` and `sfs.teacher.gate` (VLMs) → `sfs.eval.baseline` and `sfs.eval.cascade_run` (results). With the derived MEVA folders `meva/crops` and `meva/previews` next to `code`, the dashboard also searches the 3,726 MEVA test clips."));
B.push(H2("A.6", "Code Listings"));
B.push(P("The two trained models are defined as follows.", { keepNext: true }));
B.push(CAP("Listing", "Temporal transformer (re-ranker), sfs/rerank/model.py."));
B.push(CODE(String.raw`
class TemporalReranker(nn.Module):
    def __init__(self, frame_dim, text_dim=768, d=256, layers=2, heads=4, t_max=16, dropout=0.1):
        super().__init__()
        self.frame_in = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, d))
        self.query_in = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
        self.pos = nn.Parameter(torch.zeros(t_max, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, frames, mask, text):
        """frames (B, T, 1548), mask (B, T) True = real frame, text (B, 768) -> logits (B,)"""
        x = self.frame_in(frames) + self.pos[:frames.shape[1]]
        q = self.query_in(text).unsqueeze(1)
        seq = torch.cat([q, x], dim=1)                       # query token first
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], dim=1)
        out = self.encoder(seq, src_key_padding_mask=pad)
        return self.head(out[:, 0]).squeeze(-1)              # read out at the query token
`));
B.push(AFTER_TABLE());
B.push(CAP("Listing", "Label-aware sigmoid loss for the LoRA training, sfs/train/loss.py (Eq. 1)."));
B.push(CODE(String.raw`
def label_aware_sigmoid_loss(img_emb, cap_emb, labels, logit_scale, logit_bias):
    """img_emb (N, 768), cap_emb (M, 768), labels (N, M): 1 if caption j describes crop i's activity."""
    logits = img_emb @ cap_emb.T * logit_scale.exp() + logit_bias
    target = labels.float()
    return F.binary_cross_entropy_with_logits(logits, target, reduction="sum") / img_emb.shape[0]
`));
B.push(AFTER_TABLE());

// ---------------------------------------------------------------- appendix B
B.push(H1("B", "Additional Results"));
B.push(CAP("Table", "VLMs tested as re-rankers (gate G1)."));
B.push(TABLE([
  ["VLM", "Params", "VRAM (GB)", "s / call", "G1 gain A / B", "Verdict"],
  ["Qwen3-VL-2B-Instruct", "2B", "6.1", "0.17", "+0.009 / +0.002", "fail"],
  ["Qwen3-VL-Reranker-2B", "2B", "4.4", "0.16", "+0.003 / +0.012", "fail (2B teacher)"],
  ["Qwen3-VL-2B, 16 frames", "2B", "–", "–", "+0.006 / +0.009", "fail"],
  ["Gemma 4 E4B-it", "4B eff.", "7.8", "0.16", "−0.014 / +0.011", "fail"],
  ["Qwen2.5-VL-3B-Instruct", "3B", "8.5", "0.12", "+0.001 / +0.004", "fail"],
  ["Qwen2.5-VL-7B-Instruct (FP8)", "7B", "11.4", "0.19", "+0.001 / +0.014", "fail"],
  ["Qwen3-VL-8B-Instruct (FP8)", "8B", "13.0", "0.13", "+0.007 / +0.019", "fail by 0.001"],
  ["Qwen3-VL-32B-Instruct (FP8)", "32B", "12.3", "1.74", "+0.017 / +0.037", "**pass**"],
], { widths: [0.33, 0.1, 0.12, 0.1, 0.18, 0.17], size: 19, align: ["left", "right", "right", "right", "center", "center"] }));
B.push(TNOTE("Re-ranking the adapted retriever’s top 50 on validation. G1 gain = validation Q1 mAP added; pass bar +0.02 on one protocol."));
B.push(CAP("Table", "Hypotheses fixed before the results, and the verdicts."));
B.push(TABLE([
  ["Hypothesis", "Verdict", "Basis"],
  ["LoRA adaptation beats the frozen encoder", "Confirmed", "Q1 mAP 0.179 → 0.313 (A); all CIs exclude zero"],
  ["A distilled student keeps most of the teacher’s quality", "Rejected", "KD ≪ GT; GT+KD ≈ GT (2B and 32B teachers)"],
  ["The accuracy-vs-cost curve is strongly concave", "Partly", "concave after the student (knee at 5–10 calls); flat for the exhaustive 2B VLM"],
  ["The temporal transformer beats LoRA, most on motion activities", "Confirmed", "motion AP +0.36 vs +0.13 for the rest"],
  ["Cascade reaches ≥90 % of the exhaustive VLM at ≤10 % of the calls", "Confirmed, trivially", "152 % at 4 % of the calls (A)"],
], { widths: [0.45, 0.15, 0.4] }));
B.push(AFTER_TABLE());
{
  const half = Math.round(TEXT_W / 2);
  const cellImg = (name, alt) => new TableCell({ width: { size: half, type: WidthType.DXA }, borders: { top: NONE, bottom: NONE, left: NONE, right: NONE },
    margins: { left: 40, right: 40 }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [image(name, { w: PX(half) - 10, alt })] })] });
  B.push(new Table({ columnWidths: [half, TEXT_W - half], width: { size: TEXT_W, type: WidthType.DXA }, layout: TableLayoutType.FIXED,
    borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
    rows: [new TableRow({ cantSplit: true, children: [cellImg("fig4b_detect.png", "Detect tab: alerts from an uploaded CCTV clip"), cellImg("fig4c_map.png", "3D map of clip embeddings with query results lit up")] })] }));
}
B.push(CAP("Figure", "Dashboard on new video: alerts and the clip map."));
B.push(CAPNOTE("Alerts detected in an uploaded CCTV clip (left); the 3D map of clip embeddings with the top results of a query lit up (right)."));
B.push(P(`**Alerts from new video.** YOLOv8m ${c("yolo")} with ByteTrack ${c("bytetrack")} detects people, vehicles and backpacks; an object seen for 0.5 s with confidence ≥0.5 raises one alert, held until it has been gone for 2 s. Each alert keeps a 5-second clip (11 frames at 2 fps) that goes through the same crops, encoders and transformer as the MEVA clips, so alerts and MEVA clips share one score scale. A 2.5-minute garage CCTV clip produced 22 alerts.`));

// ================================================================== assemble
const headerLine = { bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000", space: 1 } };
const HEAD_TXT = "Deep Learning Project Report";
const emptyHF = { headers: { default: new Header({ children: [new Paragraph({ children: [] })] }) },
  footers: { default: new Footer({ children: [new Paragraph({ children: [] })] }) } };
const pageFooter = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 22 })] })] });
const page = { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN, header: 708, footer: 708 } };

const document = new Document({
  creator: INFO.student, title: INFO.title, description: "Deep Learning project report (course template)",
  features: { updateFields: true },
  styles: {
    default: {
      document: { run: { font: FONT, size: BODY }, paragraph: { spacing: { after: 120, line: 276 } } },
      heading1: { run: { font: FONT, size: 32, bold: true, color: "000000" }, paragraph: { spacing: { before: 360, after: 180 }, keepNext: true, keepLines: true } },
      heading2: { run: { font: FONT, size: 27, bold: true, color: "000000" }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, keepLines: true } },
    },
    paragraphStyles: [
      { id: "Caption", name: "Caption", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 22, italics: false, color: "000000" }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 } } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 284 } } } }] },
    { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] },
  ] },
  sections: [
    { properties: { page }, ...emptyHF, children: titlePage },
    { properties: { page: { ...page, pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } },
      headers: { default: new Header({ children: [new Paragraph({ border: headerLine, children: [new TextRun({ text: HEAD_TXT, font: FONT, size: 20, italics: true })] })] }) },
      footers: { default: pageFooter }, children: front },
    { properties: { page: { ...page, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } },
      headers: { default: new Header({ children: [new Paragraph({ border: headerLine, tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W }],
        children: [new TextRun({ text: HEAD_TXT, font: FONT, size: 20, italics: true }), new TextRun({ children: [new Tab()], size: 20 }),
          new SimpleField('STYLEREF "Heading 1"', "")] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 22 })] })] }) },
      children: B },
  ],
});

// ------------------------------------------------------------------ write, then set the marked math runs upright
(async () => {
  const buf = await Packer.toBuffer(document);
  const zip = await JSZip.loadAsync(buf);
  let xml = await zip.file("word/document.xml").async("string");
  const before = (xml.match(/⁣/g) || []).length;
  // <m:r> ... <m:t ...>⁣text  ->  <m:r><m:rPr><m:sty m:val="p"/></m:rPr> ... <m:t ...>text
  xml = xml.replace(/<m:r>((?:(?!<\/m:r>).)*?)(<m:t(?: [^>]*)?>)⁣/g, '<m:r><m:rPr><m:sty m:val="p"/></m:rPr>$1$2');
  const left = (xml.match(/⁣/g) || []).length;
  if (left) throw new Error(`${left} of ${before} upright math runs not converted`);
  zip.file("word/document.xml", xml);
  const out = await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });
  fs.writeFileSync(OUT, out);
  console.log("wrote", OUT, out.length, "bytes;", before, "upright math runs;", EQN, "equations");
})().catch((e) => { console.error(e); process.exit(1); });
