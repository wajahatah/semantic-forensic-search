// Shared parts of the presentation builders (scripts/deck_build.js: 12 slides; scripts/deck6_build.js: 6 slides):
// the control-room theme, the two slide layouts, the drawing helpers and the accessors into report/numbers.json.
// Every number on a slide is read from report/numbers.json through these accessors; none is typed by hand.
"use strict";
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "..");
const pptxgen = require(require.resolve("pptxgenjs", { paths: [path.join(ROOT, "report", "build")] }));

const N = JSON.parse(fs.readFileSync(path.join(ROOT, "report", "numbers.json"), "utf8"));
const FIG = (n) => path.join(ROOT, "report", "figures", n);

// ---------------------------------------------------------------- numbers
const f = (x, d = 3) => Number(x).toFixed(d);
const f2 = (x) => f(x, 2);
const base = (P, sys, q, m) => N.baseline[P].systems[sys][q][m];
const lora = (run, q, m) => N.lora.runs[run][q][m];
const cas = (blk, P, q, sys, m) => N.cascade[blk].summary[`${P}/${q}`][sys][m];
const g1 = (key, cfg, P) => N.g1[key].configs[cfg][P].gain;
const g0 = (key, m) => N.g0[key][m];
const D = N.dataset;
const bsd = (blk, key) => N.bootstrap[blk][key];
const pa = (act, P) => N.lora.per_activity.find((r) => r.protocol === P && r.activity === act);
const sg = (x) => (x >= 0 ? "+" : "") + f(x);
const num = { N, D, f, f2, base, lora, cas, g1, g0, bsd, pa, sg };

// ---------------------------------------------------------------- theme (control-room look of the dashboard)
const THEME = {
  name: "SFS Control Room", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: "E6EDF3", lt1: "0B0F14", dk2: "9AA7B5", lt2: "121821", accent1: "22D3EE", accent2: "F59E0B",
            accent3: "34D399", accent4: "A78BFA", accent5: "F87171", accent6: "60A5FA", hlink: "22D3EE", folHlink: "A78BFA" },
};
const H = THEME.colors;

// A new presentation with the two layouts (TITLE, CONTENT) and helpers bound to it.
function createDeck(title) {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";                                 // 10 x 5.625 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.author = "Wajahat Ahmed";
  pres.title = title;
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "TITLE", background: { color: H.lt1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.35, w: 8.6, h: 1.5, fontSize: 32, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 1.0, y: 2.95, w: 8.0, h: 0.8, fontSize: 16, color: C.text2, align: "center", valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: H.lt1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 9.0, h: 0.68, fontSize: 22, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
      { text: { text: "Semantic Forensic Search · Wajahat Ahmed · CS-5138 Deep Learning", options: { x: 0.5, y: 5.22, w: 7, h: 0.28, fontSize: 9, color: C.text2, margin: 0 } } },
    ],
    slideNumber: { x: 9.0, y: 5.22, w: 0.5, h: 0.28, fontSize: 9, color: H.dk2, align: "right" },
  });

  // ---------------------------------------------------------------- helpers (positions in inches)
  let nObj = 0;
  const nm = (s) => `${s} ${++nObj}`;
  function card(slide, x, y, w, h, head, body, o = {}) {
    slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: C.background2 }, line: { color: "1F2A37", width: 0.75 }, rectRadius: 0.08, objectName: nm("card") });
    slide.addText(head, { x: x + 0.18, y: y + 0.1, w: w - 0.36, h: 0.36, fontSize: o.headSize || 14, bold: true, color: o.headColor || C.accent1, margin: 0, isTextBox: true, valign: "middle", objectName: nm("card head") });
    slide.addText(body, { x: x + 0.18, y: y + 0.5, w: w - 0.36, h: h - 0.6, fontSize: o.bodySize || 12.5, color: C.text1, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 4, objectName: nm("card body") });
  }
  function stat(slide, x, y, w, big, label, o = {}) {
    slide.addText(big, { x, y, w, h: o.bigH || 0.62, fontSize: o.bigSize || 30, bold: true, color: o.color || C.accent1, margin: 0, isTextBox: true, valign: "bottom", objectName: nm("stat") });
    slide.addText(label, { x, y: y + (o.bigH || 0.62) + 0.04, w, h: o.labelH || 0.5, fontSize: o.labelSize || 12, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: nm("stat label") });
  }
  function pill(slide, x, y, w, text, color = C.accent1, o = {}) {
    slide.addShape(pres.ShapeType.roundRect, { x, y, w, h: o.h || 0.4, fill: { color: C.background1 }, line: { color, width: 1 }, rectRadius: 0.2, objectName: nm("pill") });
    slide.addText(text, { x, y, w, h: o.h || 0.4, fontSize: o.size || 12, bold: true, color, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: nm("pill text") });
  }
  const bullets = (items, size = 13) => items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, fontSize: size, color: H.dk1, paraSpaceAfter: 5 } }));
  // A fresh object on every call: pptxgenjs converts option objects in place.
  const chartFrame = () => ({
    chartArea: { fill: { color: H.lt1 } }, plotArea: { fill: { color: H.lt2 } },
    catAxisLabelColor: H.dk2, valAxisLabelColor: H.dk2, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
    valGridLine: { color: "1F2A37", size: 0.5 }, catGridLine: { style: "none" }, catAxisLineColor: "1F2A37", valAxisLineColor: "1F2A37",
    showTitle: true, titleColor: H.dk1, titleFontSize: 12, legendColor: H.dk1, legendFontSize: 10,
  });
  const pngSize = (b) => ({ w: b.readUInt32BE(16), h: b.readUInt32BE(20) });
  function img(slide, name, x, y, w, alt) {
    const s = pngSize(fs.readFileSync(FIG(name)));
    const h = w * s.h / s.w;
    slide.addImage({ path: FIG(name), x, y, w, h, altText: alt, objectName: nm("image") });
    return h;
  }

  // Write the file, then (when PPTX_SKILL_DIR names the folder holding scripts/apply_theme.js) the theme colours.
  async function write(out) {
    await pres.writeFile({ fileName: out });
    const skill = process.env.PPTX_SKILL_DIR;
    if (skill) {
      const { applyTheme } = require(path.join(skill, "scripts", "apply_theme.js"));
      await applyTheme(out, THEME);
    }
    console.log("wrote", out, fs.statSync(out).size, "bytes");
  }

  return { pres, C, nm, card, stat, pill, bullets, chartFrame, img, write };
}

module.exports = { ROOT, N, FIG, THEME, H, num, createDeck };
