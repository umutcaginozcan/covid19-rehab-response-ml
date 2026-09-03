// Zoo scoreboard -> Word table (Table S3)
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, VerticalAlign, VerticalMergeType, PageOrientation,
} = require('docx');

const RES = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'results', 'metrics', 'zoo_results.json'), 'utf8'));
const OUTDIR = process.env.VO2_OUT || path.join(__dirname, '..', 'results', 'tables');

const FONT = 'Times New Roman';
const SZ = 16;
const noBorder = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
const hLine = { style: BorderStyle.SINGLE, size: 6, color: '000000' };
const softLine = { style: BorderStyle.SINGLE, size: 4, color: 'C8C8C8' };

function cell(children, { w, top, bottom, soft, vMerge } = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    verticalMerge: vMerge,
    borders: {
      top: top ? hLine : noBorder,
      bottom: bottom ? hLine : (soft ? softLine : noBorder),
      left: noBorder, right: noBorder,
    },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 26, bottom: 26, left: 50, right: 50 },
    children,
  });
}
const para = (runs, align) => new Paragraph({ alignment: align || AlignmentType.CENTER, spacing: { before: 0, after: 0 }, children: runs });
const run = (text, opts = {}) => new TextRun({ text, font: FONT, size: SZ, ...opts });
const fn = (runs) => new Paragraph({ spacing: { before: 40, after: 0 }, children: runs });
const it = (t) => new TextRun({ text: t, font: FONT, size: SZ, italics: true });

function buildDoc(titleNo, titleRest, metricCols, primaryKey, winnerOf, rowsData, decimals, footTexts, out, metW) {
  // factorial zoo scoreboard table
  const W_FAM = 900, W_MODEL = 1850, W_FEAT = 1950;
  const W_MET = metW || Math.floor((14400 - W_FAM - W_MODEL - W_FEAT) / metricCols.length);
  const COLW = [W_FAM, W_MODEL, W_FEAT, ...metricCols.map(() => W_MET)];
  const TOTAL = COLW.reduce((a, b) => a + b, 0);

  const winner = winnerOf(rowsData);
  const rows = [];
  rows.push(new TableRow({
    tableHeader: true,
    children: [
      cell([para([run('Family', { bold: true })], AlignmentType.LEFT)], { w: W_FAM, top: true, bottom: true }),
      cell([para([run('Model', { bold: true })], AlignmentType.LEFT)], { w: W_MODEL, top: true, bottom: true }),
      cell([para([run('Feature set', { bold: true })])], { w: W_FEAT, top: true, bottom: true }),
      ...metricCols.map(([label, key]) => cell(
        [para([run(label + (key === primaryKey ? '†' : ''), { bold: true })])],
        { w: W_MET, top: true, bottom: true })),
    ],
  }));
  const last = rowsData.length - 1;
  rowsData.forEach((r, i) => {
    const bottom = i === last;
    const isWin = r === winner;
    const newFam = i === 0 || rowsData[i - 1].family !== r.family;
    const newMod = i === 0 || rowsData[i - 1].model !== r.model || newFam;
    const endMod = i === last || rowsData[i + 1].model !== r.model || rowsData[i + 1].family !== r.family;
    rows.push(new TableRow({
      children: [
        cell([para(newFam ? [run(r.family, { bold: true })] : [run('')], AlignmentType.LEFT)],
             { w: W_FAM, bottom, vMerge: newFam ? VerticalMergeType.RESTART : VerticalMergeType.CONTINUE }),
        cell([para(newMod ? [run(r.model, { bold: isWin })] : [run('')], AlignmentType.LEFT)],
             { w: W_MODEL, bottom, soft: endMod, vMerge: newMod ? VerticalMergeType.RESTART : VerticalMergeType.CONTINUE }),
        cell([para([run(r.feats, { bold: isWin })])], { w: W_FEAT, bottom, soft: endMod }),
        ...metricCols.map(([, key]) => {
          const d = decimals[key] ?? 2;
          const txt = `${r.mean[key].toFixed(d)} ± ${r.sd[key].toFixed(d)}`;
          return cell([para([run(txt, { bold: isWin || key === primaryKey })])],
                      { w: W_MET, bottom, soft: endMod });
        }),
      ],
    }));
  });

  const doc = new Document({
    styles: { default: { document: { run: { font: FONT, size: SZ } } } },
    sections: [{
      properties: {
        page: {
          size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE },
          margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 },
        },
      },
      children: [
        new Paragraph({
          spacing: { after: 160 },
          children: [run(titleNo + ' ', { bold: true, size: 20 }), run(titleRest, { size: 20 })],
        }),
        new Table({ width: { size: TOTAL, type: WidthType.DXA }, columnWidths: COLW, rows }),
        ...footTexts.map(t => fn(typeof t === 'string' ? [run(t)] : t)),
      ],
    }],
  });
  Packer.toBuffer(doc).then(b => { fs.writeFileSync(out, b); console.log('written:', out, b.length, 'bytes'); });
}

const METHODS = 'Models were evaluated with person-grouped repeated 5-fold cross-validation (5 repeats, 25 folds); both records of an individual enrolled in two arms were kept in the same fold. Values are mean ± SD across the 25 validation folds. All predictors are baseline measurements plus the training-group indicator. Median imputation and standardization were fit within each training fold; elastic-net and ridge penalties were tuned by nested 3-fold cross-validation. Classification threshold 0.5.';
const FEATSETS = 'Feature sets — group-only: training-arm indicators; compact-5: training arm, daily step count, physical-activity duration, baseline 6-MWT distance, regular exercise habit (an exploratory, hypothesis-generated set; see footnote); full-211: from the complete pool of 211 candidate predictors (209 baseline variables plus the two training-arm indicators), 6 were selected within each training fold by forward sequential selection or backward elimination (RFE, step 10); test folds never entered selection.';

buildDoc(
  'Table S3',
  'Model-family comparison for predicting a clinically meaningful improvement in peak oxygen uptake (Δ ≥ 1.2 mL·kg⁻¹·min⁻¹) after eight weeks of training in patients with post-COVID-19 syndrome',
  [['AUC', 'AUC'], ['Accuracy', 'ACC'], ['Balanced accuracy', 'BAL'], ['F1', 'F1'],
   ['Sensitivity', 'SENS'], ['Specificity', 'SPEC'], ['PPV', 'PPV'], ['NPV', 'NPV'],
   ['MCC', 'MCC'], ['Brier score', 'BRIER']],
  'F1',
  rows => rows.filter(r => ['group-only', 'compact-5'].includes(r.feats))
    .reduce((a, b) => (b.mean.F1 > a.mean.F1 ? b : a)),
  RES.vo2, {},
  [
    [it('AUC'), run(' area under the ROC curve, '), it('PPV'), run(' positive predictive value, '),
     it('NPV'), run(' negative predictive value, '), it('MCC'), run(' Matthews correlation coefficient, '),
     it('Brier score'), run(' mean squared error of predicted probabilities (lower is better), '),
     it('SD'), run(' standard deviation, '), it('VO₂peak'), run(' peak oxygen uptake')],
    '†F1 score was the primary metric; among the named feature sets, the model with the highest mean F1 is shown in bold. Responder: increase of ≥ 1.2 mL·kg⁻¹·min⁻¹ in VO₂peak (mean rehabilitation gain reported for interstitial lung disease); n = 53 records (24 responders, 29 non-responders). The compact-5 set originates from an exploratory, multiplicity-charged search (nested cross-validated within-trained AUC 0.85, max-statistic p ≈ 0.004) and is presented as hypothesis-generating.',
    METHODS, FEATSETS,
  ],
  `${OUTDIR}/TableS3_ML_vo2_scoreboard.docx`,
);


const BLOCKS_PATH = path.join(__dirname, '..', 'results', 'metrics', 'block_results.json');
if (fs.existsSync(BLOCKS_PATH)) {
  const FAMORDER = ['Demographics', 'History', 'Pulmonary', 'Muscle', 'CPET', 'Walk test',
                    'Activity', 'Symptoms/QoL', 'Fitness', 'Reference'];
  const blocks = JSON.parse(fs.readFileSync(BLOCKS_PATH, 'utf8'))
    .slice().sort((a, b) => FAMORDER.indexOf(a.family) - FAMORDER.indexOf(b.family));
  buildDoc(
    'Table S4',
    'Block-wise signal scan for predicting a clinically anchored improvement in peak oxygen uptake (Δ ≥ 1.2 mL·kg⁻¹·min⁻¹): physiologically coherent variable blocks versus the selected activity-based set',
    [['AUC', 'AUC'], ['Balanced accuracy', 'BAL'], ['F1', 'F1'],
     ['Sensitivity', 'SENS'], ['Specificity', 'SPEC']],
    'AUC',
    rows => rows.find(r => r.model === 'compact-5 (selected set)'),
    blocks, {},
    [
      [it('AUC'), run(' area under the ROC curve, '), it('SD'), run(' standard deviation, '),
       it('VO₂peak'), run(' peak oxygen uptake')],
      '†AUC was the primary metric; the selected activity-based set (compact-5) is shown in bold. All models additionally include the training-arm indicators. Evaluation on the trained subset (IMT + HIIT; n = 36, 23 responders) with person-grouped repeated 5-fold cross-validation; values are mean ± SD across the 25 validation folds. For each block the better of logistic regression and elastic-net logistic is shown.',
      'Blocks were defined mechanically from variable names as physiologically coherent groups, blind to the outcome, and capped at 12 variables (lowest missingness first). "Best single variable" reports the best-performing of all 209 baseline variables and is therefore optimistic by construction; even so, no single variable approached the daily-activity block or its union with walking capacity (compact-5, an exploratory set; selection-corrected nested cross-validated AUC 0.85).',
    ],
    `${OUTDIR}/TableS4_ML_vo2_blocks.docx`,
  );
}
