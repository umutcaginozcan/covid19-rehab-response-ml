// Zoo scoreboard -> Word tables (Table 15-16)
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, VerticalAlign, VerticalMergeType, PageOrientation,
} = require('docx');

const RES = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'results', 'metrics', 'zoo_results.json'), 'utf8'));
const OUTDIR = process.env.SIXMWT_OUT || path.join(__dirname, '..', 'results', 'tables');

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
const FEATSETS = 'Feature sets — group-only / baseline + group: training-arm indicators (plus baseline 6-MWT distance for regression); compact-8: training arm, baseline 6-MWT distance, MIP, VO₂peak, FSS, SGRQ total, age, BMI; full-16 additionally: sex, time since COVID-19 diagnosis, CT involvement >50%, hospitalization, Charlson index, smoking history, exercise habit, MEP, quadriceps strength; full-211: from the complete pool of 211 baseline variables, 6 were selected within each training fold by forward sequential selection or backward elimination (RFE, step 10); test folds never entered selection.';

buildDoc(
  'Table 15',
  'Model-family comparison for predicting a clinically meaningful improvement in six-minute walk distance (Δ ≥ 25 m) after eight weeks of training in patients with post-COVID-19 syndrome',
  [['AUC', 'AUC'], ['Accuracy', 'ACC'], ['Balanced accuracy', 'BAL'], ['F1', 'F1'],
   ['Sensitivity', 'SENS'], ['Specificity', 'SPEC'], ['PPV', 'PPV'], ['NPV', 'NPV'],
   ['MCC', 'MCC'], ['Brier score', 'BRIER']],
  'F1',
  rows => rows.find(r => r.model === 'Logistic regression' && r.feats === 'compact-8'),
  RES.clf, {},
  [
    [it('AUC'), run(' area under the ROC curve, '), it('PPV'), run(' positive predictive value, '),
     it('NPV'), run(' negative predictive value, '), it('MCC'), run(' Matthews correlation coefficient, '),
     it('Brier score'), run(' mean squared error of predicted probabilities (lower is better), '),
     it('SD'), run(' standard deviation, '), it('6-MWT'), run(' six-minute walk test')],
    '†F1 score was the prespecified primary metric for model selection; the selected model (logistic regression, compact-8) is shown in bold. Responder: increase of ≥ 25 m in 6-MWT distance; n = 55 records (31 responders, 24 non-responders).',
    METHODS, FEATSETS,
  ],
  `${OUTDIR}/Table15_ML_responder_scoreboard.docx`,
);

buildDoc(
  'Table 16',
  'Model-family comparison for predicting post-training six-minute walk distance in patients with post-COVID-19 syndrome',
  [['R²', 'R2'], ['MAE (m)', 'MAE'], ['RMSE (m)', 'RMSE'], ['Pearson r', 'PEARSON']],
  'R2',
  rows => rows.find(r => r.model === 'Linear regression' && r.feats === 'base+group'),
  RES.reg, { MAE: 1, RMSE: 1 },
  [
    [it('R²'), run(' coefficient of determination, '), it('MAE'), run(' mean absolute error, '),
     it('RMSE'), run(' root mean square error, '), it('m'), run(' meters, '),
     it('SD'), run(' standard deviation, '), it('6-MWT'), run(' six-minute walk test')],
    '†R² was the primary metric for model selection; the selected model (linear regression, baseline + group) is shown in bold. Target: 6-MWT distance after the eight-week training period; n = 55 records.',
    'Person-grouped repeated 5-fold cross-validation (5 repeats); values are mean ± SD across the 25 validation folds; all predictors are baseline measurements plus the training-group indicator; imputation and scaling were fit within training folds. Feature sets as defined in Table 15.',
  ],
  `${OUTDIR}/Table16_ML_regression_scoreboard.docx`,
  1550,
);
