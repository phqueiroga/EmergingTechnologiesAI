const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, UnderlineType,
} = require("docx");
const fs = require("fs");

const FONT = "Calibri";
const SIZE = 24; // 12pt in half-points

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    children: [new TextRun({ text, font: FONT, size: SIZE, ...opts })],
  });
}

function pRuns(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    ...opts,
    children: runs.map(r => new TextRun({ font: FONT, size: SIZE, ...r })),
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 280, after: 140 },
    children: [new TextRun({ text, font: FONT, bold: true, size: 28 })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text, font: FONT, bold: true, size: 25 })],
  });
}

function bullet(text) {
  return new Paragraph({
    spacing: { after: 80 },
    bullet: { level: 0 },
    children: [new TextRun({ text, font: FONT, size: SIZE })],
  });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 1200, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "E7E6E6" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text, font: FONT, size: 18, bold: !!opts.header })],
    })],
  });
}

function resultsTable() {
  const header = ["Strategy", "Model", "MAE(frac)", "R2", "Recall", "Lead(min)", "Energy(kWh)", "Size(KB)"];
  const widths = [1400, 2200, 1100, 900, 900, 1100, 1500, 1100];
  const rows = [
    ["reference", "Ridge", "0.633", "-5.51", "0.41", "20.7", "7.0e-9", "0.6"],
    ["reference", "HistGB", "0.283", "-0.59", "0.40", "23.0", "1.1e-6", "358.4"],
    ["reference", "RandomForest", "0.280", "-0.60", "0.41", "23.0", "4.8e-6", "37656.8"],
    ["reference", "MLP", "0.540", "-4.20", "0.29", "30.1", "1.5e-6", "96.0"],
    ["half-rate", "Ridge", "0.323", "-1.08", "0.26", "18.7", "5.3e-9", "0.6"],
    ["half-rate", "HistGB", "0.230", "-0.05", "0.30", "20.4", "1.1e-6", "358.2"],
    ["half-rate", "RandomForest", "0.223", "-0.02", "0.32", "20.4", "5.1e-6", "37656.8"],
    ["half-rate", "MLP", "0.266", "-0.41", "0.32", "28.9", "1.5e-6", "95.9"],
    ["top-10-feat.", "Ridge", "0.543", "-3.43", "0.39", "20.6", "4.7e-9", "0.5"],
    ["top-10-feat.", "HistGB", "0.305", "-0.79", "0.43", "23.0", "7.7e-7", "329.1"],
    ["top-10-feat.", "RandomForest", "0.305", "-0.81", "0.43", "23.0", "2.8e-6", "37656.8"],
    ["top-10-feat.", "MLP", "0.322", "-0.88", "0.30", "17.6", "1.7e-6", "75.3"],
  ];
  return new Table({
    width: { size: 10200, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ children: header.map((h, i) => cell(h, { header: true, width: widths[i] })) }),
      ...rows.map(r => new TableRow({ children: r.map((v, i) => cell(v, { width: widths[i] })) })),
    ],
  });
}

function conditionTable() {
  const header = ["Condition", "Speed/Load", "Recall (ref.)", "Lead time (ref., min)", "Interpretation"];
  const widths = [1300, 1900, 1600, 2200, 3200];
  const rows = [
    ["1", "1800 rpm / 4000 N", "0.00 - 0.25", "0 - 28", "Weak / unreliable"],
    ["2", "1650 rpm / 4200 N", "0.13 - 0.29", "~7.5", "Moderate, short warning"],
    ["3", "1500 rpm / 5000 N", "0.68 - 0.94", "~54.5", "Strong, practically useful"],
  ];
  return new Table({
    width: { size: 10200, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ children: header.map((h, i) => cell(h, { header: true, width: widths[i] })) }),
      ...rows.map(r => new TableRow({ children: r.map((v, i) => cell(v, { width: widths[i] })) })),
    ],
  });
}

const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    children: [
      // Header block
      p("Pedro Henrique Teixeira Queiroga — Student ID: 25113046", { bold: true }),
      p("MSc/PGDip AI for Business, National College of Ireland — H9ETS Emerging Artificial Intelligence Technologies and Sustainability"),
      p("[CONFIRME nome, nº de estudante e nome exato do curso antes de submeter]", { italics: true, color: "C00000" }),

      new Paragraph({
        spacing: { before: 200, after: 60 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({
          text: "Green AI Trade-offs in Vibration-Based Remaining Useful Life Prediction for Rolling-Element Bearings",
          bold: true, size: 30, font: FONT,
        })],
      }),
      p("[RASCUNHO DE TRABALHO — reescrever com suas próprias palavras antes de submeter. O brief proíbe explicitamente submeter conteúdo copiado de IA generativa e o Turnitin faz varredura eletrônica. Use este documento como andaime de conteúdo/estrutura, não como texto final.]",
        { italics: true, bold: true, color: "C00000" }),

      h1("1. Introduction and Motivation"),
      p("Predictive maintenance (PdM) sits at the intersection of two sustainability concerns that are usually discussed separately. On one side, unplanned failure of rotating machinery (bearings, motors, compressors) causes material waste, unplanned energy spikes, and safety risk across manufacturing and pharmaceutical production lines; being able to anticipate failure and act only when needed is a direct contribution to operational and environmental efficiency. On the other side, the machine learning systems used to make those predictions are not free: training and running models has a measurable energy and carbon cost, and the Green AI literature (Schwartz et al., 2020) argues that this cost is too often ignored when researchers report only accuracy."),
      p("This report addresses both sides at once. Using the FEMTO/PRONOSTIA bearing degradation dataset (Nectoux et al., 2012), originally built for the IEEE PHM 2012 Prognostics Challenge, this project trains and compares several regression models for Remaining Useful Life (RUL) estimation from vibration signals, and evaluates them not only on predictive accuracy but on the energy, carbon and model-size cost of achieving it. Three configurations are compared: a full-cost reference configuration, and two Green AI strategies (halved sampling rate, and a reduced feature set), to test whether meaningful cost reductions are possible without an unacceptable loss of predictive maintenance value."),
      p("The research question this report answers is: can the computational and energy cost of RUL prediction on this dataset be substantially reduced while preserving enough predictive-maintenance value (measured as failure-detection recall and real warning lead time, not only regression error) to remain useful in practice?"),

      h1("2. Background Research"),
      p("Sustainability in AI is usually discussed on two separate axes. The first is AI for sustainability — using AI systems to reduce environmental impact in other domains, of which predictive maintenance is a canonical example: replacing calendar-based or reactive maintenance with condition-based maintenance reduces both unnecessary part replacement and unplanned downtime (Lei et al., 2018), which directly supports resource-efficiency goals aligned with the UN Sustainable Development Goals (notably SDG 9, Industry, Innovation and Infrastructure, and SDG 12, Responsible Consumption and Production)."),
      p("The second axis, sustainability of AI itself, is the Green AI agenda (Schwartz et al., 2020), which argues that the field has optimised almost exclusively for accuracy (\"Red AI\") while treating computational and energy cost as a footnote. Strubell et al. (2019) quantified this concern concretely by estimating the carbon footprint of training large NLP models, and the same logic applies at a much smaller scale to industrial ML: a predictive-maintenance model deployed on thousands of pieces of equipment across a plant multiplies a small per-inference cost into a non-trivial operational one, and the training/retraining cycle is repeated as equipment and conditions change."),
      p("The FEMTO/PRONOSTIA dataset itself has an established literature. It was produced on the PRONOSTIA experimental platform (Nectoux et al., 2012) for accelerated bearing degradation testing, and became the basis of the IEEE PHM 2012 Data Challenge, whose scoring function deliberately penalised late (over-optimistic) RUL predictions more heavily than early ones, reflecting the asymmetric real-world cost of missing a failure versus being overly conservative. More recent work by Juodelyte et al. (2022) uses this same dataset to motivate predictive maintenance specifically in pharmaceutical manufacturing contexts, where equipment maintenance is additionally subject to regulatory audit; notably, that work avoids regressing an absolute RUL value across bearings and instead predicts discrete degradation stages, for reasons that this project's own experimentation independently reproduced (see Section 3.4)."),

      h1("3. Experiment Details"),
      p("This section documents the experimental pipeline as it was actually built, including the mistakes made and corrected along the way, because each correction is itself evidence about how this dataset behaves and what a valid RUL pipeline requires."),

      h2("3.1 Dataset and initial ingestion problem"),
      p("The Learning_set portion of the FEMTO/PRONOSTIA dataset was used: six bearings, two per operating condition (1800 rpm/4000 N, 1650 rpm/4200 N, 1500 rpm/5000 N), each with horizontal and vertical accelerometer recordings sampled at 25.6 kHz in 0.1 s bursts (2 560 points) taken every 10 seconds until failure, giving 7 534 vibration recordings in total."),
      p("An early version of the ingestion pipeline selected every CSV file under each bearing folder (glob '*.csv'), which in the original PRONOSTIA distribution also includes single-column temp_*.csv temperature logs. Feeding a temperature file into a vibration-feature extractor either crashed the pipeline outright, or, worse, in an earlier run silently produced a feature table in which most rows were duplicated or constant across the two vibration channels. This is a critical failure mode precisely because it does not always crash: a degenerate feature table still trains a model and produces plausible-looking metrics, which is why a manual validation gate (Section 3.3) was added rather than trusting downstream numbers by inspection alone."),
      p("The fix was to (1) restrict the file glob to acc_*.csv only, and (2) make the raw-signal reader robust to the exact column layout by always reading the last two numeric columns as (horizontal, vertical) acceleration, since PRONOSTIA distributions vary between a plain two-column layout and a six-column layout with four leading timestamp fields (hour, minute, second, microsecond). This was confirmed empirically by inspecting a raw file before writing the feature extractor, rather than assuming a layout."),

      h2("3.2 Feature extraction"),
      p("Twelve features were extracted per channel (24 total) covering both time- and frequency-domain descriptors of vibration health, following standard bearing-diagnostics practice: RMS, standard deviation, peak-to-peak amplitude, crest factor, skewness, kurtosis, signal energy, spectral centroid, spectral entropy, and three frequency-band energy ratios (0-1 kHz, 1-5 kHz, 5 kHz-Nyquist). This combination captures both amplitude growth (RMS, energy) and the change in vibration character associated with developing defects (kurtosis, crest factor, spectral shift) that is characteristic of bearing degradation."),

      h2("3.3 Validation gate"),
      p("Given the earlier silent-corruption failure, a validation gate was implemented and run before any model training: it checks that the feature table has exactly 24 non-null columns, that no feature is near-constant (<=1% unique values) or predominantly zero (>50% of rows), and that the horizontal and vertical channels are not identical across more than half of observations. On the corrected pipeline and real data this gate passed: 0 NaNs, and the majority of features had close to 7 534 unique values (one per observation), confirming the ingestion fix was effective and not merely silencing the symptom."),

      h2("3.4 Target formulation: absolute RUL fails to generalise"),
      p("The initial RUL target was defined as absolute remaining minutes: rul_min = (n_obs_bearing - 1 - i) x 10s / 60 for observation i. A first experiment trained models on four bearings (conditions 2 and 3) and tested on two held-out bearings (condition 1), using a GroupShuffleSplit by bearing. Every model/strategy combination produced negative R2 (as low as -1.35) and MAE of 120-185 minutes, worse than predicting the mean."),
      p("Two diagnostics isolated the cause. First, a same-bearing random row holdout (train/test split within Bearing1_1 only) achieved R2 = 0.993, proving the extracted features do carry a strong, learnable degradation signal and ruling out a data-quality explanation. Second, a within-condition, cross-bearing test (train on one bearing, test on its sibling in the same operating condition) showed positive R2 for condition 2 (0.25-0.31) but still strongly negative for conditions 1 and 3. The distinguishing factor was total bearing lifetime: e.g. Bearing1_1 survived 2 803 recordings (~467 minutes) while its condition-sibling Bearing1_2 survived only 871 (~145 minutes). A model trained on one bearing's absolute time-to-failure scale extrapolates poorly to a bearing with a very different total lifespan, even under identical operating conditions - individual manufacturing variance in rolling-element bearings is well documented and is exactly why Juodelyte et al. (2022) avoid absolute RUL regression."),
      p("The target was therefore changed to normalised RUL, rul_frac = rul_min / max(rul_min) per bearing (1 = start of life, 0 = failure), which is scale-invariant across bearings of different total lifespan. Re-running the within-condition diagnostic showed a partial but real improvement (e.g. condition 2 RandomForest R2 up to 0.615; condition 1 Ridge R2 improved from -4.82 to -0.04 under the half-rate strategy), and this normalised target was used for all subsequent experiments."),

      h2("3.5 Train/test protocol"),
      p("Following the diagnostic evidence and the documented PRONOSTIA/IEEE PHM 2012 protocol of training and testing within the same operating condition, the final experiments use, for each of the three conditions, one bearing for training and its sibling for testing (e.g. Bearing1_1 -> Bearing1_2), rather than a cross-condition split. Results are reported both per condition and averaged, because - as Section 4.2 shows - averaging alone would hide a real and practically important difference in viability between conditions."),

      h2("3.6 Models and Green AI strategies"),
      p("Four regressors spanning a deliberate range of computational cost were compared: Ridge regression (linear, near-zero training/inference cost), HistGradientBoosting (histogram-based gradient boosting, moderate cost), RandomForest (300 trees, higher cost and by far the largest serialized size), and a small MLP (two hidden layers, 64/32 units, early stopping). Three data configurations were compared for each model:"),
      bullet("reference: all 24 features at the native 25.6 kHz sampling rate."),
      bullet("half-rate: signal decimated by 2 (12.8 kHz) before feature extraction, same 24 features - tests whether cheaper data acquisition/processing is viable."),
      bullet("top-10-features: the 10 most important features (ranked by RandomForest impurity importance on the reference configuration), same sampling rate - tests whether a smaller, cheaper feature pipeline is viable."),
      p("This 4-model x 3-strategy x 3-condition design (36 runs) isolates cost reduction on two independent axes (signal acquisition rate and feature dimensionality) so that their effects are not confounded."),

      h2("3.7 Evaluation metrics: why MAE alone is insufficient"),
      p("Mean Absolute Error treats every prediction error identically regardless of how close the bearing is to failure, but in predictive maintenance a large error early in a bearing's life is inconsequential while the same error close to failure can mean a missed warning. This asymmetry is not a new observation for this dataset specifically: the original IEEE PHM 2012 Challenge scoring function penalised late (over-optimistic) predictions more heavily than early ones for exactly this reason. Two additional, maintenance-relevant metrics were therefore added:"),
      bullet("Critical recall: of the test observations truly within the last 20% of remaining life, the fraction the model correctly flagged (predicted rul_frac <= 0.20)."),
      bullet("Lead time (minutes): for each test bearing, the real time-to-failure remaining at the first point, in chronological order, where the model raises a correct critical alarm - i.e. how much real warning the model would have given in practice."),
      p("Sustainability cost was measured per run as wall-clock training/inference time, energy (estimated as constant-power draw x elapsed time, using an assumed 15 W CPU-share proxy since hardware-level energy measurement (e.g. CodeCarbon on real hardware) was not available in the execution sandbox used for this project - this is a stated limitation, see Section 6), resulting CO2e (using a global-average grid intensity of 429 gCO2/kWh), and serialized model size in KB."),

      new Paragraph({ children: [new PageBreak()] }),

      h1("4. Sustainability AI Technologies"),
      h2("4.1 Approach"),
      p("The overall approach frames Green AI not as a single technique but as a family of configuration-level decisions, each independently measurable against the same accuracy/viability metrics: how much raw signal is acquired and processed (sampling-rate strategy), how many derived features are computed and fed to the model (feature-selection strategy), and which model family is used (algorithm choice). This mirrors the framing in Schwartz et al. (2020) that Green AI gains often come from cheaper data pipelines and model choice rather than from optimising a single fixed model further."),

      h2("4.2 Experiment plan actually executed"),
      p("1) Establish a correct, validated ingestion and feature pipeline (Section 3.1-3.3). 2) Establish a target formulation and train/test protocol under which the features demonstrably carry a learnable signal (Section 3.4-3.5) - without this step, any later cost/accuracy trade-off comparison would be measuring noise, not a real trade-off. 3) Run the reference configuration and the two Green AI strategies across all four models and three operating conditions, recording both predictive-maintenance and sustainability metrics for each of the 36 runs. 4) Compare configurations using a Pareto-efficiency analysis on two independent axes - energy versus critical recall, and energy versus lead time - because, as Section 5 shows, no single accuracy metric captures what 'a good trade-off' means for this task."),

      h1("5. Sustainability Evaluation"),
      h2("5.1 Aggregate results"),
      p("Table 1 reports each strategy/model combination's mean performance and cost across the three operating conditions."),
      resultsTable(),
      p(""),
      h2("5.2 Condition-dependent viability"),
      p("Averaging across conditions hides a large and practically important difference. Table 2 summarises the reference configuration's critical recall and lead time by condition."),
      conditionTable(),
      p("Condition 3 (highest load, lowest speed) is the only regime in which the models are close to practically useful for real maintenance planning (up to 94-96% recall and ~54 minutes of warning). Condition 1 (highest speed) is essentially unusable across every model tested. This is a genuine finding about the dataset and the task, not a modelling defect: it should be reported as a scoped limitation of what this pipeline can currently claim, rather than glossed over in an aggregate number."),

      h2("5.3 Trade-off analysis"),
      p("Two Pareto-efficiency views were built on the aggregated results (energy vs. critical recall, energy vs. lead time). The top-10-features strategy dominates or matches the reference configuration's recall while using roughly half the training energy of the reference HistGradientBoosting/RandomForest runs, and its reduced feature computation lowers acquisition-side cost as well - a rare case where the cheaper option does not cost accuracy. RandomForest is never Pareto-optimal on either axis: its recall/lead-time gains over HistGradientBoosting are marginal, while its serialized size (~37 MB versus <1 KB for Ridge and ~330-360 KB for HistGradientBoosting) makes it disproportionately expensive to store or deploy at scale. The MLP achieves the longest lead times (28-30 minutes) but with mid-range recall (~29-32%), meaning it is confident and early when it does raise an alarm but misses more cases than the tree-based models - a genuine trade-off between warning frequency and warning notice, not a strictly dominated option."),
      p("half-rate downsampling produced a mixed but not purely negative result: for several model/condition combinations (e.g. Ridge under condition 1 and condition 3) it improved R2 relative to the reference rate while halving energy, plausibly because decimation removes high-frequency noise components that do not transfer between individual bearings; for other combinations it was roughly cost-neutral on accuracy. It is therefore a defensible but weaker Green AI lever than feature reduction for this dataset."),

      h2("5.4 Recommendation"),
      p("HistGradientBoosting combined with the top-10-features strategy is the recommended configuration: it is Pareto-optimal on both the energy-recall and energy-lead-time views simultaneously, achieves the highest recall observed (43%) at roughly a third of the reference configuration's training energy, and avoids RandomForest's disproportionate model-size cost. Where maximising warning lead time is the operational priority over detection frequency (e.g. long-lead maintenance scheduling with tolerance for missed low-priority alarms), the reference-rate MLP is the defensible alternative, at a higher but still modest energy cost."),

      h1("6. Conclusion"),
      p("This project set out to compare Green AI strategies for vibration-based RUL prediction on the FEMTO/PRONOSTIA dataset, and the process of getting to a trustworthy comparison turned out to be as informative as the comparison itself. Two ingestion/methodology errors - including non-vibration files in feature extraction, and using an absolute RUL target across bearings with very different total lifespans - initially produced results that looked interpretable (consistent negative R2, plausible-looking MAE) but were not measuring what they appeared to measure. Systematic validation (a same-bearing sanity check, a within-condition generalisation test, and cross-checking against the literature's own choice to avoid absolute RUL regression) was necessary before any Green AI comparison could be considered valid."),
      p("With a corrected pipeline, the evidence supports a scoped but real conclusion: reducing the feature set to the 10 most important features is a genuinely low-cost, low-risk Green AI strategy for this task, reducing energy without sacrificing - and in several cases improving - failure-detection recall, while reducing sampling rate is a weaker and more condition-dependent lever. Predictive-maintenance viability itself, however, is strongly condition-dependent in this dataset, and any deployment claim should be scoped to the operating regime it was validated on rather than presented as universal."),
      p("Future work should: measure energy directly on target hardware (e.g. CodeCarbon/RAPL) rather than via the time-based proxy used here; evaluate on the official PHM 2012 held-out Test_set bearings rather than only Learning_set siblings; model operating condition explicitly (e.g. condition-specific models or condition as an input feature) instead of assuming a single global model; and, following Juodelyte et al. (2022), evaluate whether reframing the task as degradation-stage classification rather than continuous RUL regression further improves cross-bearing generalisation, which the within-condition diagnostics in this report suggest is the dataset's main limiting factor."),

      new Paragraph({ children: [new PageBreak()] }),

      h1("Bibliography"),
      p("Juodelyte, D., Cheplygina, V., Graversen, T. and Bonnet, P. (2022) 'Predicting Bearings' Degradation Stages for Predictive Maintenance in the Pharmaceutical Industry', Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. Available at: https://arxiv.org/abs/2203.03259 (Accessed: 17 August 2026)."),
      p("Lei, Y., Li, N., Guo, L., Li, N., Yan, T. and Lin, J. (2018) 'Machine learning-based condition monitoring approaches for non-stationary rotating machinery prognostics', Mechanical Systems and Signal Processing, 104, pp. 799-834."),
      p("Nectoux, P., Gouriveau, R., Medjaher, K., Ramasso, E., Chebel-Morello, B., Zerhouni, N. and Varnier, C. (2012) 'PRONOSTIA: An experimental platform for bearings accelerated degradation tests', IEEE International Conference on Prognostics and Health Management, Denver, CO."),
      p("Schwartz, R., Dodge, J., Smith, N.A. and Etzioni, O. (2020) 'Green AI', Communications of the ACM, 63(12), pp. 54-63."),
      p("Strubell, E., Ganesh, A. and McCallum, A. (2019) 'Energy and Policy Considerations for Deep Learning in NLP', Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 3645-3650."),
      p("[ADICIONE/CONFIRME todas as referências: verifique se você de fato leu cada uma antes de citá-la, e adicione qualquer material de aula que voce for usar.]", { italics: true, bold: true, color: "C00000" }),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/home/user/EmergingTechnologiesAI/report/TABA_H9ETS_draft.docx", buf);
  console.log("written");
});
