# Task type detection

The goal: turn the comp's metadata into one canonical `task_type` value plus the structured metric / submission-format fields. Every later stage branches on these.

## Allowed `task_type` values

```
tabular-binary
tabular-multiclass
tabular-regression
tabular-ranking
image-classification
image-segmentation
image-detection
image-instance-segmentation
nlp-classification
nlp-regression
nlp-token-classification
nlp-generation
time-series-forecast
time-series-event-detection
audio-classification
multimodal
graph
recommendation
other
```

If the comp does not fit any of the above, set `task_type: other` and record the actual nature in `comp_profile.metric.description`.

## Detection algorithm

Run these checks in order, stopping at the first match.

### Step 1 — read sample_submission.csv

```bash
head -n 3 runs/<slug>/data/raw/sample_submission.csv
wc -l runs/<slug>/data/raw/sample_submission.csv
```

| sample_submission column pattern | Task type |
|---|---|
| `id,target` with target ∈ {0,1} | `tabular-binary` |
| `id,target` with target real-valued | `tabular-regression` |
| `id,<class_a>,<class_b>,...` (one column per class with probabilities) | `tabular-multiclass` |
| `id,Predicted` where Predicted is RLE / mask string | `image-segmentation` |
| `id,box_x,box_y,box_w,box_h,...` or COCO-style JSON expected | `image-detection` |
| Time-series schema: `series_id, step, event, score` | `time-series-event-detection` |
| `id, predicted_label_seq` (variable-length sequences) | `nlp-token-classification` |
| `id, generated_text` | `nlp-generation` |
| `(user_id, item_id, score)` | `recommendation` |

### Step 2 — cross-check with evaluation metric

`kaggle competitions view <slug>` returns an `evaluationMetric` string. Match it to a canonical metric:

| metric string contains | metric.name | direction |
|---|---|---|
| `AUC` / `area under ROC` | `auc` | maximize |
| `LogLoss` / `Log Loss` | `log_loss` | minimize |
| `Accuracy` | `accuracy` | maximize |
| `F1` / `F-beta` | `f1` / `f_beta` | maximize |
| `Quadratic Kappa` / `Cohen Kappa` | `quadratic_weighted_kappa` | maximize |
| `RMSE` / `Root Mean Squared Error` | `rmse` | minimize |
| `RMSLE` | `rmsle` | minimize |
| `MAE` | `mae` | minimize |
| `MAP` / `Mean Average Precision` | `map` | maximize |
| `mAP@N` | `map_at_n` | maximize |
| `IoU` / `Jaccard` / `Dice` | `iou` / `dice` | maximize |
| `NDCG` | `ndcg` | maximize |
| `BLEU` / `ROUGE` | `bleu` / `rouge` | maximize |
| `pinball` | `pinball_loss` | minimize |

If the metric is custom or named after the comp (e.g. "Sleep AP"), set `metric.name` to a snake_case slug of the comp's wording and record the description verbatim. Look for a `metric.py` or `competition_metric.py` in the dataset — code-only comps often ship one. If present, record its path.

Cross-check: an `accuracy` metric is inconsistent with a `tabular-regression` task type. If they disagree, escalate.

### Step 3 — inspect the data

```bash
ls -la runs/<slug>/data/raw/
```

| Files present | Hints |
|---|---|
| `train.csv`, `test.csv`, `sample_submission.csv` | Tabular |
| `train_images/` or `images/` folder | CV |
| `train.parquet` with a single text column | NLP |
| Time-stamped data (`timestamp`, `date`, or hourly partitions) | Time series |
| Multiple modalities (text + image) | Multimodal |
| `.wav` or `.flac` | Audio |
| Edge lists / adjacency files | Graph |

### Step 4 — overview page (last resort)

If the above are still ambiguous, fetch the Overview page via WebFetch and look for phrases like:

- "binary classification" / "multi-class classification" → tabular-binary or tabular-multiclass
- "segmentation mask" → image-segmentation
- "bounding box" → image-detection
- "forecast the next N days" → time-series-forecast
- "detect events in time series" → time-series-event-detection
- "answer questions about" → nlp-generation or nlp-classification depending on output

## Submission format detection

After task_type is set:

| task_type | Default submission format |
|---|---|
| tabular-* | CSV with `id` + target columns |
| image-classification | CSV with `id` + predicted class or probabilities |
| image-segmentation | CSV with `id` + RLE encoding (rarely PNG masks via kernel) |
| image-detection | CSV with `id` + box list, or kernel-only with COCO-format output |
| nlp-token-classification | CSV with `id` + predicted token labels |
| time-series-event-detection | CSV with `series_id, step, event, score` |

Always confirm against the actual `sample_submission.csv` columns. If the actual file has different columns, the detection was wrong — re-do.

## Edge cases worth knowing about

- **Code-only competitions.** No CSV upload allowed; user must publish a Kaggle notebook that emits `submission.csv` at runtime. Detection: comp page mentions "code competition" or `kaggle competitions view` shows `kernelType` ∈ {`script`, `notebook`}. Set `submission.code_only: true` and `submission.format: code-only-notebook`.
- **Synthetic + real data.** Some "Playground Series" comps have synthetic train + real test (or vice versa). Note in `rules_summary.md`; it affects whether external data helps.
- **Multi-target regression.** Submission has many real-valued columns. Treat as `tabular-regression` with `metric.multi_target: true` and record per-target metric handling.
- **Hierarchical / sub-segmented metrics.** E.g. `weighted-by-group F1`. Record the description verbatim; do not collapse to a simpler name.

## What to put in `comp_profile.yaml`

After detection, write the full profile per `state-contract.md`. The fields the detection sets:

```yaml
task_type: <one of the allowed values>
metric:
  name: <slug>
  direction: maximize | minimize
  description: <verbatim from comp page if non-standard>
  evaluation_script_path: <path-in-data/raw if shipped, else null>
  multi_target: false
submission:
  format: csv | code-only-notebook | code-only-script
  columns: [<list>]
  size_limit_mb: <int or null>
  code_only: false
```

If any field cannot be determined, set it to `null` and escalate. Do not guess.
