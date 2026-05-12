# nlp-hf template

Skeleton for NLP competitions using HuggingFace Transformers.

**Status:** skeleton. `build_model_and_tokenizer()` and `tokenize_batch()` raise `NotImplementedError` for the agent to implement per competition. The outer loop (optimizer / scheduler / mixed precision / fold-by-fold checkpoints / progress logging) is functional.

## What the agent must implement

```python
def build_model_and_tokenizer(cfg):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=<comp specific>,
    )
    return model, tok

def tokenize_batch(batch, tokenizer, cfg):
    # The default _make_loader already handles single-text tokenization.
    # Override only if you need pair-text, sliding windows, or special handling.
    ...

def compute_metric(metric_name, y_true, y_pred):
    # Implement comp-specific metric.
    ...
```

For other heads:
- Token classification: use `AutoModelForTokenClassification`, output is per-token logits, metric is per-token-then-aggregate
- Regression: `AutoModelForSequenceClassification(num_labels=1)`, the model emits a scalar
- Seq2seq: `AutoModelForSeq2SeqLM`, inference needs `.generate()`

## Config (`template.yaml`)

Defaults: `microsoft/deberta-v3-base`, max_seq_length 512, 3 epochs, AdamW lr 2e-5, linear warmup-then-decay, AMP on.

Common overrides:
- `model_name` — `microsoft/deberta-v3-large`, `roberta-base`, `xlm-roberta-base`, `microsoft/deberta-v2-xxlarge`, etc.
- `max_seq_length` — 512 is the standard; 1024 doubles VRAM and slows ~1.6× (super-linear in attention)
- `gradient_accumulation_steps` — when batch is too small for a stable gradient on long sequences

## Compute requirements

- DeBERTa-v3-base at max_seq=512: ~8 GB VRAM. base/medium model.
- DeBERTa-v3-large at max_seq=512: ~16–20 GB VRAM. Worth it for many comps.
- Long-sequence (>1024 tokens) needs special handling — usually a sliding window / span head.

## Mandatory pre-training checks

Before invoking:
- Check Rule 10 pre-trained-models cutoff: many comps allow only pretrained models released before a cutoff date. `microsoft/deberta-v3-base` is from 2021; check the comp's cutoff.
- VRAM check: `torch.cuda.get_device_properties(0).total_memory >= template.min_vram_gb * 1e9`.
- Internet at infer time: usually off in code-only comps. The template assumes weights are pre-downloaded into the Kaggle Datasets mount; the agent must copy weights to `data/models/` before training in those cases.

## What this template does NOT cover

- Multi-task / multi-head models (e.g. classification + extraction simultaneously) — write a custom train.py.
- Domain-adaptive pretraining (MLM continuation on the comp's text) — a separate run before fine-tuning.
- LoRA / adapter-only fine-tuning — agent edits the model construction to wrap with `peft`.
- Long-document split-and-aggregate strategies — agent writes a sliding-window inference loop.
