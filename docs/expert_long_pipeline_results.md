# Expert–Long Slide Generation Pipeline Outputs

## A) Parsed & cleaned paper
**File:** `parsed_clean/acl17.json`
```json
{
  "doc_id": "acl17",
  "title": "Transition-based Semantic Dependency Parsing with Pointer Networks",
  "abstract": "We present a transition-based parser using pointer networks for labeled semantic dependencies...",
  "sections": [
    {
      "heading": "1 Introduction",
      "paragraphs": [
        "Semantic dependency parsing predicts labeled predicate-argument graphs.",
        "Recent neural approaches show gains; we explore a pointer-network decoder."
      ]
    },
    {
      "heading": "2 Method",
      "paragraphs": [
        "We encode tokens with a BiLSTM.",
        "A pointer network decoder predicts head indices and labels via attention."
      ]
    },
    {
      "heading": "3 Experiments",
      "paragraphs": [
        "We evaluate on DM, PAS, and PSD.",
        "Our model outperforms graph-based baselines."
      ]
    },
    {
      "heading": "4 Ablations",
      "paragraphs": [
        "Removing label attention hurts labeled F1 by 0.8.",
        "Replacing pointer with bilinear head selection reduces UAS by 0.9."
      ]
    },
    {
      "heading": "5 Conclusion",
      "paragraphs": [
        "Pointer networks are effective for semantic dependency parsing.",
        "Future work includes multilingual evaluation."
      ]
    }
  ],
  "figures": [
    {"id": "Fig1", "caption": "Model architecture of the pointer network parser."},
    {"id": "Fig2", "caption": "Ablation study across datasets."}
  ]
}
```

## B) Retrieval indexes
**Lexical index (BM25):** `indexes/acl17_bm25/`  
**Semantic index (embeddings):** `indexes/acl17_embeddings.faiss`  
**Probe:** `indexes/acl17_probe.json`
```json
{
  "unit": "sentence",
  "records": [
    {"sid": "2-1", "section": "2 Method", "text": "We encode tokens with a BiLSTM."},
    {"sid": "2-2", "section": "2 Method", "text": "A pointer network decoder predicts head indices and labels via attention."},
    {"sid": "3-2", "section": "3 Experiments", "text": "Our model outperforms graph-based baselines."}
  ]
}
```

## C) TG (Topic Generation) — training example & model output
**Training data line:** `tg_sft/train.jsonl`
```json
{
  "input": {
    "title": "Transition-based Semantic Dependency Parsing with Pointer Networks",
    "abstract": "We present a transition-based parser using pointer networks for labeled semantic dependencies...",
    "headings": ["1 Introduction", "2 Method", "3 Experiments", "4 Ablations", "5 Conclusion"],
    "persona": {"audience": "expert", "length": "long"}
  },
  "target_topics": [
    "Motivation & Task Definition",
    "Pointer-Network Transition System",
    "Training Objective & Labeling",
    "Datasets & Experimental Setup",
    "Main Results on DM/PAS/PSD",
    "Ablation: Label Attention & Decoder Choices",
    "Error Analysis & Discussion",
    "Conclusion & Future Work"
  ]
}
```

**TG generation:** `tg_outputs/acl17.topics.json`
```json
{
  "doc_id": "acl17",
  "topics": [
    "Motivation: Semantic Dependency Parsing",
    "Model Overview: Pointer-Network Decoder",
    "Encoder & Attention Mechanisms",
    "Transition System & Label Prediction",
    "Training Objective & Hyperparameters",
    "Datasets: DM, PAS, PSD",
    "Results vs Graph-based Baselines",
    "Ablation: Label Attention / Head Selection",
    "Error Patterns & Limitations",
    "Conclusion & Future Directions"
  ]
}
```

## D) CE (Content Extraction) — per-topic candidates & selected bullets
**Candidates pack:** `candidates/acl17/topic_02.json`
```json
{
  "doc_id": "acl17",
  "topic": "Model Overview: Pointer-Network Decoder",
  "candidates": [
    {"sid":"2-1","section":"2 Method","text":"We encode tokens with a BiLSTM."},
    {"sid":"2-2","section":"2 Method","text":"A pointer network decoder predicts head indices and labels via attention."},
    {"sid":"3-2","section":"3 Experiments","text":"Our model outperforms graph-based baselines."}
  ],
  "retrieval": {"bm25_k":10,"emb_k":10,"fusion":"MMR"}
}
```

**CE output bullets:** `ce_outputs/acl17/topic_02.bullets.json`
```json
{
  "doc_id": "acl17",
  "topic": "Model Overview: Pointer-Network Decoder",
  "bullets": [
    "Tokens are encoded with a BiLSTM encoder.",
    "A pointer-network decoder selects head indices and labels via attention."
  ],
  "evidence": [
    {"section":"2 Method","sid_list":["2-1","2-2"]}
  ]
}
```

## E) Preference data & reward signals (Expert–Long)
**TG pair:** `prefs/tg_pairs.jsonl`
```json
{
  "prompt": {...},
  "chosen": [
    "Motivation: Semantic Dependency Parsing",
    "Model Overview: Pointer-Network Decoder",
    "Encoder & Attention Mechanisms",
    "Transition System & Label Prediction",
    "Training Objective & Hyperparameters",
    "Datasets: DM, PAS, PSD",
    "Results vs Graph-based Baselines",
    "Ablation: Label Attention / Head Selection",
    "Error Patterns & Limitations",
    "Conclusion & Future Directions"
  ],
  "rejected": [
    "Introduction",
    "Background",
    "Method",
    "Experiments",
    "Results",
    "Conclusion"
  ],
  "criteria": ["comprehensibility","length_satisfaction"],
  "persona": "expert-long"
}
```

**CE pair:** `prefs/ce_pairs.jsonl`
```json
{
  "prompt": {...},
  "chosen": {
    "bullets": [
      "Evaluated on DM, PAS, and PSD with labeled/unlabeled F1.",
      "Model surpasses graph-based baselines across datasets."
    ]
  },
  "rejected": {
    "bullets": [
      "We describe experiments.",
      "Results are promising."
    ]
  },
  "criteria": ["comprehensibility","length_satisfaction"]
}
```

**Reward scoring trace:** `rewards/scoring_examples.json`
```json
{
  "tg_example": {"criterion": "comprehensibility","persona": "expert-long","score_chosen": 0.82,"score_rejected": 0.41},
  "ce_example": {"criterion": "length_satisfaction","persona": "expert-long","score_chosen": 0.77,"score_rejected": 0.35}
}
```

## F) Alignment pass (before → after)
**Before align:**
```json
{
  "slide": {
    "title": "Transition System & Label Prediction",
    "bullets": [
      "Decoder predicts head indices and labels via attention.",
      "We encode tokens with a BiLSTM.",
      "Training objective is cross-entropy on transitions."
    ]
  }
}
```

**After align:**
```json
{
  "slide": {
    "title": "Transition System & Label Prediction",
    "bullets": [
      "BiLSTM encodes token representations.",
      "Pointer decoder selects head indices with attention.",
      "Labels predicted jointly; optimized with cross-entropy over transitions."
    ]
  }
}
```

## G) Factuality/NLI gating
```json
{
  "slide_idx": 6,
  "title": "Main Results on DM/PAS/PSD",
  "bullets": [
    "Model surpasses graph-based baselines on labeled F1.",
    "Achieves state-of-the-art on all datasets."
  ],
  "flags": [
    {"type":"contradiction","bullet_idx":1,"reason":"No evidence claims SOTA on all datasets."}
  ]
}
```

## H) Visual selection & layout
```json
{
  "slide_idx": 2,
  "title": "Model Overview: Pointer-Network Decoder",
  "figure": {"id":"Fig1","caption":"Model architecture of the pointer network parser."},
  "layout_id": "T+I+B"
}
```

## I) Final export manifest (3-slide sample)
```json
{
  "deck_meta": {
    "doc_id": "acl17",
    "title": "Transition-based Semantic Dependency Parsing with Pointer Networks",
    "persona": "expert",
    "length": "long",
    "slides_planned": 10
  },
  "slides": [
    {
      "title": "Motivation: Semantic Dependency Parsing",
      "bullets": [
        "Task: predict labeled predicate–argument graphs.",
        "Goal: improve over graph-based parsers using a pointer decoder."
      ]
    },
    {
      "title": "Model Overview: Pointer-Network Decoder",
      "bullets": [
        "BiLSTM encodes token representations.",
        "Pointer decoder selects head indices with attention; labels predicted jointly."
      ],
      "figure": {"id":"Fig1","caption":"Model architecture of the pointer network parser."},
      "layout_id": "T+I+B"
    },
    {
      "title": "Main Results on DM/PAS/PSD",
      "bullets": [
        "Evaluated on DM, PAS, PSD with labeled/unlabeled F1.",
        "Outperforms graph-based baselines across datasets."
      ],
      "figure": {"id":"Fig2","caption":"Ablation study across datasets."},
      "layout_id": "T+I+B"
    }
  ]
}
```

## J) End-to-end eval snapshot
```json
{
  "tg_rougeL": 0.46,
  "ce_precision": 0.71,
  "ce_recall": 0.68,
  "ce_f1": 0.695,
  "human_eval": {
    "n_reviewers": 3,
    "comprehensibility_mean": 4.3,
    "length_satisfaction_mean": 4.1
  },
  "pf_vs_sft": {
    "tg_rougeL_delta": "+0.05",
    "ce_f1_delta": "+0.04"
  }
}
```
