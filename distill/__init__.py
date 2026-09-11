"""Offline knowledge-distillation tooling: teacher embeddings -> soft-logit targets.

The pipeline is deliberately teacher-agnostic:
  teachers.py    -- swappable ClipTeacher implementations (BirdNET-ONNX, Perch)
  extract.py     -- dump per-clip teacher embeddings to distill/emb/{teacher}.npz
  build_targets.py -- fit a linear probe on your labels -> distill/targets/{teacher}.npz
  store.py       -- tiny npz (de)serialization keyed by itemid

Student training only ever reads a targets file, so swapping or ensembling
teachers never touches the model or the Trainer.
"""
