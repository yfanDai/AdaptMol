# AdaptMol

A few-shot molecular property prediction framework based on **Prototypical Networks** with adaptive molecular attribute fusion.

---

## Project Structure

```
AdaptMol/
├── main.py                         # Main training entry point (meta-learning loop)
├── requirements.txt                 # Pip environment dependencies
├── README.md
│
├── src/                            # Core source code
│   ├── models/
│   │   ├── gnn_models.py           # GNN encoder + attributes_GNN with cross-attention
│   │   └── prototype.py            # Prototypical network loss and metric functions
│   ├── data/
│   │   ├── dataset.py              # MyDataset: few-shot episode sampler
│   │   └── graph_data_pre.py       # SMILES → PyG graph data conversion
│   └── utils/
│       ├── utils.py                # ChemBERTa SMILES attribute generation
│       └── logger.py               # Logging utility
│
├── scripts/                        # Utility / analysis scripts
│   ├── data_split.py               # Dataset train/val/test splitting
│   ├── interpretation_fp.py        # Fingerprint interpretation
│   ├── interpretation_graph.py     # Graph-level interpretation (MCTS)
│   ├── MCTS_explain_model.py       # MCTS explainability module
│   └── positive_data_process.py    # Positive sample preprocessing
│
├── data/                           # Dataset CSV files
│   ├── tox21.csv
│   ├── sider.csv
│   ├── muv.csv
│   └── tdc.csv
│
├── model_gin/                      # Pre-trained GNN weights
│   └── gin_supervised_contextpred.pth
│
└── cache/                          # Auto-generated at runtime (ChemBERTa embeddings)
```

---

## Datasets

The following CSV files are included in the `data/` directory. Each file contains a `smiles` column and task label columns.

| Dataset | File | Tasks |
|---------|------|-------|
| Tox21 | `data/tox21.csv` | NR-AR, NR-AR-LBD, NR-AhR, ... (12 tasks) |
| SIDER | `data/sider.csv` | SIDER1–SIDER27 (27 tasks) |
| MUV | `data/muv.csv` | MUV-466, MUV-548, ... (17 tasks) |
| TDC | `data/tdc.csv` | bbb_martins, cyp2c9_veith, ... (10 tasks) |

CSV format example:
```
smiles,NR-AR,NR-AR-LBD,...
CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0,0,...
```

## Pre-trained GNN Weights


The pre-trained GIN weights are included in `model_gin/gin_supervised_contextpred.pth`, used when `--pretrained_bool True` (default).

---

## Environment Setup

```bash
pip install -r environment.txt
```

---

## Training

### Training with ChemBERTa SMILES Embeddings

```bash
python main.py \
  --dataset tox21 \
  --attribute_type smiles \
  --smiles_attributes Chemberta \
  --gpu 0
```

### Training Without Attributes (Pure GNN)

```bash
python main.py \
  --dataset tox21 \
  --with_attr False \
  --gpu 0
```

---

## Output

Training logs and model checkpoints are saved to `--log_dir` (default: `./log/`):

```
log/
├── run<timestamp>.log    # Training log with AUC / F1 / PR-AUC metrics
└── run<timestamp>.pth    # Best model checkpoint (saved by best AUC)
```

---

## Evaluation Metrics

- **AUROC** (Area Under ROC Curve)
- **F1-score**
- **PR-AUC** (Area Under Precision-Recall Curve)

Results are tracked across all test tasks, and best results per metric are recorded with early stopping (patience = 100 evaluation steps).

---


## Notes

- The first run will auto-generate a molecular graph cache file (`data/<dataset>.pickle`) to speed up subsequent runs.
- Attribute features are cached in `cache/` after first computation.
- `model_gin/gin_supervised_contextpred.pth` is the pre-trained GIN weight file used when `--pretrained_bool True`.
- `data_split.py` contains a hardcoded absolute path — update it before use.
