# DSTSANet
DSTSANet: A Decoupled Spatio-Temporal Sparse Attention Adaptive Network for Traffic Flow Forecasting

![Structure](structure_13.pdf)

## Ablation: Frequency-Branch Encoder Sharing

The default DSTSANet applies **different temporal encoders** to the two frequency
branches:
- Low-frequency branch → **Temporal Attention**
- High-frequency branch → **Temporal Causal Convolution**

To verify that this heterogeneous design is necessary (rather than just the
choice of one particular encoder), two ablation variants are provided. Both
variants apply the **same** temporal encoder to both branches, but with
**independent parameters** (no weight sharing), so that the comparison isolates
the effect of encoder heterogeneity while keeping the model capacity comparable.

| Variant              | Low-branch encoder | High-branch encoder |
| -------------------- | ------------------ | ------------------- |
| `none` (default)     | Temporal Attention | Temporal Conv       |
| `attn_shared`        | Temporal Attention | Temporal Attention  |
| `conv_shared`        | Temporal Conv      | Temporal Conv       |

### Run a single ablation experiment

```bash
python main.py \
    --traffic_file ./data/PeMSD7/PeMSD7.npz \
    --adj_file ./data/PeMSD7/adj.npy \
    --tem_adj_file ./data/PeMSD7/tem_adj.npy \
    --model_file ./work_dirs/PeMSD7/PeMSD7.pth \
    --log_file ./log/PeMSD7/log_train.txt \
    --ablation attn_shared
```

`--ablation` accepts `none` (default), `attn_shared`, `conv_shared`.

### Generate commands for all dataset × variant combinations

```bash
# Print all commands to stdout
python run_ablation.py

# Save them to a shell script
python run_ablation.py --save ablation.sh
bash ablation.sh

# Actually run them sequentially
python run_ablation.py --execute
```

### Implementation

The ablation encoders live in `model/ablation.py`:
- `DualEncoderSharedAttn` — both branches use `TemAttn`, independent weights.
- `DualEncoderSharedConv` — both branches use `TemConv`, independent weights.

`DSTSANet` (in `model/models.py`) selects the encoder class based on the
`ablation` argument. The rest of the model (embedding, sparse STF extraction,
adaptive fusion, prediction head) is unchanged across variants.