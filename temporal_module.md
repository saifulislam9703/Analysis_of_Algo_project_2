# Temporal Module: Multi-Step Snapshot Stacking

## 1. Problem — Why Single-Step Was Not Enough

The original model received one pair and predicted one step forward:

```
Input:   surge_t   [B, N]       ← current surge (observed)
         era5_tp1  [B, N, 3]    ← ERA5 forcing at t+1 (known ahead of time)
Output:  surge_tp1 [B, N]       ← predicted surge at t+1
```

This discards all temporal history. The model cannot learn how a storm is
evolving — it only sees a snapshot.

---

## 2. What SurgeTemporal Does (the reference)

SurgeTemporal stacks **T = 12 consecutive hourly frames** in a 3D tensor and
processes them through a 3D SwinTransformer:

```
Input  x : (B, 4, H, W, 12)   ← 4 channels (1 surge + 3 ERA5) × 12 time steps
                                  slot 0  = full observed surge (IC)
                                  slots 1–11 = zeros at interior, GT at boundary rim

Target y : (B, 1, H, W, 12)   ← slot 0 target is NaN (IC, not supervised)
                                  slots 1–11 = ground-truth surge
```

The 3D SwinTransformer uses **windowed attention** over (H, W, T_w=2) windows,
so each layer sees both spatial neighbours and 2 adjacent time steps at once.

---

## 3. Why We Cannot Use the Same Approach Directly

SurgeTemporal works on a **regular grid** (H × W pixels). SurgeGNN works on
an **unstructured triangular mesh** with 481,783 nodes — there is no (H, W)
tensor, so no 3D patch embedding or windowed 3D convolution is possible.

---

## 4. Our Solution: Fold T into Batch + Spatio-Temporal Transformer

### Core idea

Instead of a 3D tensor, we:

1. **Embed** all T snapshots independently with shared weights → `[B*T, N, D]`
2. **Run the spatial encoder/decoder** treating T as extra batch samples
3. **Apply factored attention** at the coarsest bottleneck (L3, only 700 nodes):
   - Even layers → spatial attention over 700 nodes (per time step)
   - Odd layers  → temporal attention over T frames (per node)

### Simple example (T=3 for clarity)

```
surge_history : [B, 3, N]     slot 0 = IC,  slot 1 = 0,  slot 2 = 0
era5_sequence : [B, 3, N, 3]  ERA5 at t, t+1, t+2

Step 1 — fold T into batch
    surge_flat : [B*3, N]       (treat each timestep as a separate graph)
    era5_flat  : [B*3, N, 3]

Step 2 — embed + encode  (shared GnBlock weights)
    h0  [B*3, N0, D]  →  skip0
    h1  [B*3, N1, D]  →  skip1
    h2  [B*3, N2, D]  →  skip2
    h3  [B*3, N3, D]           ← L3 bottleneck (N3 = 700)

Step 3 — SpatioTemporalTransformer (4 layers)
    Layer 0 (spatial):
        reshape [B*3, 700, D] → directly apply attention over 700 nodes
        each time step attends to all 700 spatial neighbours

    Layer 1 (temporal):
        reshape [B*3, 700, D] → [B*700, 3, D]
        each node attends to its T=3 time steps

    Layer 2 (spatial): same as layer 0
    Layer 3 (temporal): same as layer 1

    output: [B*3, 700, D]

Step 4 — decode  (shared GnBlock weights)
    unpool + skip at each level → [B*3, N0, D]

Step 5 — output head
    [B*3, N0, D] → Linear(D,1) → [B*3, N0]
    reshape       → [B, 3, N0]
    slice [:, 1:] → [B, 2, N0]   ← predictions at slots 1 and 2
```

For T=12 the output is `[B, 11, N0]` — 11 hourly forecasts from one pass.

---

## 5. SpatioTemporalTransformer Layer Detail

```python
class SpatioTemporalTransformer(nn.Module):
    """
    n_layers TransformerEncoderLayer instances, alternating:
      even i → spatial  attention:  h [B*T, N, D]  (apply directly)
      odd  i → temporal attention:  reshape to [B*N, T, D], apply, reshape back
    """
    def forward(self, h, T):
        # h : [B*T, N, D]
        BT, N, D = h.shape
        B = BT // T
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:                          # spatial
                h = layer(h)                        # [B*T, N, D]
            else:                                   # temporal
                h = h.reshape(B, T, N, D)
                h = h.permute(0, 2, 1, 3)          # [B, N, T, D]
                h = h.reshape(B*N, T, D)
                h = layer(h)                        # [B*N, T, D]
                h = h.reshape(B, N, T, D)
                h = h.permute(0, 2, 1, 3)          # [B, T, N, D]
                h = h.reshape(BT, N, D)
        return h
```

This is the graph analogue of SwinTransformer's alternating spatial and
temporal windows — the only difference is that "spatial" here means attending
over all N3=700 graph nodes instead of a (H_w × W_w) pixel patch.

---

## 6. Dataset Changes

Each sample is now a **sliding window of T=12 consecutive hours**:

```
surge_history : float32 [T, N]      slot 0 = IC (NaN→0),  slots 1..11 = 0
era5_sequence : float32 [T, N, 3]   ERA5 forcing at all T steps
surge_targets : float32 [T-1, N]    ground truth at slots 1..11, NaN on land
```

Window indexing in a single zarr year (T_year hours, T=12):

```
sample 0:  hours  [0,  1,  2, ..., 11]
sample 1:  hours  [1,  2,  3, ..., 12]
sample 2:  hours  [2,  3,  4, ..., 13]
  ...
```

Total windows per year = T_year − (T − 1) = T_year − 11.

---

## 7. Loss and Metrics

Multi-step MSE over all 11 forecast leads simultaneously:

```python
loss = MaskedMSELoss()(pred, surge_targets)   # pred/target: [B, 11, N]
```

NaN land nodes are excluded at every lead time with `torch.where`.

Per-lead RMSE is logged separately:

```
train/rmse_t01  ← RMSE at +1 h
train/rmse_t02  ← RMSE at +2 h
...
train/rmse_t11  ← RMSE at +11 h
```

---

## 8. Parameter Budget

Adding temporal attention costs **zero extra parameters** because the 4
transformer layers are simply split 2 spatial + 2 temporal (same total count).

| Component | Params |
|---|---|
| input_proj | 33,536 |
| node_emb ×4 | 68,608 |
| coarse_proj ×3 | 49,536 |
| enc (3 × GraphLevel) | 1,587,456 |
| SpatioTemporalTransformer (2 spatial + 2 temporal) | 793,088 |
| dec (3 × GraphLevel) | 1,587,456 |
| out_head | 129 |
| **Total** | **4,119,809 (~4.12 M)** |

---

## 9. Config (`configs/train_full.yaml`)

```yaml
seq_steps:      12   # T: slot 0 = IC, slots 1..11 = targets
time_interval:  1    # hours between slots
```

---

## 10. Key Files

| File | What Changed |
|---|---|
| `src/model/transformer.py` | Added `SpatioTemporalTransformer` |
| `src/model/hierarchy_unet.py` | New `forward(surge_history, era5_sequence)` with T folded into batch |
| `src/dataset.py` | Returns T-step windows instead of single (t, t+1) pairs |
| `src/loss.py` | `MaskedMSELoss` works for `[B, T-1, N]`; added `masked_rmse_per_lead` |
| `src/train.py` | New batch keys; per-lead-time WandB logging |
| `configs/train_full.yaml` | Added `seq_steps`, `time_interval` |
