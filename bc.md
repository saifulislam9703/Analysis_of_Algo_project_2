# Boundary Condition (BC) and Initial Condition (IC)

## 1. What Problem This Solves

Without BC injection the model receives:

```
surge_history slot 0 : full observed surge (IC) ← known
surge_history slot 1 : all zeros               ← unknown
...
surge_history slot 11: all zeros               ← unknown
```

The model must predict 11 hours from the IC alone using only ERA5 wind/pressure
forcing. For long leads this becomes hard because the model has no information
about where the storm-surge signal is entering the domain from the open ocean.

BC injection provides the **ground-truth surge at open-ocean boundary nodes**
for every future time slot — giving the model the "incoming wave" at the
seaward edge at each step, which it can then propagate inward.

---

## 2. How SurgeTemporal Handles BC (the reference)

SurgeTemporal works on a **regular 603 × 934 pixel grid**. Its boundary is
simply the **outer 3-pixel rectangular rim** of that grid:

```
┌──────────────────────────┐  ← rim (3 px wide, GT injected here)
│  ┌────────────────────┐  │
│  │                    │  │
│  │  interior (zeros)  │  │
│  │                    │  │
│  └────────────────────┘  │
└──────────────────────────┘  ← rim
```

For each future slot k = 1..11:
- **Rim pixels**      → receive ground-truth CORA value
- **Interior pixels** → receive 0 (unknown)

`boundary_width = 3` → ~9,186 rim pixels → **1.6% of the grid**.

This design is trivial on a regular grid but has no direct equivalent on an
unstructured mesh because there is no rectangular structure to take a "rim" of.

---

## 3. What "Open Boundary" Means in ADCIRC

In the ADCIRC coastal ocean model, open boundaries are **explicitly listed in
the fort.14 mesh file** as elevation-specified boundary segments — the seaward
edges of the mesh where the model domain is truncated against the open ocean.
Tidal and surge forcing is injected there from external data.

The fort.14 format stores them as:

```
NOPE   = number of open boundary segments
NETA   = total open boundary nodes
NVELL  = nodes per segment
IBTYPE = boundary type (e.g. 3 = elevation-specified)
NBDV   = node numbers along the segment (in order)
```

We do not have the original fort.14 for this mesh subset, so we must infer
the open boundary from geometry.

---

## 4. Our Domain: Northern USA (35–46°N, 77–55°W)

```
          46°N ─────────────────────────── (lat_max)
               |  Gulf of Maine   shallow|
               |  coast (depth ≈ 7 m)   |
               |                        |
               |    continental shelf   |
               |                        |
               |                        |
          35°N ─────────────────────────── (lat_min)
               |                        |
              lon_min=-77°W          lon_max=-55°W

  West edge  (lon≈-77°W):  US coastline     depth ≈ -16 m  → land boundary
  North edge (lat≈46°N) :  Gulf of Maine    depth ≈   7 m  → shallow coastal
  South edge (lat≈35°N) :  open Atlantic    depth ≈ 867 m  → OPEN OCEAN ✓
  East edge  (lon≈-55°W):  deep Atlantic    depth ≈5435 m  → OPEN OCEAN ✓
```

Storms propagate **northward** from the open Atlantic; the storm-surge signal
enters the domain via the **south** and **east** truncation boundaries.

---

## 5. Why Depth > 200 m Was Wrong

The original `build_hierarchy.py` labelled a node as OBC if it was on the
topological mesh boundary **AND** its depth exceeded 200 m.

```python
# build_hierarchy.py (original heuristic — DO NOT USE)
node_type[is_boundary & (depth > 200.0)] = NODE_OBC   # 188 nodes
```

Problems:
- Missed 258 shelf nodes at the south edge that are at 50–200 m depth but are
  still part of the open-ocean boundary
- Captured 52 deep nodes elsewhere on the boundary that are NOT on the
  south/east open-ocean face
- A depth threshold identifies **how deep a node is**, not **which face of
  the domain it sits on** — those are different questions

---

## 6. The Correct Definition

A node is an **open-ocean boundary (OBC) seed** if it satisfies ALL of:

1. On the topological mesh boundary (an edge that belongs to exactly one
   triangle — the mesh is truncated here)
2. Located on the **south face** (lat < lat_min + 0.5°) OR the **east face**
   (lon > lon_max − 0.5°)
3. depth > 0 m (in water — not intertidal or dry land)

### Simple example with 5 boundary nodes

```
Node  lat    lon     depth   is_south?  is_east?  OBC seed?
A     35.1  -60.0   3200 m     yes        no          ✓
B     35.3  -58.0   4800 m     yes        no          ✓
C     35.5  -55.5   5300 m     no         yes         ✓
D     45.8  -70.0      6 m     no         no          ✗  (north, shallow)
E     40.0  -77.0    -12 m     no         no          ✗  (west, intertidal)
```

Nodes A, B, C are seeds. D is excluded (north, shallow). E is excluded (west,
intertidal — this is the coastline).

---

## 7. K-Hop Expansion (the "rim")

After selecting seeds we expand K=2 hops on the L0 mesh graph to form a
buffer zone analogous to SurgeTemporal's 3-pixel rim.

```
Seed nodes (394)
  ↓ 1 hop: add immediate graph neighbours
  ↓ 2 hops: add neighbours of neighbours
Final mask: 1,253 nodes  (0.26% of mesh)
```

### Why K=2?

| Method | Boundary nodes | % of domain |
|---|---|---|
| SurgeTemporal rim (3 px) | 9,186 | 1.6% |
| Old depth>200m | 188 | 0.04% |
| Geographic seeds (K=0) | 394 | 0.08% |
| Geographic seeds + K=2 BFS | 1,253 | 0.26% |

The mesh is much coarser at the deep open-ocean boundary than a regular pixel
grid, so fewer nodes cover the same physical extent. K=2 gives a physically
appropriate buffer region.

---

## 8. BC Injection in the Dataset

For each sample window of T=12 hours:

```python
# Slot 0: full IC (all nodes, NaN→0)
surge_history[0] = norm_surge(anomaly[t0])
nan_to_num(surge_history[0], nan=0.0)

# Slots 1..11
for k in range(1, T):
    surge_history[k] = 0.0           # default: interior unknown
    
    if bc_mask is not None:
        # inject GT only at open-ocean boundary nodes
        gt = norm_surge(anomaly[t0 + k])
        nan_to_num(gt, nan=0.0)
        surge_history[k][bc_mask] = gt[bc_mask]
    
    surge_targets[k-1] = norm_surge(anomaly[t0 + k])   # NaN kept for loss
```

**What the model sees at each node for slot k > 0:**

| Node type | Value in surge_history[k] |
|---|---|
| BC mask = True (boundary) | Ground-truth normalised surge |
| BC mask = False (interior) | 0.0 (unknown) |
| Land / always-dry | 0.0 (NaN→0 at boundary, 0 at interior) |

**What the model predicts:**

```
pred : [B, 11, N]   ← all nodes, all 11 future steps
loss : MaskedMSE over interior+boundary ocean nodes only (NaN land excluded)
```

The loss is computed **everywhere including boundary nodes** — the model must
both honour the injected boundary values AND extrapolate the interior correctly.

---

## 9. Ablation: Disabling BC

Set `use_boundary_bc: false` in the config (or omit `bc_mask_path`) to run
without BC injection. Slots 1..11 are all zeros everywhere. This is the
"no-boundary" ablation equivalent to SurgeTemporal's `use_boundary_bc: false`
experiments.

---

## 10. Generating the Mask

Run once before training:

```bash
python scripts/build_bc_mask.py \
    --mesh_zarr /orange/zhe.jiang-aistorm/ml_data/region_zarr_jiang/northern_usa/mesh_info.zarr \
    --hierarchy hierarchy.npz \
    --out       bc_mask.npz \
    --hops      2 \
    --south_margin 0.5 \
    --east_margin  0.5
```

Output `bc_mask.npz` contains:

| Key | Shape | Description |
|---|---|---|
| `l0_bc_mask` | `bool [N0]` | Full BC mask (seeds + K-hop expansion) |
| `obc_seeds` | `bool [N0]` | Geographic seeds only (no expansion) |
| `n_boundary` | scalar | Count of BC nodes |
| `hops` | scalar | K used |
| `south_margin` | scalar | Degrees from lat_min |
| `east_margin` | scalar | Degrees from lon_max |

---

## 11. Config (`configs/train_full.yaml`)

```yaml
bc_mask_path:    /blue/zhe.jiang/shared/saiful.islam/SurgeGNN/bc_mask.npz
use_boundary_bc: true    # false = no-BC ablation
```

---

## 12. Key Files

| File | Role |
|---|---|
| `scripts/build_bc_mask.py` | Generates `bc_mask.npz` from mesh geometry |
| `src/dataset.py` | Loads mask, injects GT at boundary nodes for slots 1..T-1 |
| `configs/train_full.yaml` | `bc_mask_path`, `use_boundary_bc` |
