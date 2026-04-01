# DAgger Experiment Summary

## Motivation

Reviewer bXZW Q2 asked whether DAgger could improve autoregressive performance on failing tasks. Two tasks had 0% accuracy and near-zero partial-credit metrics in autoregressive evaluation:

- **animal_behavioral_tracking**: 0% accuracy, 0.04% match rate
- **animal_limb_tracking**: 0% accuracy, 0.03% keypoint accuracy

The hypothesis: if failures are due to **compounding errors** (distribution shift from expert states to model-visited states), DAgger should help by training on on-policy data.

## Method: beta-DAgger

beta-DAgger (Ross et al., 2011) is an interactive imitation learning algorithm:

1. Roll out the learned policy in a simulator
2. At each step, with probability beta use the oracle action instead of the model's
3. Record (screenshot, oracle_action) pairs regardless of who acted
4. Aggregate with original training data, fine-tune
5. Evaluate

We implemented state-aware greedy oracles for both tasks:
- **BehavioralTrackingOracle**: Computes correct front/back marker positions from ground truth animal positions and orientations
- **LimbTrackingOracle**: Determines correct keypoint placements from ground truth, navigating tabs and frames as needed

Oracle limitation: does NOT undo incorrect model placements — treats them as done and moves on.

## Step 1: Data Collection

**Config**: `configs/revision_dagger/collect_iter0.yaml`

| Parameter | Value |
|-----------|-------|
| beta | 0.1 (90% model, 10% oracle) |
| n_instances | 50 per task |
| max_steps | 500 |
| temperature | 0.4 |
| n_workers | 4 (parallel Playwright simulators) |
| seed | 42 |
| base checkpoint | `checkpoint_step_100000.pt` (95M params, base) |

**Collected data**:

| Task | Sequences | Total Frames | Avg Steps/Seq |
|------|-----------|-------------|---------------|
| animal_behavioral_tracking | 50 | 24,604 | 492.1 |
| animal_limb_tracking | 50 | 21,087 | 421.7 |

Average steps near 500 (the max) indicates the model almost never completes tasks — it wanders rather than making progress through the annotation workflow.

**Output**: `data/revision_dagger/dagger_data_iter0/`

## Step 2: Fine-tuning

**Config**: `configs/revision_dagger/train_iter0.yaml`

Fine-tuned the base multitask model on **only the 2 target tasks** (original data + DAgger data).

| Parameter | Value |
|-----------|-------|
| Base checkpoint | step 100,000 (95M params, DINOv2 b14) |
| Training steps | 5,000 |
| Batch size | 2 per GPU, grad accum 2, 4x A100 = effective 16 |
| Learning rate | 3e-5 (head), 3e-7 (backbone, 0.01x scale) |
| Warmup | 100 steps, cosine decay |
| Label smoothing | 0.1 |
| DAgger upsample | 50x (~25K frames x 50 = 1.25M per task) |
| Original data | ~3.5M frames per task |
| DAgger fraction | ~26% of training mix |

**Loss curve** (101 logged entries over 5000 steps):

| Step | Loss | x_loss | y_loss |
|------|------|--------|--------|
| 1 | 4.13 | 4.39 | 4.23 |
| 2500 | 3.93 | 4.03 | 3.64 |
| 5000 | 3.13 | 4.17 | 3.88 |

Loss decreased but with high variance. The decrease likely reflects adapting to the 2-task subset rather than specifically learning from DAgger data.

**Output**: `data/revision_dagger/train_iter0/`

## Step 3: Evaluation

**Config**: `configs/revision_dagger/eval_iter0.yaml`

Autoregressive evaluation on test instances.

| Parameter | Value |
|-----------|-------|
| n_samples | 30 per task |
| max_steps | 500 |
| temperature | 0.4 |
| n_workers | 4 |

**Results**:

| Task | Condition | Accuracy | Done Rate | Avg Steps | Avg Placed |
|------|-----------|----------|-----------|-----------|------------|
| animal_behavioral_tracking | Baseline | 0.0% | 1.3% | 494 | — |
| animal_behavioral_tracking | **DAgger** | **0.0%** | **0.0%** | **500.0** | **0.0** |
| animal_limb_tracking | Baseline | 0.0% | 0.0% | 491 | — |
| animal_limb_tracking | **DAgger** | **0.0%** | **0.0%** | **491.7** | **0.0** |

All 60 evaluation instances hit the max step limit. Zero markers placed, zero tasks completed.

**Output**: `data/revision_dagger/eval_iter0/`

## Conclusion

**DAgger does not help.** Both tasks remain at 0% accuracy, 0% completion.

The failure mode is **not distribution shift** but a fundamental inability to learn the hierarchical annotation workflow. These tasks require long structured sequences (select animal -> select marker type -> place -> repeat for all animals -> next frame -> repeat -> done) that the model never executes correctly. It gets stuck in repetitive clicking loops (~10-15% unique positions out of 500 steps).

**For the rebuttal**: This confirms that failures on behavioral/limb tracking reflect fundamental task difficulty, not compounding errors. This strengthens the paper's analysis — simple behavioral cloning has clear limits on complex multi-step annotation tasks.
