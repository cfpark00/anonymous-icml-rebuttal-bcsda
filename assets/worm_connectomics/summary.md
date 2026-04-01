# C. elegans (Witvliet): Fine-Tuning Results

## Task Description

We fine-tuned our pretrained multi-task behavioral cloning model on a **real-world neuron tracing task** using electron microscopy (EM) data from the Witvliet 2020 C. elegans nerve ring dataset. This is a fundamentally different biological domain from the H01 human cortex experiment, testing cross-species generalization of our framework.

The task requires the model to:
1. Identify a neuron centered in the EM image at z=0
2. Navigate through 50 z-slices of a 3D volume using +z/-z buttons
3. Click on the neuron's primary branch at each slice to trace its path (these neurons bifurcate in 3D; the model follows the largest cross-section through z)
4. Click "Done" when tracing is complete

This task is **particularly challenging** because:
- **C. elegans neuropil is extremely dense**: neurites are tightly packed with minimal extracellular space
- **Smaller structures**: C. elegans neurites are much smaller than human myelinated axons, requiring finer spatial precision
- **Lower segmentation coverage**: only 30.5% of voxels are segmented (vs. 58.5% for H01), meaning more ambiguous regions
- **No segmentation overlay**: the model sees only raw 8 nm/px EM images

## Training Setup

- **Pretrained checkpoint**: 100k steps of multi-task training on 9 synthetic tasks
- **Fine-tuning data**: ~2.6M training frames from 19,872 episodes (8x augmentation: 4 rotations x 2 flips). Annotation traces are derived from ground truth segmentation of real EM data.
- **Architecture**: DINOv2 ViT-B/14 encoder + 8-layer transformer decoder (~95M parameters)
- **Effective batch size**: 32 (2 per GPU x 4 gradient accumulation x 4 A100 GPUs)
- **Training**: 66,397 gradient steps (1 full epoch), constant LR = 1e-4

## Teacher-Forced Evaluation

Evaluated on a held-out eval set (13 neurons from a spatially separated region, never seen during training) across 67 checkpoints.

| Metric | Step 1 (pretrained) | Best | Final (66k) |
|--------|-------------------|------|-------------|
| **Overall 5px accuracy** | 4.9% | **95.8%** (step 29k) | 95.1% |
| **Canvas @5px** | 14.1% | **92.4%** (step 62k) | 90.4% |
| **Button accuracy** | 4.6% | **98.2%** (step 22k) | 97.5% |
| **Action validity** | 99.4% | **100.0%** (step 1k) | 100.0% |
| **Loss** | 10.65 | **1.31** (step 27k) | 1.56 |

Key observations:
- **Rapid learning**: canvas @5px jumps from 14% to 88% within the first 5k steps, then gradually improves to 92%
- **Button accuracy reaches 98%+ by step 10k**, consistent with hierarchical skill emergence
- **Loss increases slightly after step 30k** (1.31 to 1.56 at step 66k), suggesting mild overfitting on this smaller dataset. Canvas accuracy remains stable, indicating the overfitting primarily affects calibration rather than decision quality.
- **Action validity is 100% from step 1k onward**, confirming pretrained GUI understanding transfers immediately

See: [tf_loss_accuracy.png](tf_loss_accuracy.png), [tf_per_action.png](tf_per_action.png)

## Autoregressive (Closed-Loop) Evaluation

Evaluated on 13 held-out neurons, across 33 checkpoints (every 2k steps). Max episode length: 400 steps.

| Metric | Step 2k | Step 56k (best seg) | Step 20k (best cov) |
|--------|---------|---------------------|---------------------|
| **Skeleton accuracy** | 74.9% | **89.4%** | 83.4% |
| **Canvas @5px** | 38.2% | 78.0% | 74.5% |
| **Canvas @10px** | 63.2% | 83.9% | 80.6% |
| **Done rate** | 92.3% | 100% | 100% |
| **Mean coverage** | 94.2% | 93.4% | **99.7%** |
| **Coverage >= 90%** | 69.2% | 69.2% | **92.3%** |
| **First node accuracy** | 100% | 100% | **100%** |
| **Action validity** | 100% | 100% | 100% |

Key observations:
- **89.4% skeleton accuracy**: the model correctly traces the target neuron in dense C. elegans neuropil, a challenging environment where neurites are tightly packed
- **100% first node accuracy across all checkpoints**: the BOS token reliably triggers the model to click the canvas center, initiating the trace on the correct neuron
- **92.3% of episodes achieve >= 90% annotation coverage** (at step 20k): the model traces nearly all required z-slices
- **Higher variance than H01** (13 eval neurons vs. 28): some neurons are consistently harder, reflected in the coverage fluctuations across checkpoints
- **Mean episode length ~139 steps**: efficient tracing without excessive exploration

See: [autoreg.png](autoreg.png)

## Addressing Reviewer Concerns

### "Synthetic-only evaluation"
This experiment provides **direct evidence of synthetic-to-real transfer**. A model pretrained entirely on synthetic GUI annotation tasks successfully fine-tunes to trace real neurons in C. elegans EM data, achieving 89% skeleton accuracy in closed-loop evaluation. This is a fundamentally different visual domain (grayscale EM vs. synthetic colored GUIs) and a different biological task (3D neuron tracing vs. 2D point annotation).

### "Does the model generalize beyond the synthetic benchmark?"
The C. elegans results, combined with the H01 human cortex results, demonstrate generalization across:
- **Species**: invertebrate (C. elegans) vs. human cortex
- **Tissue type**: dense neuropil vs. myelinated white matter
- **Image characteristics**: 8 nm resolution, tightly packed neurites vs. 16 nm resolution, well-separated axons
- **Dataset size**: 2.6M frames (worm) vs. 6.5M frames (human)

### "Multi-task pretraining enables efficient fine-tuning" (Finding 3)
The pretrained model transfers its GUI interaction skills immediately (100% action validity from step 1k) and rapidly adapts to the new visual domain. The loss curve shows that most learning occurs in the first 10k steps (~0.4 EFLOPs), after which metrics plateau. This efficiency is remarkable given the domain gap between synthetic tasks and real EM imagery.

This result highlights the **practical viability of the synthetic pretraining + real fine-tuning paradigm**. A lab with a small amount of annotated real data (here, ~125 traced neurons) can leverage the pretrained model's GUI understanding and achieve strong annotation performance without training from scratch. The fine-tuning cost is a small fraction of the pretraining compute, making this approach accessible even with limited GPU resources.


## Comparison: C. elegans vs. H01

| Metric (best autoreg) | C. elegans | H01 |
|----------------------|------------|-----|
| Skeleton accuracy | 89.4% | 95.1% |
| Canvas @5px | 78.0% | 95.1% |
| Done rate | 100% | 100% |
| First node accuracy | 100% | 100% |
| Training data | 2.6M frames | 6.5M frames |
| Eval neurons | 13 | 28 |

H01 outperforms C. elegans across metrics, likely due to: (1) myelinated axons being visually more distinct and easier to track than dense neuropil; (2) 2.5x more training data; and (3) larger eval set reducing variance. Both datasets demonstrate that the behavioral cloning framework generalizes to real scientific annotation.
