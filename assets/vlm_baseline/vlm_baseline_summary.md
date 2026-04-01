# VLM Baseline Summary

Addresses NjKC W3: "No baselines or VLM comparisons."

Two VLMs evaluated: **Gemini 3 Flash Preview** and **Qwen3-VL-32B-Instruct** (via OpenRouter).

---

## Teacher-Forced: Action Accuracy

64 samples/task, GT screenshots, no compounding errors.

| Task | Gemini | Qwen |
|------|--------|------|
| cell_lineage_tracking | 81.2% ±4.9% | 73.4% ±5.6% |
| neuron_tracking | 78.1% ±5.1% | 75.0% ±5.4% |
| multichannel_image_alignment | 42.2% ±6.1% | 48.4% ±6.3% |
| animal_limb_tracking | 39.1% ±6.1% | 37.5% ±6.0% |
| animal_behavioral_tracking | 35.9% ±6.0% | 25.0% ±5.4% |
| colored_dot_tracking | 29.7% ±5.7% | 18.8% ±4.9% |
| 3d_exploration_classification | 17.2% ±4.7% | 25.0% ±5.4% |
| spectral_plume_finding | 3.1% ±2.2% | 0.0% |
| road_network_construction | 1.6% ±1.5% | 1.6% ±1.6% |

SEs are bootstrap (10k resamples, n=64/task).

## Teacher-Forced: Placement@5px

Only tasks with canvas placement actions. BC (28M, step 36k) included.

| Task | BC | Gemini | Qwen |
|------|-----|--------|------|
| colored_dot_tracking | **94.1%** | 80.0% | 25.0% |
| cell_lineage_tracking | **54.8%** | 32.4% | 0.0% |
| neuron_tracking | **52.9%** | 28.0% | 4.0% |
| animal_limb_tracking | **29.7%** | 24.0% | 8.3% |
| multichannel_image_alignment | **20.7%** | 18.5% | 12.9% |

BC wins every task. VLM placement samples are small (5–34 per task).

## VLM Autoregressive: Success Rate

32 instances/task, closed-loop with scaffold (text state + 3 screenshots).

| Task | Gemini | Qwen |
|------|--------|------|
| cell_lineage_tracking | **81.2%** | 0% |
| animal_behavioral_tracking | **53.1%** | 0% |
| 3d_exploration_classification | **34.4%** | 0% |
| colored_dot_tracking | 0% | 0% |
| neuron_tracking | 0% | 0% |
| animal_limb_tracking | 0% | 0% |
| multichannel_image_alignment | 0% | 0% |
| road_network_construction | 0% | 0% |
| spectral_plume_finding | 0% | 0% |

Qwen: 0% on all 9 tasks (repetition loops, API failures). Gemini: 3/9 tasks with nonzero success.

---

Data: `data/revision_aggregate/all_results/results.json`, `tables.json`
Figures: `data/revision_aggregate/all_results/figures/`
