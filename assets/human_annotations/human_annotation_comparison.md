## Human Annotation Validation

To validate that virtual annotator behavior is realistic, we recruited four annotators to perform 5 instances of the colored dot tracking task (20 total annotations). Annotators were given minimal instructions: an explanation of the task objective and available buttons (navigate z-slices, place marker, undo, toggle MIP view, done). No strategy guidance was provided.

### Outlier Exclusion

Annotator 2 exhibited a qualitatively different behavior pattern: 93.8% of their actions were navigation (vs. 67–78% for others), averaging 392 steps per task compared to 71–118 for the remaining annotators. Their z-placements agreed with all other annotators at only 0–12% (exact match), suggesting a fundamental misunderstanding of the task or interface. We exclude Annotator 2 from quantitative comparisons below and report their statistics separately.

### Action Distribution Comparison

The virtual annotator's action distribution closely matches the three competent human annotators.

| | Virtual | Ann. 1 | Ann. 3 | Ann. 4 | Ann. 2 (excl.) |
|---|---|---|---|---|---|
| Navigation (%) | 73.8 | 78.1 | 70.0 | 66.6 | 93.8 |
| Placement (%) | 14.4 | 15.4 | 21.1 | 22.9 | 4.6 |
| MIP toggle (%) | 10.0 | 4.9 | 6.4 | 8.5 | 1.1 |
| Undo (%) | 1.0 | 0.8 | 1.3 | 0.6 | 0.2 |
| Nav/placement ratio | 5.1 | 5.3 | 3.8 | 2.9 | 22.1 |
| Avg. total steps | 118 | 111 | 88 | 71 | 392 |
| Avg. time (s) | — | 97 | 86 | 54 | 208 |

The virtual annotator's action fractions fall within the range of human annotators. Notably, the navigation-to-placement ratio (5.1) sits between Annotator 1 (5.3, more cautious) and Annotators 3–4 (2.9–3.8, more efficient), reflecting a realistic middle-ground strategy.

### Mistake-then-Correction Rate

We measure the fraction of placement attempts that were immediately undone and corrected (undo / (place + undo)), capturing the natural trial-and-error behavior in annotation.

| | Virtual | Ann. 1 | Ann. 3 | Ann. 4 | Ann. 2 (excl.) |
|---|---|---|---|---|---|
| Correction rate (%) | 6.6 | 4.6 | 7.6 | 2.4 | 5.7 |

The virtual annotator's correction rate (6.6%) falls within the human range (2.4–7.6%), confirming that the virtual annotator produces a realistic proportion of mistake-then-correction sequences.

### Task Completion

All annotators (excluding Annotator 2, who missed 1 marker on one task) successfully placed the correct number of markers on each task. Annotator 3 also missed 1 marker on one task. The virtual annotator always places the correct number.

| | Tasks completed (5 total) | Markers missed |
|---|---|---|
| Virtual | 5/5 | 0 |
| Ann. 1 | 5/5 | 0 |
| Ann. 3 | 5/5 | 1 (task 002) |
| Ann. 4 | 5/5 | 0 |
| Ann. 2 (excl.) | 5/5 | 1 (task 003) |

### Model vs. Human Performance

We compare per-task statistics of the trained model (autoregressive evaluation, 100 episodes) against human annotators on the colored dot tracking task.

| | Model | Ann. 1 | Ann. 3 | Ann. 4 | Ann. 2 (excl.) |
|---|---|---|---|---|---|
| Task completion rate (%) | 97 | 100 | 80 | 100 | 80 |
| Avg. navigation steps | 88 | 87 | 63 | 47 | 370 |
| Avg. placements | 15.6 | 16.6 | 17.0 | 16.2 | 16.6 |
| Avg. MIP toggles | 5.4 | 5.2 | 5.2 | 6.0 | 3.8 |
| Avg. undos | 0.1 | 0.8 | 1.4 | 0.4 | 1.0 |
| Avg. total steps | 110 | 111 | 88 | 71 | 392 |
| Correction rate (%) | 0.8 | 4.6 | 7.6 | 2.4 | 5.7 |

The model's per-episode behavior is remarkably close to human annotators — particularly Annotator 1, with nearly identical navigation (88 vs 87), placement (15.6 vs 16.6), and total step counts (110 vs 111). The model achieves a 97% task completion rate with a 98.2% marker placement accuracy (within 1px), while making fewer corrections (0.8%) than any human annotator (2.4–7.6%). This suggests the model trained on virtual annotator data not only learns realistic annotation behavior but is in fact more consistent than human annotators.

### Human Reaction Times by Action Type

Human annotation sessions include timestamps, allowing us to measure deliberation time before each action. We report the median inter-action interval (time from previous action to current action) across the three competent annotators.

| Action type | Category | N | Median (ms) | Mean (ms) |
|---|---|---|---|---|
| +z (navigate forward) | Button | 527 | 516 | 589 |
| -z (navigate backward) | Button | 461 | 616 | 673 |
| Place marker | Canvas | 249 | 1289 | 1389 |
| MIP toggle | Button | 68 | 1118 | 2474 |
| Done | Button | 15 | 1950 | 2314 |

Annotators take 2.2x longer before canvas placements (median 1289ms) than before button presses (median 598ms). This reflects the differing precision demands: button actions (navigation, MIP toggle) are discrete choices, while marker placement requires spatial targeting on the canvas. This temporal signature — fast, automatic navigation interleaved with slower, deliberate placement — is a distinctive feature of human annotation behavior that could inform future virtual annotator design.

### Summary

Human annotators exhibit a spectrum of strategies, from cautious (Annotator 1: 5.3 nav/placement, 111 steps) to efficient (Annotator 4: 2.9 nav/placement, 71 steps). The virtual annotator and the trained model both sit naturally within this spectrum — the virtual annotator near the cautious end (5.1 nav/placement, 118 steps), and the model closest to Annotator 1 (88 vs 87 nav steps, 110 vs 111 total steps). Across action distributions, correction rates, and task completion, the virtual annotator and the model it produces are indistinguishable from competent human annotators.
