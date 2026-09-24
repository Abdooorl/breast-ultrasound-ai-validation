## Table 3. Prespecified confidence-based abstention threshold sweep

| Threshold | Overall coverage | Malignant coverage | Malignant abstained, n | Accepted accuracy | Accepted malignant sensitivity* | Accepted specificity* | Accepted malignant FN | Accepted FP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.9649 | 1.0000 | 0 | 0.4613 | 0.4333 | 0.9262 | 51 | 42 |
| 0.55 | 0.9151 | 0.9778 | 2 | 0.4592 | 0.4318 | 0.9311 | 50 | 37 |
| 0.60 | 0.8624 | 0.9444 | 5 | 0.4652 | 0.4471 | 0.9325 | 47 | 34 |
| 0.65 | 0.8053 | 0.9111 | 8 | 0.4691 | 0.4634 | 0.9316 | 44 | 32 |
| 0.70 | 0.7189 | 0.8556 | 13 | 0.4949 | 0.4935 | 0.9348 | 39 | 27 |
| 0.75 | 0.6486 | 0.8000 | 18 | 0.4989 | 0.4861 | 0.9326 | 37 | 25 |
| 0.80 | 0.5637 | 0.6667 | 30 | 0.5195 | 0.5500 | 0.9354 | 27 | 21 |
| 0.85 | 0.4553 | 0.6111 | 35 | 0.5434 | 0.5818 | 0.9297 | 23 | 18 |
| 0.90 | 0.3397 | 0.4778 | 47 | 0.6121 | 0.6279 | 0.9365 | 16 | 12 |
| 0.95 | 0.1742 | 0.1889 | 73 | 0.8403 | 0.7059 | 0.9608 | 5 | 4 |

*Note:* Overall coverage is the proportion of all 683 external images receiving an accepted prediction. Malignant coverage is the proportion of the 90 malignant images receiving an accepted prediction. Metrics marked with an asterisk were calculated only among accepted cases. FN = false negative; FP = false positive. Abstained observations were reported separately and were not counted as correct predictions. The complete prespecified threshold sweep is shown rather than selecting a single threshold after observing external-test labels.
