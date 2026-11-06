# Logistic regression results

Produced by `python experiments/logistic_regression.py` (seed 0).

## Binary: breast cancer (80/20 split)

| Model | Test accuracy |
|---|---|
| minigrad (SGD, 1000 full-batch steps) | 0.9737 |
| scikit-learn LogisticRegression | 0.9825 |

## Multi-class: softmax regression on Fashion-MNIST

| Train accuracy | Test accuracy | Training time (5 epochs) |
|---|---|---|
| 0.8512 | 0.8347 | 1.4 s |
