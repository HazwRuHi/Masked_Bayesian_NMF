# Masked_Bayesian_NMF
This repository hosts the Code for the paper "Core-periphery Detection Based on Masked Bayesian Non-negative Matrix Factorization" accepted at IEEE Transactions on Computational Social Systems (TCSS).


## Dependencies
- numpy 1.20+
- scikit-learn 0.24+

## How to run
To perform core peirphery detection on synthetic networks, you can run:
```
python main.py --network_size=5000 --epochs=20 --a=5 --b=10 --sigma_overline=1 --sigma_hat=1 --mu_hat=1 --k=32
```
