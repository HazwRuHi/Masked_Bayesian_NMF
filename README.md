# Masked_Bayesian_NMF
This repository hosts the Code for the paper "Core-periphery Detection Based on Masked Bayesian Non-negative Matrix Factorization" accepted by IEEE Transactions on Computational Social Systems (TCSS).


## Dependencies
- numpy 1.20+
- scikit-learn 0.24+

## How to run
To perform core peirphery detection on synthetic networks, you can run:
```
python main.py --network_size=5000 --epochs=20 --a=5 --b=10 --sigma_overline=1 --sigma_hat=1 --mu_hat=1 --k=32
```

## Citation
If our work could help your research, please cite: [Core-Periphery Detection Based on Masked Bayesian Nonnegative Matrix Factorization](https://ieeexplore.ieee.org/abstract/document/10399942) 

```
@article{10399942,
  author={Wang, Zhonghao and Yuan, Ru and Fu, Jiaye and Wong, Ka-Chun and Peng, Chengbin},
  journal={IEEE Transactions on Computational Social Systems}, 
  title={Core–Periphery Detection Based on Masked Bayesian Nonnegative Matrix Factorization}, 
  year={2024},
  volume={11},
  number={3},
  pages={4102-4113},
  keywords={Bayes methods;Matrix decomposition;Symmetric matrices;Measurement;Complex networks;Brain modeling;Biological system modeling;Complex networks;core–periphery detection;nonnegative matrix factorization (NMF)},
  doi={10.1109/TCSS.2023.3347406}
}
```
