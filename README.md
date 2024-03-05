# FactorVAE
### Adapted by Tom Frew

From the [original repository](https://github.com/x7jeon8gi/FactorVAE):
> An unofficial PyTorch implementation of FactorVAE, proposed in "FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns" by Duan et al.(https://ojs.aaai.org/index.php/AAAI/article/view/20369)

This is the version of FactorVAE I have reproduced on my machine. I am still looking to find the researchers' original parameters, but they are currently:

- Number of Stocks: 300 (from the all)
- Epochs: 50
- Learning Rate: 0.0005
- Batch Size: 30
- Sequence Length: 20
- Number of Factors: 8
- Hidden Space Size: 20
- Seed: 42

## Sample results

```
                                                  risk
excess_return_without_cost mean               0.000641
                           std                0.016776
                           annualized_return  0.152517
                           information_ratio  0.589304
                           max_drawdown      -0.300378
excess_return_with_cost    mean              -0.000845
                           std                0.016765
                           annualized_return -0.201010
                           information_ratio -0.777203
                           max_drawdown      -0.539686
```

![Backest executed on my machine](./backtest_plotly/local_backtest.png)
