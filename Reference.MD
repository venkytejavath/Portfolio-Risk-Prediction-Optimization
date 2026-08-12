# Reference

## Primary Reference

This project's methodology (ticker selection, Dirichlet-based weight-generation scheme, annualized volatility target construction, and Random Forest-based risk classification/regression) follows and extends the framework introduced in:

> Rathi, V., Kshirsagar, M., and Ryan, C. (2024). **Enhancing Portfolio Performance: A Random Forest Approach to Volatility Prediction and Optimization.** In *Proceedings of the 16th International Conference on Agents and Artificial Intelligence (ICAART 2024) — Volume 3*, pages 1278–1285.
> DOI: [10.5220/0012464600003636](https://doi.org/10.5220/0012464600003636)
> ISBN: 978-989-758-680-4 · ISSN: 2184-433X
> Published under **CC BY-NC-ND 4.0** by SCITEPRESS – Science and Technology Publications, Lda.

### BibTeX

```bibtex
@inproceedings{rathi2024enhancing,
  title     = {Enhancing Portfolio Performance: A Random Forest Approach to Volatility Prediction and Optimization},
  author    = {Rathi, Vedant and Kshirsagar, Meghana and Ryan, Conor},
  booktitle = {Proceedings of the 16th International Conference on Agents and Artificial Intelligence (ICAART 2024) - Volume 3},
  pages     = {1278--1285},
  year      = {2024},
  publisher = {SCITEPRESS -- Science and Technology Publications, Lda.},
  doi       = {10.5220/0012464600003636},
  isbn      = {978-989-758-680-4},
  issn      = {2184-433X}
}
```

**Extensions made in this project (beyond the original paper):** XGBoost as a second model family alongside Random Forest, Optuna-based Bayesian hyperparameter tuning (in place of the original's exhaustive grid search), SHAP-based interpretability analysis (summary and interaction plots, in place of tree-based importance alone), and a modern portfolio theory optimizer built with `scipy.optimize` (SLSQP) supporting flexible, user-defined allocation constraints.

---

## Full Bibliography

As cited in Rathi, Kshirsagar & Ryan (2024):

- Aboussalah, A. M. and Lee, C.-G. (2020). Continuous control with stacked deep dynamic recurrent reinforcement learning for portfolio optimization. *Expert Systems with Applications*, 140:112891.
- Bartram, S. M., Branke, J., De Rossi, G., and Motahari, M. (2021). Machine learning for active portfolio management. *The Journal of Financial Data Science*, 3(3):9–30.
- Cervelló-Royo, R. and Guijarro, F. (2020). Forecasting stock market trend: A comparison of machine learning algorithms. *Finance, Markets and Valuation*, 6(1):37–49.
- Chen, W., Zhang, H., Mehlawat, M. K., and Jia, L. (2021). Mean–variance portfolio optimization using machine learning-based stock price prediction. *Applied Soft Computing*, 100:106943.
- Choijil, E., Méndez, C. E., Wong, W.-K., Vieito, J. P., and Batmunkh, M.-U. (2022). Thirty years of herd behavior in financial markets: A bibliometric analysis. *Research in International Business and Finance*, 59:101506.
- Christensen, K., Siggaard, M., and Veliyev, B. (2021). A machine learning approach to volatility forecasting. Available at SSRN.
- Dunford, R., Su, Q., and Tamang, E. (2014). The Pareto principle.
- Elton, E. J. and Gruber, M. J. (1997). Modern portfolio theory, 1950 to date. *Journal of Banking & Finance*, 21(11-12):1743–1759.
- Eom, C., Kaizoji, T., and Scalas, E. (2019). Fat tails in financial return distributions revisited: Evidence from the Korean stock market. *Physica A: Statistical Mechanics and its Applications*, 526:121055.
- Erasmus, A., Brunet, T. D., and Fisher, E. (2021). What is interpretability? *Philosophy & Technology*, 34(4):833–862.
- Ertenlice, O. and Kalayci, C. B. (2018). A survey of swarm intelligence for portfolio optimization: Algorithms and applications. *Swarm and Evolutionary Computation*, 39:36–52.
- Gao, Y., Gao, Z., Hu, Y., Song, S., Jiang, Z., and Su, J. (2021). A framework of hierarchical deep q-network for portfolio management. In *ICAART (2)*, pages 132–140.
- Hu, Y., Ni, J., and Wen, L. (2020). A hybrid deep learning approach by integrating LSTM-ANN networks with GARCH model for copper price volatility prediction. *Physica A: Statistical Mechanics and its Applications*, 557:124907.
- Hwang, E. and Hong, W.-T. (2021). A multivariate HAR-RV model with heteroscedastic errors and its WLS estimation. *Economics Letters*, 203:109855.
- Idrees, S. M., Alam, M. A., and Agarwal, P. (2019). A prediction approach for stock market volatility based on time series data. *IEEE Access*, 7:17287–17298.
- Kim, H. Y. and Won, C. H. (2018). Forecasting the volatility of stock price index: A hybrid model integrating LSTM with multiple GARCH-type models. *Expert Systems with Applications*, 103:25–37.
- Kobets, V. and Savchenko, S. (2022). Building an optimal investment portfolio with Python machine learning tools. *CEUR Workshop Proceedings*.
- Kumar, I., Dogra, K., Utreja, C., and Yadav, P. (2018). A comparative study of supervised machine learning algorithms for stock market trend prediction. In *2018 Second International Conference on Inventive Communication and Computational Technologies (ICICCT)*, pages 1003–1007. IEEE.
- Leung, M. F., Jawaid, A., Ip, S.-W., Kwok, C.-H., and Yan, S. (2023). A portfolio recommendation system based on machine learning and big data analytics. *Data Science in Finance and Economics*, 3(2):152–165.
- Liu, Y. and Tsyvinski, A. (2021). Risks and returns of cryptocurrency. *The Review of Financial Studies*, 34(6):2689–2727.
- Ma, Y., Han, R., and Wang, W. (2020). Prediction-based portfolio optimization models using deep neural networks. *IEEE Access*, 8:115393–115405.
- Ma, Y., Han, R., and Wang, W. (2021). Portfolio optimization with return prediction using deep learning and machine learning. *Expert Systems with Applications*, 165:113973.
- Mancuso, F. (2022). The efficient market hypothesis and trading AI advancements: an overview.
- Pav, S. E. (2021). *The Sharpe Ratio: Statistics and Applications*. CRC Press.
- Reinholtz, N., Fernbach, P. M., and De Langhe, B. (2021). Do people understand the benefit of diversification? *Management Science*, 67(12):7322–7343.
- Sharkawy, A.-N. (2020). Principle of neural network and its main types. *Journal of Advances in Applied & Computational Mathematics*, 7:8–19.
- Sharpe, W. F. (1998). The Sharpe ratio. *Streetwise – the Best of the Journal of Portfolio Management*, 3:169–85.
- Vidal, A. and Kristjanpoller, W. (2020). Gold volatility prediction using a CNN-LSTM approach. *Expert Systems with Applications*, 157:113481.
- Walther, T., Klein, T., and Bouri, E. (2019). Exogenous drivers of bitcoin and cryptocurrency volatility – a mixed data sampling approach to forecasting. *Journal of International Financial Markets, Institutions and Money*, 63:101133.
- Wen, F., Gong, X., and Cai, S. (2016). Forecasting the volatility of crude oil futures using HAR-type models with structural breaks. *Energy Economics*, 59:400–413.
- Zhang, C., Zhang, Y., Cucuringu, M., and Qian, Z. (2022). Volatility forecasting with machine learning and intraday commonality. *arXiv preprint arXiv:2202.08962*.

---

## Software & Libraries Used

- Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.
- Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions (SHAP). In *Advances in Neural Information Processing Systems 30*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12:2825–2830.
- Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17:261–272.
- McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference*. (`pandas`)
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3):90–95.
- `yfinance` — Yahoo Finance market data API wrapper (Ran Aroussi).
- `seaborn` — statistical data visualization (Michael Waskom).
- `ydata-profiling` — automated exploratory data analysis reporting (YData).

---

## License Note

The reference paper is distributed under **CC BY-NC-ND 4.0**:
- **Attribution** — credit must be given to the original authors.
- **NonCommercial** — the paper itself may not be used for commercial purposes.
- **NoDerivatives** — the paper (PDF) may not be modified or adapted; only unmodified copies may be shared.

This license covers the *paper document itself*, not the code in this repository. The notebooks in this project are original implementations that apply and extend the paper's described methodology; they do not reproduce or derive from the paper's text or figures.