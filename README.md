# Project
GP-DRT: Gaussian Process Distribution of Relaxation Times

This repository contains some of the source code for the paper "The Gaussian Process Distribution of Relaxation Times: A Machine Learning Tool for the Analysis and Prediction of Electrochemical Impedance Spectroscopy Data" <u>https://doi.org/10.1016/j.electacta.2019.135316</u>, which is also available in the [docs](docs) folder.

# Introduction
GP-DIP is our newly developed approach that is able to obtain both the DRT mean and covariance from the EIS data, it can also predict the DRT and the imaginary part of the impedance at frequencies not previously measured. The most important point is that the parameters that define the GP-DRT model can be selected rationally by maximizing the experimental evidence. The GP-DRT approach is tested with both synthetic experiments and “real” experiments, where the GP-DRT model can manage considerable noise, overlapping timescales, truncated data, and inductive features.

# Dependencies
`numpy`

`scipy`
 
`matplotlib`


# Tutorials

* **simple_ZARC_model.ipynb**: one ZARC model based synthetic EIS data for GP-DRT.

* **truncated_ZARC_model.ipynb**: truncated ZARC model at high frequency for GP-DRT prediction

* **double_ZARC_model.ipynb**: test the GP-DRT for overlapping frequency behavior

# Citation

```
@article{liu2019gaussian,
  title={The Gaussian process distribution of relaxation times: A machine learning tool for the analysis and prediction of electrochemical impedance spectroscopy data},
  author={Liu, Jiapeng and Ciucci, Francesco},
  journal={Electrochimica Acta},
  pages={135316},
  year={2019},
  publisher={Elsevier}
}
```

# References
1. Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools. Electrochimica Acta, 184, 483-499.[doi.org/10.1016/j.coelec.2018.12.003](https://doi.org/10.1016/j.coelec.2018.12.003)
1. Ciucci, F. (2018). Modeling electrochemical impedance spectroscopy. Current Opinion in Electrochemistry, 132-139.
2. Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A  Bayesian and hierarchical Bayesian approach. Electrochimica Acta, 167, 439-454.

