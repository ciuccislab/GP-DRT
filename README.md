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

### Citation

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
