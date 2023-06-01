# Re-calibrating Feature Attributions for Model Interpretation

Code for ICLR 2023 "Re-calibrating Feature Attributions for Model Interpretation" by Peiyu Yang, Naveed Akhtar, Zeyi Wen, Mubarak Shah and Ajmal Mian.

# Abstract

Trustworthy machine learning necessitates meticulous regulation of model reliance on non-robust features. In this paper, we proposed a framework to delineate such features by attributing model predictions to the input. Within this framework, the attributions of robust features exhibit certain consistency, while non-robust features are susceptible to attribution fluctuations. This suggests a strong correlation between model reliance on non-robust features and the smoothness of the marginal density of input samples. Hence, we propose to regularize the gradients of the marginal density w.r.t. the input features. We devise an efficient implementation of our regularization to address the potential numerical instability of the underlying optimization process. In contrast, we reveal that the baseline input gradient regularization smooths the implicit conditional or joint density, resulting in its limited robustness. Experiments validate the effectiveness of our technique through the mitigation of spurious correlation learned by the model and addressing feature leakage. We also demonstrate that our regularization enables the model to exhibit robustness against perturbations in pixel values, input gradients and density, enhancing its desirability for robust modeling.

# Prerequisites

- python 3.9.2
- matplotlib 3.5.1
- numpy 1.21.5
- pytorch 1.12.0
- torchvision 0.13.1
