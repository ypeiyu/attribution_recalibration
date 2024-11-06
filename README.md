# Re-Calibrating Feature Attributions for Model Interpretation (ICLR 2023 Spotlight)

This code implements attribution re-calibration algorithm from the following paper:

> Peiyu Yang, Naveed Akhtar, Zeyi Wen, Mubarak Shah, and Ajmal Mian
>
> [Re-calibrating Feature Attributions for Model Interpretation](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Xqmlj18AAAAJ&sortby=pubdate&citation_for_view=Xqmlj18AAAAJ:kzcrU_BdoSEC)


## Introduction
Due to its desirable theoretical properties, path integration is a widely used scheme for feature attribution to interpret model predictions. However, these methods rely on taking absolute values of attributions to provide sensible explanations, which contradicts their premise and theoretical guarantee. We address this by computing an appropriate reference for the path integration scheme. Our scheme can be incorporated into the existing integral-based attribution methods. Extensive results show a marked performance boost for a range of integral-based attribution methods by enhancing them with our scheme.
![attribution_recalibration](figs/attribution_recalibration.png)

## Prerequisites

- python 3.9.2
- matplotlib 3.5.1
- numpy 1.21.5
- pytorch 1.12.0
- torchvision 0.13.1
- tqdm 4.64.0

## Incorporated Attribution Methods
|    **Attribution Method**     |                                                                                                  **Source**                                                                                                   | Paper Link     |
|:-----------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------|
|           InputGrad           |                                 Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps [[Paper](https://arxiv.org/pdf/1312.6034.pdf)]                                  | Gradient-based |
|            GradCAM            | Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf)] | Gradient-based |
|          SmoothGrad           |                                                                                  SmoothGrad: removing noise by adding noise [[Paper](https://arxiv.org/pdf/1706.03825.pdf?source=post_page---------------------------)]                                                                                  | Gradient-based |
|           FullGrad            |                                                                         Full-Gradient Representation for Neural Network Visualization [[Paper](https://proceedings.neurips.cc/paper/2019/file/80537a945c7aaa788ccfcdf1b99b5d8f-Paper.pdf)]                                                                        | Gradient-based |
|              IG               |                                                                                    Axiomatic Attribution for Deep Networks [[Paper](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf)]                                                                                   | Gradient-based |
|          IG-Uniform           |                                                                            Visualizing the Impact of Feature Attribution Baselines [[Paper](https://distill.pub/2020/attribution-baselines/?utm_source=researcher_app&utm_medium=referral&utm_campaign=RESR_MRKT_Researcher_inbound)]                                                                           | Gradient-based |
|             IG-SG             |                                                                                  SmoothGrad: removing noise by adding noise                                                                                   | Integral-based |
|             IG-SQ             |                                                                       A Benchmark for Interpretability Methods in Deep Neural Networks                                                                        | Integral-based |
|              EG               |                                                    Improving performance of deep learning models with axiomatic attribution priors and expected gradients [[Paper](https://arxiv.org/pdf/1906.10670.pdf)]                                                    | Integral-based |
|              AGI              |                                                                  Explaining Deep Neural Network Models with Adversarial Gradient Integration [[Paper](https://www.ijcai.org/proceedings/2021/0396.pdf)]                                                                 | Integral-based |
|              LPI              |                                                                                    Local Path Integration for Attribution [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25422)]                                                                                    | Integral-based |
| Class-Contrastive Explanation |                                                                Why Not Other Classes?: Towards Class-Contrastive Back-Propagation Explanations [[Paper](https://openreview.net/pdf?id=X5eFS09r9hm)]                                                               | Integral-based |
|  Attribution Re-calibration   |                                                                         Re-calibrating Feature Attributions for Model Interpretation [[Paper](https://openreview.net/pdf?id=WUWJIV2Yxtp)]                                                                         | Integral-based |

## Incorporated Attribution Methods
- **Input Gradients** [[Paper](https://arxiv.org/pdf/1312.6034.pdf)]
- **Grad CAM** [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf)]
- **Smooth Gradients** [[Paper](https://arxiv.org/pdf/1706.03825.pdf?source=post_page---------------------------)]
- **Full-Gradients** [[Paper](https://proceedings.neurips.cc/paper/2019/file/80537a945c7aaa788ccfcdf1b99b5d8f-Paper.pdf)]
- **Integrated Gradients (IG)** [[Paper](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf)]
- **IG with Smooth Gradients** [[Paper](https://arxiv.org/pdf/1706.03825.pdf?source=post_page---------------------------)]
- **IG with Squared Gradients** [[Paper](https://proceedings.neurips.cc/paper/2019/file/fe4b8556000d0f0cae99daa5c5c5a410-Paper.pdf)]
- **IG with Uniform References** [[Paper](https://distill.pub/2020/attribution-baselines/?utm_source=researcher_app&utm_medium=referral&utm_campaign=RESR_MRKT_Researcher_inbound)]
- **Expected Gradients** [[Paper](https://arxiv.org/pdf/1906.10670.pdf)]
- **Adversarial Gradient Integration** [[Paper](https://www.ijcai.org/proceedings/2021/0396.pdf)]
- **Local Gradient Integration** [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25422)]
- **Class-Contrastive Back-Propagation Explanations** [[Paper](https://openreview.net/pdf?id=X5eFS09r9hm)]
- **Attribution Re-calibration** [[Paper](https://openreview.net/pdf?id=WUWJIV2Yxtp)]

## Re-calibrating attributions

### Step 1: Preparing dataset.
```
dataset\DATASET
```

### Step 2: Preparing models.
```
pretrained_models\YOUR_MODEL
```

### Step 3: Re-calibrating attributions (*IG Uniform*).

```
python main.py -attr_method=IG_Uniform -model resnet34 -dataset ImageNet -metric visualize -k 5 -bg_size 10
```

## Quantitatively evaluations
```
python main.py -attr_method=IG_Uniform -model resnet34 -dataset ImageNet -metric DiffID -k 5 -bg_size 10
```

## Bibtex
If you found this work helpful for your research, please cite the following papers:
```
@artical{yang2023recalibrating,
    title={Re-Calibrating Feature Attributions for Model Interpretation},
    author={Peiyu, Yang and Naveed, Akhtar and Zeyi, Wen and Mubarak, Shah and Ajmal, Mian},
    booktitle={International Conference on Learning Representations {ICLR}},
    year={2023}
}
```
```
@artical{yang2023local,
    title={Local Path Integration for Attribution},
    author={Peiyu, Yang and Naveed, Akhtar and Zeyi, Wen and Ajmal, Mian},
    booktitle={AAAI Conference on Artificial Intelligence {AAAI}},
    year={2023}
}
```
