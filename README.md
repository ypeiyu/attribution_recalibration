# Re-Calibrating Feature Attributions for Model Interpretation

This code implements attribution re-calibration algorithm from the following paper:

> Peiyu Yang, Naveed Akhtar, Zeyi Wen, Mubarak Shah, and Ajmal Mian
>
> [Re-calibrating Feature Attributions for Model Interpretation](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Xqmlj18AAAAJ&sortby=pubdate&citation_for_view=Xqmlj18AAAAJ:kzcrU_BdoSEC)


## Introduction
Due to its desirable theoretical properties, path integration is a widely used scheme for feature attribution to interpret model predictions. However, these methods rely on taking absolute values of attributions to provide sensible explanations, which contradicts their premise and theoretical guarantee. We address this by computing an appropriate reference for the path integration scheme. Our scheme can be incorporated into the existing integral-based attribution methods. Extensive results show a marked performance boost for a range of integral-based attribution methods by enhancing them with our scheme.
![LPI](figs/attribution_recalibration.png)

## Prerequisites

- python 3.9.2
- matplotlib 3.5.1
- numpy 1.21.5
- pytorch 1.12.0
- torchvision 0.13.1

## Re-calibrating attributions

### Step 1: Preparing dataset.
```
dataset\DATASET
```

### Step 2: Preparing models.
```
pretrained_models\YOUR_MODEL
```

### Step 3: Re-calibrating attributions (IG_Uniform).

```
python main.py -attr_method=IG_Uniform -model resnet34 -dataset ImageNet -metric visualize -k 5 -bg_size 10
```

## Quantitatively Evaluations
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