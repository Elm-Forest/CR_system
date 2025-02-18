# A Web-Based System for Cloud Removal in Single-Temporal Multispectral Images Using Ensemble Learning

> This is my undergraduate graduation project. The following is an incomplete document for reference only.

<div>
  <img src="https://github.com/Elm-Forest/CR_system/raw/refs/heads/master/.github/imgs/s2_img2.png?raw=true" width="20%" alt="" style="display: inline-block">
	<img src="https://github.com/Elm-Forest/CR_system/raw/refs/heads/master/.github/imgs/sar2.png?raw=true" width="20%" alt="" style="display: inline-block">
  <img src="https://github.com/Elm-Forest/CR_system/raw/refs/heads/master/.github/imgs/predict2.png?raw=true" width="20%" alt="" style="display: inline-block">
</div>
Left to right: Cloud-covered Optical Image, SAR Image, Cloud-removed Optical Image.

## Installation

Python 3.7. is required

```shell
git clone ..
cd CR_system
pip install -r requirements.txt
```

Make sure you have a c++ project build environment ready

```shell
# install kernelconv2d, ref: https://github.com/xufangchn/GLF-CR#prerequisites--installation
cd ./glf_cr/FAC/kernelconv2d/
python setup.py clean
python setup.py install --user
```

## Prepare Data

SEN12MS-CR DATASET Ref: https://patricktum.github.io/cloud_removal/sen12mscr/

## Prepare Weights



| Model    | Download | Repo |
|----------|-----|-----|
| DSen2-CR <sup>[[1]](#refer-anchor-1)</sup> | [weight](https://drive.google.com/file/d/1L3YUVOnlg67H5VwlgYO9uC9iuNlq7VMg/view) |  https://github.com/xufangchn/GLF-CR   |
| GLF-CR <sup>[[2]](#refer-anchor-2)</sup>  | [weight](https://drive.google.com/file/d/11EYrrqLzlqrDgrJNgIW7IY0nSz_S5y9Z/view?usp=sharing) |  https://github.com/ameraner/dsen2-cr   |
| UnCRtainTS <sup>[[3]](#refer-anchor-3)</sup> | [weight](https://u.pcloud.link/publink/show?code=kZsdbk0Z5Y2Y2UEm48XLwOvwSVlL8R2L3daV) |   https://github.com/PatrickTUM/UnCRtainTS  |

```shell
cd weights
# Download the model for ensemble learning and move here
```

## Running

Before running, configure `utils/common.py` firstly

**Run as a Web System**

```shell
# config utils/common.py
cd CR_system/web_service
python main_web_service.py

```

**Run as a Test Case**

```shell
# config utils/common.py
cd CR_system
python test.py
```
## Reference
<span id="refer-anchor-1">
[1] Meraner, Andrea et al. “Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion.” Isprs Journal of Photogrammetry and Remote Sensing 166 (2020): 333 - 346.<br>
</span>
<span id="refer-anchor-2">
[2] Xu, Fang et al. “GLF-CR: SAR-enhanced cloud removal with global–local fusion.” ISPRS Journal of Photogrammetry and Remote Sensing (2022): n. pag.<br>
</span>
<span id="refer-anchor-3">
[3] P. Ebel, V. Garnot, M. Schmitt, J. Wegner and X. X. Zhu. UnCRtainTS: Uncertainty Quantification for Cloud Removal in Optical Satellite Time Series. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2023.
</span>