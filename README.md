# RB-Modulation: Training-Free Personalization of Diffusion Models using Stochastic Optimal Control

Official PyTorch implementation of [**RB-Modulation: Training-Free Personalization of Diffusion Models using Stochastic Optimal Control**](https://arxiv.org/pdf/2405.17401).


<!-- [![Star on GitHub](https://img.shields.io/github/stars/LituRout/RB-Modulation.svg?style=social)](https://github.com/LituRout/RB-Modulation/stargazers) -->

Given reference images of preferred style or content, our method, **RB-Modulation**, offers a plug-and-play solution for (a) stylization with various prompts, and (b)
composition with reference content images while maintaining sample diversity and prompt alignment.

![teaser](./assets/web1.png)


## 🔥 Updates
- [x] **[2024.08.23]** [RB-Modulation](https://rb-modulation.github.io/) Code Release!
- [x] **[2024.05.29]** [Paper](https://arxiv.org/pdf/2405.17401) is published on arXiv!


## Installation

```
# Download pretrained models.
cd third_party/StableCascade/models
bash download_models.sh essential big-big bfloat16
cd ..

# Install dependencies following the original [StableCascade](https://github.com/Stability-AI/StableCascade/blob/master/inference/readme.md)
conda create -n rbm python==3.9
pip install -r requirements.txt
pip install jupyter notebook opencv-python matplotlib ffty

# Download [pre-trained CSD weights](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view) and put it under `third_party/CSD/checkpoint.pth`.

# Install LangSAM
pip install  git+https://github.com/IDEA-Research/GroundingDINO.git
pip install segment-anything==1.0
git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
pip install -e .
```
## Try it!
```commandline
jupyter notebook rb-modulation.ipynb
```

## Citation

```
@article{rout2024rbmodulation,
  title={RB-Modulation: Training-Free Personalization of Diffusion Models using Stochastic Optimal Control},
  author={Litu Rout and Yujia Chen and Nataniel Ruiz and Abhishek Kumar and Constantine Caramanis and Sanjay Shakkottai and Wen-Sheng Chu},
  journal={arXiv preprint arXiv:2405.17401},
  year={2024}
}
```

<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=LituRout/RB-Modulation&type=Date)](https://star-history.com/#LituRout/RB-Modulation&Date) -->

## Disclaimer
This is not an officially supported Google product
