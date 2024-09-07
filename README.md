
<div align="center">
<h1>RB-Modulation: Training-Free Personalization of Diffusion Models using Stochastic Optimal Control</h1>

<a href='https://rb-modulation.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/pdf/2405.17401'><img src='https://img.shields.io/badge/ArXiv-Preprint-red'></a>
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-red)](https://huggingface.co/spaces/fffiloni/RB-Modulation)
[![GitHub](https://img.shields.io/github/stars/google/RB-Modulation?style=social)](https://github.com/google/RB-Modulation)
</div>

Official PyTorch implementation of [**RB-Modulation: Training-Free Personalization of Diffusion Models using Stochastic Optimal Control**](https://arxiv.org/pdf/2405.17401).


<!-- [![Star on GitHub](https://img.shields.io/github/stars/google/RB-Modulation.svg?style=social)](https://github.com/google/RB-Modulation/stargazers) -->

Given reference images of preferred style or content, our method, **RB-Modulation**, offers a plug-and-play solution for (a) stylization with various prompts, and (b)
composition with reference content images while maintaining sample diversity and prompt alignment.

![teaser](./assets/web1.png)


## 🔥 Updates
- [x] **[2024.09.02]** RB-Modulation [Demo](https://huggingface.co/spaces/fffiloni/RB-Modulation) on Hugging Face! Thanks [Sylvain Filoni](https://huggingface.co/fffiloni).
- [x] **[2024.08.23]** RB-Modulation [Code](https://github.com/google/RB-Modulation) Release!
- [x] **[2024.05.29]** [Paper](https://arxiv.org/pdf/2405.17401) is published on arXiv!


## 📥 Installation

```
# Download pretrained models.
cd third_party/StableCascade/models
bash download_models.sh essential big-big bfloat16
cd ..

# Install dependencies following the original [StableCascade](https://github.com/Stability-AI/StableCascade/blob/master/inference/readme.md)
conda create -n rbm python==3.9
pip install -r requirements.txt
pip install jupyter notebook opencv-python matplotlib ftfy

# Download [pre-trained CSD weights](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view) and put it under `third_party/CSD/checkpoint.pth`.

# Install LangSAM
pip install  git+https://github.com/IDEA-Research/GroundingDINO.git
pip install segment-anything==1.0
git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
pip install -e .
```

## 🚀 Try it!
```commandline
jupyter notebook rb-modulation.ipynb
```

## 🤗 Gradio interface
We also support a Gradio <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> interface for better experience:
[Web demonstration](https://huggingface.co/spaces/fffiloni/RB-Modulation)🔥
```bash
# Make sure you have the docker correctly setup.
git clone https://huggingface.co/spaces/fffiloni/RB-Modulation
cd RB-Modulation
python app.py
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

[![Star History Chart](https://api.star-history.com/svg?repos=google/RB-Modulation&type=Date)](https://star-history.com/#google/RB-Modulation&Date) -->

## Disclaimer
This is not an officially supported Google product.
