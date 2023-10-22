# UMB-Net
 Recently, CycleGAN-based methods have been widely applied to the visual task of unsupervised image dehazing and have achieved significant results. However, most existing CycleGAN-based methods ignore that the input of the dehazing generator contains two different distributions of data: hazy images from the dataset and hazy images from the rehazing generator. The presence of these disparate data distributions can often lead to confusion in the learning process of the dehazing generator, consequently limiting the final dehazing performance. Moreover, Using loss functions to constrain the network for reconstructing clear images is an indirect constraint., making it difficult to compensate for the missing high-frequency information, such as textures and structures, in the features extracted by the network from hazy images. To address these issues, in this paper, we propose a Unsupervised Multi-Branch with High-Frequency Components Enhancement Network (UMB-Net) which is mainly composed of a Unsupervised Multi-Branch Dehazing Network (UMDN) and a High-Frequency Components Enhancement (HFCE) module. Specifically, UMDN is proposed to build a novel unsupervised dehazing network with a shared encoding and three decoding branches. The shared encoding reduces parameter and computation complexity while enhancing feature consistency for both decoding branches. The three decoding branches form an unsupervised dehazing structure, which addresses the issue of learning confusion of the generator. Furthermore, based on a key observation—hazy images and their corresponding clear images exhibit only subtle differences in high-frequency information, the HFCE module is designed to simultaneously encode RGB images and their corresponding high-frequency images and adaptively fuse them in the decoding layer. This effectively compensates for the missing high-frequency information in the network. Experimental results on challenging benchmark datasets demonstrate the superiority of our UMB-Net over SOTA unsupervised image dehazing methods. 
<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/thislzm/PSMB-Net/">
    <img src="images/UMB.png" alt="Logo" width="800" height="500">
  </a>
  <h3 align="center">Unsupervised Multibranch Dehazing Networks</h3>
  <p align="center">
  <a href="https://github.com/thislzm/PSMB-Net/">
    <img src="images/HFEM.png" alt="Logo" width="1000" height="640">
  </a>
  </p>
  <h3 align="center">Hihg-Frequency Components Enhanced module</h3>

  <p align="center">
    Partial Siamese Networks with Multiscale Bi-codec Information Fusion Module for Remote Sensing Single Image Dehazing
    <br />
    <a href="https://github.com/thislzm/PSMB-Net"><strong>Exploring the documentation for PSMB-Net »</strong></a>
    <br />
    <br />
    <a href="https://github.com/thislzm/PSMB-Net">Check Demo</a>
    ·
    <a href="https://github.com/thislzm/PSMB-Net/issues">Report Bug</a>
    ·
    <a href="https://github.com/thislzm/PSMB-Net/issues">Pull Request</a>
  </p>

</p>

## Contents

- [Dependencies](#dependences)
- [Filetree](#filetree)
- [Pretrained Model](#pretrained-weights-and-dataset)
- [Train](#train)
- [Test](#test)
- [Clone the repo](#clone-the-repo)
- [Qualitative Results](#qualitative-results)
  - [Results on Statehaze1k-Thin remote sensing Dehazing Challenge testing images:](#results-on-statehaze1k-thin-remote-sensing-dehazing-challenge-testing-images)
  - [Results on Statehaze1k-Moderate remote sensing Dehazing Challenge testing images:](#results-on-statehaze1k-moderate-remote-sensing-dehazing-challenge-testing-images)
  - [Results on Statehaze1k-Thick remote sensing Dehazing Challenge testing images:](#results-on-statehaze1k-thick-remote-sensing-dehazing-challenge-testing-images)
  - [Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images:](#results-on-ntire-2021-nonhomogeneous-dehazing-challenge-testing-images)
  - [Results on RESIDE-Outdoor Dehazing Challenge testing images:](#results-on-reside-outdoor-dehazing-challenge-testing-images)
- [Copyright](#copyright)
- [Thanks](#thanks)

### Dependences

1. Pytorch 1.8.0
2. Python 3.7.1
3. CUDA 11.7
4. Ubuntu 18.04

### Filetree

```
├── README.md
├── /PSMB-Net/
|  ├── train.py
|  ├── test.py
|  ├── Model.py
|  ├── Model_util.py
|  ├── perceptual.py
|  ├── train_dataset.py
|  ├── test_dataset.py
|  ├── utils_test.py
|  ├── make.py
│  ├── Parameter_test.py
│  ├── Loss.py
│  │  ├── __init__.py
│  ├── /datasets_train/
│  │  ├── /hazy/
│  │  ├── /clean/
│  ├── /datasets_test/
│  │  ├── /hazy/
│  │  ├── /clean/
│  ├── /output_result/
├── LICENSE.txt
└── /images/
```

### Pretrained Weights and Dataset

Download our model weights on Baidu cloud disk: https://pan.baidu.com/s/1dePHGG4MYvyuLW5rZ0D8VA?pwd=lzms

Download our test datasets on Baidu cloud disk: https://pan.baidu.com/s/1HK1oy4SjZ99N-Dh-8_s0hA?pwd=lzms


### Train

```shell
python train.py -train_batch_size 4 --gpus 0 --type 5
```

### Test

 ```shell
python test.py --gpus 0 --type 5
 ```

### Clone the repo

```sh
git clone https://github.com/thislzm/PSMB-Net.git
```

### Qualitative Results

#### Results on Statehaze1k-Thin remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/thin.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-Moderate remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/moderate.png" style="display: inline-block;" />
</div>

#### Results on Statehaze1k-Thick remote sensing Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/thick.png" style="display: inline-block;" />
</div>

#### Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/nhhaze.png" style="display: inline-block;" />
</div>

#### Results on RESIDE-Outdoor Dehazing Challenge testing images
<div style="text-align: center">
<img alt="" src="/images/outdoor.png" style="display: inline-block;" />
</div>




### Copyright

The project has been licensed by MIT. Please refer to for details. [LICENSE.txt](https://github.com/thislzm/PSMB-Net/LICENSE.txt)

### Thanks


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)


<!-- links -->
[your-project-path]:thislzm/PSMB-Net
[contributors-shield]: https://img.shields.io/github/contributors/thislzm/PSMB-Net.svg?style=flat-square
[contributors-url]: https://github.com/thislzm/PSMB-Net/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/thislzm/PSMB-Net.svg?style=flat-square
[forks-url]: https://github.com/thislzm/PSMB-Net/network/members
[stars-shield]: https://img.shields.io/github/stars/thislzm/PSMB-Net.svg?style=flat-square
[stars-url]: https://github.com/thislzm/PSMB-Net/stargazers
[issues-shield]: https://img.shields.io/github/issues/thislzm/PSMB-Net.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/thislzm/PSMB-Net.svg
[license-shield]: https://img.shields.io/github/license/thislzm/PSMB-Net.svg?style=flat-square
[license-url]: https://github.com/thislzm/PSMB-Net/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian
