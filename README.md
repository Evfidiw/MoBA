# MoBA: Mixture of Bi-directional Adapter for Multi-modal Sarcasm Detection



## Overview

This repository contains the offical implementation of paper:

_MoBA: Mixture of Bi-directional Adapter for Multi-modal Sarcasm Detection_, ACM MM 2024



## Experiments

1. Install requirements.

    ````
    pip install -r requirements.txt
    ````



2. Download the data.

    Plesae visit [here](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection) to download the image data and put them into folder `data/dataset_image`.

     

3. Download pre-train models.
   
    Please donwload the [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) locally.



4. Run the scripts.

    ```
    # MMSD2.0
    sh scripts/2train.sh
    
    # MMSD
    sh scripts/train.sh
    ```



## Reference

If you find this project useful for your research, please consider citing the following paper.



## Acknowledge

- We sincerely thank the [MultiCLIP](https://github.com/JoeYing1019/MMSD2.0?tab=readme-ov-file), as most of our code is based on it.
- We also greatly appreciate [HKE](https://github.com/less-and-less-bugs/HKEmodel) and [DynRT](https://github.com/TIAN-viola/DynRT).