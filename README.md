# Policy Contrastive Decoding

Official implementation of the paper "[Policy Contrastive Decoding](https://arxiv.org/abs/2505.13255)".

> **Note**: We are doing our best to improve this work. If you have any questions or suggestions, please feel free to create an issue in this repo or contact us at shihan.wu.koorye@outlook.com.

## News

- [2025/5/20] The code is released.

## Introduction

> **Abstract** Generalist robot policies, or robotic foundation models, hold immense potential to enable flexible, general-purpose and dexterous robotic systems. Despite their advancements, our empirical experiments reveal that existing robot policies are prone to learning spurious correlations from pre-training trajectories, adversely affecting their generalization capabilities during inference. To tackle this, we propose a novel Policy Contrastive Decoding (PCD) approach, which redirects the robot policy’s focus toward object-relevant visual clues by contrasting action probability distributions derived from original and object-masked visual inputs. As a training-free method, our PCD can be used as a plugin to improve different types of robot policies without needing to finetune or access model weights. We conduct extensive experiments on top of three open-source robot policies, including the autoregressive policy OpenVLA and the diffusion-based policies Octo and Pi-0. The obtained results in both simulation and real-world environments prove PCD’s flexibility and effectiveness, e.g., PCD enhances the state-of-the-art policy Pi-0 by 8% in the simulation environment and by 108% in the real-world environment.

![Policy Contrastive Decoding](examples/method.png)

## Experiments

### Overall Performance

**Simpler**

![Simpler Results](examples/simpler_results.png)

**Real-world**

![Real-world Results](examples/real_results.png)

### Performance on Different Factors

**Simpler**

![Simpler Factors](examples/simpler_factors.png)

**Real-world**

![Real-world Factors](examples/real_factors.png)

## Videos

### Simulated Environments

**Baseline**

| Pick Coke Can | Move Near | Carrot Plate | Eggplant Basket |
| :-----------: | :-------: | :---------: | :-------: |
| ![Pick Coke Can](examples/videos/main/simpler/baseline/pick_coke_can.gif) | ![Move Near](examples/videos/main/simpler/baseline/move_near.gif) | ![Carrot Plate](examples/videos/main/simpler/baseline/carrot_plate.gif) | ![Stack Cube](examples/videos/main/simpler/baseline/eggplant_basket.gif) |

**+PCD**

| Pick Coke Can | Move Near | Carrot Plate | Eggplant Basket |
| :-----------: | :-------: | :---------: | :-------: |
| ![Pick Coke Can](examples/videos/main/simpler/pcd/pick_coke_can.gif) | ![Move Near](examples/videos/main/simpler/pcd/move_near.gif) | ![Carrot Plate](examples/videos/main/simpler/pcd/carrot_plate.gif) | ![Stack Cube](examples/videos/main/simpler/pcd/eggplant_basket.gif) |

### Real-world Environments

**Baseline**

| Pick Ball | Move Near | Banana Plate | Stack Cube |
| :-------: | :-------: | :---------: | :-------: |
| ![Pick Ball](examples/videos/main/real/baseline/pick_ball.gif) | ![Move Near](examples/videos/main/real/baseline/move_near.gif) | ![Banana Plate](examples/videos/main/real/baseline/banana_plate.gif) | ![Stack Cube](examples/videos/main/real/baseline/stack_cube.gif) |

**+PCD**

| Pick Ball | Move Near | Banana Plate | Stack Cube |
| :-------: | :-------: | :---------: | :-------: |
| ![Pick Ball](examples/videos/main/real/pcd/pick_ball.gif) | ![Move Near](examples/videos/main/real/pcd/move_near.gif) | ![Banana Plate](examples/videos/main/real/pcd/banana_plate.gif) | ![Stack Cube](examples/videos/main/real/pcd/stack_cube.gif) |

## Running

Install all dependencies.

```bash
conda create -n pcd python=3.10
conda activate pcd
bash scripts/install_dependencies.sh
```

Running evaluation on simpler.

```bash
bash scripts/default/run.sh
```

## Acknowledgements

Our work is built upon the following open-source projects: [SimplerEnv](https://github.com/simpler-env/SimplerEnv), [OpenVLA](https://github.com/openvla/openvla), [Octo](https://github.com/octo-models/octo), [Open Pi-0](https://github.com/allenzren/open-pi-zero), [Grounded SAM2](https://github.com/IDEA-Research/Grounded-SAM-2), [YOLO World](https://github.com/AILab-CVC/YOLO-World), [SED](https://github.com/xb534/SED), [Inpaint Anything](https://github.com/geekyutao/Inpaint-Anything).
We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.