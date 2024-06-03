# MO-VLN: A Multi-Task Benchmark for Open-set Zero-Shot Vision-and-Language Navigation

This is a MindSpore implementation of the [paper](https://arxiv.org/abs/2306.10322).


## Installing Dependencies
- Installing the simulator following [here](https://mligg23.github.io/MO-VLN-Site/Simulation%20Environment%20API.html).

- Installing [MindSpore](https://www.mindspore.cn/install/en).
```
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0/MindSpore/unified/x86_64/mindspore-2.0.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- Installing [mindformers](https://gitee.com/mindspore/mindformers).
```
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```


## Setup
Clone the repository and install other requirements:
```
git clone https://github.com/liangcici/MO-VLN.git
cd MO-VLN/
pip install -r requirements.txt
```

### Setting up dataset
- Generate data for ObjectNav (goal-conditioned navigation given a specific object category).
```
python data_preprocess/gen_objectnav.py --map_id 3
```
map_id indicates specific scene: `{3: Starbucks; 4: TG; 5: NursingRoom}`.


## Usage
Run models with FBE:

- For ObjectNav:
```
python zero_shot_eval.py --sem_seg_model_type clip --map_id 3
```


## Citation
```
@article{liang2023mo,
  title={MO-VLN: A Multi-Task Benchmark for Open-set Zero-Shot Vision-and-Language Navigation},
  author={Liang, Xiwen and Ma, Liang and Guo, Shanshan and Han, Jianhua and Xu, Hang and Ma, Shikui and Liang, Xiaodan},
  journal={arXiv preprint arXiv:2306.10322},
  year={2023}
}
```