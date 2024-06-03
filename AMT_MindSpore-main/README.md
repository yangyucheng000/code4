# $\rm{[MindSpore-phase3]}$ $AMT$

本项目包含了以下论文的mindspore实现：

> **AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation**<br>
> [Zhen Li](https://paper99.github.io/)<sup>\*</sup>, [Zuo-Liang Zhu](https://nk-cs-zzl.github.io/)<sup>\*</sup>, [Ling-Hao Han](https://scholar.google.com/citations?user=0ooNdgUAAAAJ&hl=en), [Qibin Hou](https://scholar.google.com/citations?hl=en&user=fF8OFV8AAAAJ&view_op=list_works), [Chun-Le Guo](https://scholar.google.com/citations?hl=en&user=RZLYwR0AAAAJ),  [Ming-Ming Cheng](https://mmcheng.net/cmm)<br>
> (\* denotes equal contribution) <br>
> Nankai University <br>
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2023<br>

[[Paper](https://arxiv.org/abs/2304.09790)] [[Project Page](https://nk-cs-zzl.github.io/projects/amt/index.html)]   [[Web demos](#web-demos)]

文章官方版本仓库链接: [MCG-NKU/AMT: Official code for "AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation" (CVPR2023) (github.com)](https://github.com/MCG-NKU/AMT)

目前已经完成推理部分以及训练部分大部分代码的mindspore转化

## 正在进行中的工作

-  完整代码的mindspore实现

## Dependencies and Installation
>python 3.8 <br>
>cuda: 11.6 <br>
>mindspore: 2.2.11 

1. Clone Repo

   ```bash
   git clone https://github.com/Men1scus/AMT_MindSpore.git
   ```

2. Create Conda Environment and Install Dependencies

   ```bash
   conda env create -f environment.yaml
   conda activate amt
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.11/MindSpore/unified/x86_64/mindspore-2.2.11-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
3. Download pretrained models for demos from [Pretrained Models](#pretrained-models) and place them to the `pretrained` folder

## Quick Demo

**Note that the selected pretrained model (`[CKPT_PATH]`) needs to match the config file (`[CFG]`).**

 > Creating a video demo, increasing $n$ will slow down the motion in the video. (With $m$ input frames, `[N_ITER]` $=n$ corresponds to $2^n\times (m-1)+1$ output frames.)


 ```bash
 python demos/demo_2x.py -c [CFG] -p [CKPT] -n [N_ITER] -i [INPUT] -o [OUT_PATH] -r [FRAME_RATE]
 # e.g. [INPUT]
 # -i could be a video / a regular expression / a folder contains multiple images
 # -i demo.mp4 (video)/img_*.png (regular expression)/img0.png img1.png (images)/demo_input (folder)

 # e.g. a simple usage
 python demos/demo_2x.py -c cfgs/AMT-S.yaml -p pretrained/amt-s.ckpt -n 6 -i assets/quick_demo/img0.png assets/quick_demo/img1.png

 ```

 + Note: Please enable `--save_images` for saving the output images (Save speed will be slowed down if there are too many output images)
 + Input type supported: `a video` / `a regular expression` / `multiple images` / `a folder containing input frames`.
 + Results are in the `[OUT_PATH]` (default is `results/2x`) folder.

 ## Pretrained Models
> These pretrained models, presented in the `.ckpt` format, originated from transforming a `.pth` file.
<p id="Pretrained"></p>

<table>
<thead>
  <tr>
    <th> Dataset </th>
    <th> :link: Download Links </th>
    <th> Config file </th>
    <th> Trained on </th>
    <th> Arbitrary/Fixed </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>AMT-S</td>
    <th>
    [<a href="https://pan.baidu.com/s/1SvJ1xcFj2RNnI9O4rmo2sw?pwd=u4st">Baidu Cloud</a>] 
    [<a href="https://drive.google.com/file/d/18vlxqeHdYECdvPB-JhsM4KfjvffVVOkm/view?usp=drive_link">Google Driver</a>]</th>
    <th> [<a href="cfgs/AMT-S.yaml">cfgs/AMT-S</a>] </th>
    <th>Vimeo90k</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>AMT-L</td>
    <th>
    [<a href="https://pan.baidu.com/s/1RVm5i_XIaizGHaoXzoqVhg?pwd=xh0e">Baidu Cloud</a>]
    [<a href="https://drive.google.com/file/d/1eFDU5QSeBuQ0ker_5lvLLBc8-jhBohIx/view?usp=drive_link">Google Driver</a>]</th>
    <th> [<a href="cfgs/AMT-L.yaml">cfgs/AMT-L</a>] </th>
    <th>Vimeo90k</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>AMT-G</td>
    <th>
    [<a href="https://pan.baidu.com/s/1UjeSuqwjI1YyrBoG8XeQgg?pwd=4epu">Baidu Cloud</a>]
    [<a href="https://drive.google.com/file/d/1mAsI084EzbH5s0iP47xHUmbKHmwIsicu/view?usp=drive_link">Google Driver</a>]</th>
    <th> [<a href="cfgs/AMT-G.yaml">cfgs/AMT-G</a>] </th>
    <th>Vimeo90k</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>AMT-S</td>
    <th>
    [<a>Baidu Cloud(TBD)</a>] 
    [<a>Google Driver(TBD)</a>]</th> </th>
    <th> [<a href="cfgs/AMT-S_gopro.yaml">cfgs/AMT-S_gopro</a>] </th>
    <th>GoPro</th>
    <th>Arbitrary</th>
  </tr>
</tbody>
</table>