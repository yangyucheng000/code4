


# $\rm{[MindSpore-phase3]}$ $AMT$

本项目包含了以下论文的mindspore实现：

> **AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation**<br>
> [Zhen Li](https://paper99.github.io/)<sup>\*</sup>, [Zuo-Liang Zhu](https://nk-cs-zzl.github.io/)<sup>\*</sup>, [Ling-Hao Han](https://scholar.google.com/citations?user=0ooNdgUAAAAJ&hl=en), [Qibin Hou](https://scholar.google.com/citations?hl=en&user=fF8OFV8AAAAJ&view_op=list_works), [Chun-Le Guo](https://scholar.google.com/citations?hl=en&user=RZLYwR0AAAAJ),  [Ming-Ming Cheng](https://mmcheng.net/cmm)<br>
> (\* denotes equal contribution) <br>
> Nankai University <br>
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2023<br>

[[Paper](https://arxiv.org/abs/2304.09790)] [[Project Page](https://nk-cs-zzl.github.io/projects/amt/index.html)]   [[Web demos](#web-demos)]

文章官方版本仓库链接: https://github.com/Srameo/LED


目前已经完成部分代码的mindspore转化

## 正在进行中的工作

-  完整代码的mindspore实现

## :wrench: Dependencies and Installation

> python 3.8 <br>
> cuda 11.6 <br>
> MindSpore Nightly https://www.mindspore.cn/install/ 

1. Clone and enter the repo:
   ```bash
   git clone https://github.com/Men1scus/LED_MindSpore.git 
   cd LED_MindSpore
   ```
2. Simply run the `install.sh` for installation! Or refer to [install.md](docs/install.md) for more details.
   > We use the customized rawpy package in [ELD](https://github.com/Vandermode/ELD), if you don't want to use it or want to know more information, please move to [install.md](docs/install.md)
   ```bash
   bash install.sh
   pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
3. Activate your env and start testing!
   ```bash
   conda activate LED-ICCV23
   ```

