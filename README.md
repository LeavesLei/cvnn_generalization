# cvnn_generalization

This repository contains the code for the paper "[Spectral Complexity-scaled Generalization Bound of Complex-valued Neural Networks](https://arxiv.org/abs/2112.03467)" by Haowen Chen, Fengxiang He, Shiye Lei, and Dacheng Tao.

## Dependencies

- Python 3.6
- Pytorch 1.10
- Keras 2.2
- complexPyTorch


## Run

1. setting the data path of TinyImageNet in `main.py`:

   `tinyimagenet_path = [YOUR DATA PATH]`

2. setting the path of result folder in `main.py`:

   `figure_save_path = [YOUR FIGURE PATH]`

3. run `main.py`

   ```bash
   python main.py --dataset [YOUR DATASET NAME]
   ```

   [YOUR DATASET NAME] ∈ {'mnist', 'fashionmnist', 'cifar10', 'cifar100', 'imdb', 'tinyimagenet'}


## Citation
```
@article{chen2023spectral,
  title={Spectral Complexity-scaled Generalization Bound of Complex-valued Neural Networks},
  author={Chen, Haowen and He, Fengxiang and Lei, Shiye and Tao, Dacheng},
  journal={Artificial Intelligence},
  year={2023}
}
```

## Contact

For any issue, please kindly contact

Hoawen Chen: [haowchen@student.ethz.ch](mailto:haowchen@student.ethz.ch)  
Fengxiang He: [F.He@ed.ac.uk](mailto:F.He@ed.ac.uk)  
Shiye Lei: [leishiye@gmail.com](mailto:leishiye@gmail.com)  
Dacheng Tao: [dacheng.tao@sydney.edu.au](mailto:dacheng.tao@sydney.edu.au)

---

Last update: Tue 30 May 2023
