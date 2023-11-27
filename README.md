# Additive VQ + Product VQ + Residual VQ

This repository contains PyTorch implementation of three vector quantization (VQ) methods which can be trained with machine learning optimizers (e.g. Adam, SGD) using [NSVQ](https://ieeexplore.ieee.org/abstract/document/9696322) technique. These VQ methods are published as the paper "Stochastic Optimization of Vector Quantization Methods in Application to Speech and Image Processing" in ICASSP 2023. You can find **a short explanation of the paper** in [this medium post](https://medium.com/towards-data-science/optimizing-vector-quantization-methods-by-machine-learning-algorithms-77c436d0749d).

# **Contents of this repository**

- `additive_vq.py`: contains the main class of Additive Vector Quantization (AVQ)
- `product_vq.py`: contains the main class of Product Vector Quantization (PVQ)
- `residual_vq.py`: contains the main class of Residual Vector Quantization (RVQ)
- `train_additive_vq.py`: an example showing how to train AVQ to learn a Normal distribution
- `train_product_vq.py`: an example showing how to train PVQ to learn a Normal distribution
- `train_residual_vq.py`: an example showing how to train RVQ to learn a Normal distribution
- `complexity.py`: contains the main classes to calculate complexity of each AVQ, PVQ, and RVQ methods
- `complexity_calculator.py`: an example showing how to calcualte the complexity of AVQ, PVQ, and RVQ methods
- `plot_training_logs.py`: plots the training logs (which was saved druring execution of "train.py") in a pdf file  

Due to some limitations of TensorBoard, we prefered our own custom logging function (plot_training_logs.py).

# **Required packages**
- Python (version: 3.8 or higher)
- PyTorch (version: 1.12.1)
- Numpy (version: 1.23.5)  
- Matplotlib (version: 3.6)

You can create the Python environment by passing the following lines of codes in your terminal window in the following order:

`conda create --name vq_methods python=3.8`  
`conda activate vq_methods`  
`pip install torch==1.12.1`  
`pip install numpy==1.23.5`  
`pip install matplotlib==3.6`

The requirements to use this repository is not that much strict, becuase the functions used in the code are so basic such that they also work with higher Python, PyTorch and Numpy versions.

# **Important: Codebook Replacement**

During training, we apply codebook replacement function (explained in section III.C in [the paper](https://ieeexplore.ieee.org/abstract/document/9696322)) to discard those codebook vectors which are not involved in the vector quantization process. There are two reasons for that; 1) the codebook replacement acts as a trigger to make the codebook vectors to start updating in some applications , 2) it allows exploiting from all available codebook vectors (better vector quantization perfromance) and avoiding isolated and rarely used codebooks. The codebook replacement is implemented as a function named `replace_unused_codebooks` in each VQ method's class. The essential explanations are prepared for this function in the code. Feel free to change the **discard_threshod** and **num_batches** parameters which are related to the codebook replacement function based on your application. However, the recommended procedure of codebook replacement is in the following.

Call this function after a specific number of training batches (**num_batches**) during training. In the beginning, the number of replaced codebooks might increase (the number of replaced codebooks will be printed out during training). However, the main trend must be decreasing after some training time, because the codebooks will find a location inside the distribution which makes them useful representative of the distribution. If the replacement trend is not decreasing, increase the **num_batches** or decrease the **discard_threshold**. Stop calling the function at the latest stages of training (for example the last 1000 training batches) in order not to introduce any new codebook entry which would not have the right time to be tuned and optimized until the end of training. Remember that the number of repalced codebook vectors will be printed after each round you call the function.

# **Results directory**

The "Results" directory contains the values of objective metrics, which were used to plot the figures of the paper. The values are provided in JSON file format. We have shared these results with the aim of saving time for reproducabiltiy and making it easier for researchers to do potential comparisons. Note that in order to calculate the PESQ values, we have installed and used the **PESQ package from PyPI** ([under this link](https://pypi.org/project/pesq/)) in our Python environment.

# **Abstract of the paper**

Vector quantization (VQ) methods have been used in a wide range of applications for speech, image, and video data. While classic VQ methods often use expectation maximization, in this paper, we investigate the use of stochastic optimization employing our recently proposed noise substitution in vector quantization technique. We consider three variants of VQ including additive VQ, residual VQ, and product VQ, and evaluate their quality, complexity and bitrate in speech coding, image compression, approximate nearest neighbor search, and a selection of toy examples. Our experimental results demonstrate the trade-offs in accuracy, complexity, and bitrate such that using our open source implementations and complexity calculator, the best vector quantization method can be chosen for a particular problem.

# **Cite the paper as**

Mohammad Hassan Vali and Tom Bäckström, “Stochastic Optimization of Vector Quantization Methods in Application to Speech and Image Processing”, in Proceedings of ICASSP, 2023.

```bibtex
@inproceedings{vali2023stochastic,
  title={{S}tochastic {O}ptimization of {V}ector {Q}uantization {M}ethods in {A}pplication to {S}peech and {I}mage {P}rocessing},
  author={Vali, Mohammad Hassan and Bäckström, Tom},
  booktitle={Proceedings of ICASSP},
  year={2023}
}
```
