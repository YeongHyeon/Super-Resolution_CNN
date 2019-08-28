Super-Resolution CNN
=====

Implementation of 'Image Super-Resolution using Deep Convolutional Network'.  

## Architecture
<div align="center">
  <img src="./readme/srcnn.png" width="700">  
  <p>The architecture of the Super-Resolution Network (SRCNN).</p>
</div>
The architecture constructed by three convolutional layers, and the kernel size are 9x9, 1x1, 3x2 respectively. It used RMS loss and stochastic gradient descent opeimizer for training in this repository, but original one was trained by MSE loss (using same optimizer). The input of the SRCNN is Low-Resolution (Bicubic Interpolated) image that same size of the output image, and the output is High-Resolution.  

## Results
<div align="center">
  <img src="./readme/iteration.png" width="700">  
  <p>Reconstructed image in each iteration.</p>
</div>

<div align="center">
  <img src="./readme/comparison.png" width="700">  
  <p>Comparison between the input (Bicubic Interpolated), reconstructed image (by SRCNN) and output (High-Resolution) image.</p>
</div>

## Requirements
* Python 3.6.8  
* Tensorflow 1.14.0  
* Numpy 1.14.0  
* Matplotlib 3.1.1  

## Reference
[1] Image Super-Resolution Using Deep Convolutional Networks, Chao Dong et al., https://ieeexplore.ieee.org/abstract/document/7115171/  
[2] Urban 100 dataset, Huang et al.,  https://sites.google.com/site/jbhuang0604/publications/struct_sr  

First commit: 21.April.2018  
Version Update: 28.August.2019  
