# PanFormer
PanFormer pansharpenning method implemented in pytorch

Pretrained model is provided

Based on implementation: [https://github.com/xyc19970716/Deep-Learning-PanSharpening/tree/main](https://github.com/zhysora/PanFormer)

Paper link: [https://www.mdpi.com/2072-4292/8/7/594](https://arxiv.org/abs/2203.02916)

# Dataset

The GaoFen-2 and WorldView-3 dataset download links can be found in https://github.com/liangjiandeng/PanCollection

# Torch Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [-1, 64, 64, 64]          --
|    └─SwinModule: 2-1                   [-1, 64, 128, 128]        --
|    |    └─PatchMerging: 3-1            [-1, 128, 128, 64]        320
|    └─SwinModule: 2-2                   [-1, 64, 64, 64]          --
|    |    └─PatchMerging: 3-2            [-1, 64, 64, 64]          16,448
├─Sequential: 1-2                        [-1, 64, 64, 64]          --
|    └─SwinModule: 2-3                   [-1, 64, 64, 64]          --
|    |    └─PatchMerging: 3-3            [-1, 64, 64, 64]          320
|    └─SwinModule: 2-4                   [-1, 64, 64, 64]          --
|    |    └─PatchMerging: 3-4            [-1, 64, 64, 64]          4,160
├─ModuleList: 1                          []                        --
|    └─SwinModule: 2-5                   [-1, 64, 64, 64]          --
|    |    └─PatchMerging: 3-5            [-1, 64, 64, 64]          4,160
|    |    └─PatchMerging: 3-6            [-1, 64, 64, 64]          (recursive)
├─ModuleList: 1                          []                        --
|    └─SwinModule: 2-6                   [-1, 64, 64, 64]          --
|    |    └─PatchMerging: 3-7            [-1, 64, 64, 64]          4,160
|    |    └─PatchMerging: 3-8            [-1, 64, 64, 64]          (recursive)
├─ModuleList: 1                          []                        --
|    └─SwinModule: 2-7                   [-1, 64, 64, 64]          --
|    |    └─PatchMerging: 3-9            [-1, 64, 64, 64]          4,160
|    |    └─PatchMerging: 3-10           [-1, 64, 64, 64]          (recursive)
├─ModuleList: 1                          []                        --
|    └─SwinModule: 2-8                   [-1, 64, 64, 64]          --
|    |    └─PatchMerging: 3-11           [-1, 64, 64, 64]          4,160
|    |    └─PatchMerging: 3-12           [-1, 64, 64, 64]          (recursive)
├─ModuleList: 1                          []                        --
|    └─SwinModule: 2-9                   [-1, 64, 64, 64]          --
|    |    └─PatchMerging: 3-13           [-1, 64, 64, 64]          4,160
|    |    └─PatchMerging: 3-14           [-1, 64, 64, 64]          (recursive)
├─ModuleList: 1                          []                        --
|    └─SwinModule: 2-10                  [-1, 64, 64, 64]          --
|    |    └─PatchMerging: 3-15           [-1, 64, 64, 64]          4,160
|    |    └─PatchMerging: 3-16           [-1, 64, 64, 64]          (recursive)
├─Sequential: 1-3                        [-1, 4, 256, 256]         --
|    └─Conv2d: 2-11                      [-1, 256, 64, 64]         295,168
|    └─PixelShuffle: 2-12                [-1, 64, 128, 128]        --
|    └─ReLU: 2-13                        [-1, 64, 128, 128]        --
|    └─Conv2d: 2-14                      [-1, 256, 128, 128]       147,712
|    └─PixelShuffle: 2-15                [-1, 64, 256, 256]        --
|    └─ReLU: 2-16                        [-1, 64, 256, 256]        --
|    └─Conv2d: 2-17                      [-1, 64, 256, 256]        36,928
|    └─ReLU: 2-18                        [-1, 64, 256, 256]        --
|    └─Conv2d: 2-19                      [-1, 4, 256, 256]         2,308
==========================================================================================
Total params: 528,324
Trainable params: 528,324
Non-trainable params: 0
Total mult-adds (G): 6.19
==========================================================================================
Input size (MB): 0.25
Forward/backward pass size (MB): 100.00
Params size (MB): 2.02
Estimated Total Size (MB): 102.27
==========================================================================================
```
# Statistics

![alt text](https://github.com/nickdndndn/PanFormer/blob/main/results/Figures.png?raw=true)

# Visualization of Results

![alt text](https://github.com/nickdndndn/PanFormer/blob/main/results/Images.png?raw=true)
