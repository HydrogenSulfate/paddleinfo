# paddleinfo

> [!IMPORTANT]
> This is a PaddlePaddle backend adaptation derived from torchinfo

paddleinfo provides information complementary to what is provided by `print(your_model)` in PaddlePaddle, similar to TensorFlow's `model.summary()` API to view a visualization of the model—helpful while debugging your network. In this project, we implement a similar functionality for PaddlePaddle and create a clean, simple interface for use in your projects.

This is a completely rewritten version of the original torchsummary and torchsummaryX projects by @sksq96 and @nmhkahn, now adapted for PaddlePaddle. This project addresses longstanding issues and feature requests by introducing a modern, intuitive API tailored for PaddlePaddle users.

Supports PaddlePaddle versions 2.0+.

# Usage

``` sh
pip install paddleinfo
```


# How To Use

``` py
from paddleinfo import summary

model = ConvNet()
batch_size = 16
summary(model, input_size=(batch_size, 1, 28, 28))
```

``` log
================================================================================================================
Layer (type:depth-idx)          Input Shape          Output Shape         Param #            Mult-Adds
================================================================================================================
SingleInputNet                  [7, 1, 28, 28]       [7, 10]              --                 --
├─Conv2D: 1-1                   [7, 1, 28, 28]       [7, 10, 24, 24]      260                1,048,320
├─Conv2D: 1-2                   [7, 10, 12, 12]      [7, 20, 8, 8]        5,020              2,248,960
├─Dropout2D: 1-3                [7, 20, 8, 8]        [7, 20, 8, 8]        --                 --
├─Linear: 1-4                   [7, 320]             [7, 50]              16,050             112,350
├─Linear: 1-5                   [7, 50]              [7, 10]              510                3,570
================================================================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
Total mult-adds (M): 3.41
================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 0.40
Params size (MB): 0.09
Estimated Total Size (MB): 0.51
================================================================================================================
```