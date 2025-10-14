
import paddle

############################## 相关utils函数，如下 ##############################
############################ PaConvert 自动生成的代码 ###########################

def device2int(device):
    if isinstance(device, str):
        device = device.replace('cuda', 'gpu')
        device = device.replace('gpu:', '')
    return int(device)

def _Tensor_split(self, split_size, dim=0):
    if isinstance(split_size, int):
        return paddle.split(self, self.shape[dim] // split_size, dim)
    else:
        return paddle.split(self, split_size, dim)

setattr(paddle.Tensor, "split", _Tensor_split)

class LSTMCell(paddle.nn.LSTMCell):
    def forward(self, inputs, states = None):
        return super().forward(inputs, states)[1]
############################## 相关utils函数，如上 ##############################

