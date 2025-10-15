from __future__ import annotations
import sys

sys.path.append('/work/third_party_tools/paddleinfo')

import math
from collections import namedtuple
from typing import Any, Sequence, cast

import numpy as np
import paddle
from paddle_utils import *


class IdentityModel(paddle.nn.Layer):
    """Identity Model."""

    def __init__(self) ->None:
        super().__init__()
        self.identity = paddle.nn.Identity()

    def forward(self, x: Any) ->Any:
        return self.identity(x)


class LinearModel(paddle.nn.Layer):
    """Linear Model."""

    def __init__(self) ->None:
        super().__init__()
        self.layers = paddle.nn.Sequential(paddle.nn.Linear(in_features=128,
            out_features=128), paddle.nn.ReLU(), paddle.nn.Linear(
            in_features=128, out_features=128), paddle.nn.ReLU(), paddle.nn
            .Linear(in_features=128, out_features=1))

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.layers(x)
        return x


# class UninitializedParameterModel(paddle.nn.Layer):
#     """UninitializedParameter test"""

#     def __init__(self) ->None:
#         super().__init__()
#         self.param: paddle.nn.Parameter = None

#     def init_param(self) ->None:
#         self.param = paddle.nn.Parameter(paddle.zeros(128))

#     def forward(self, x: paddle.Tensor) ->paddle.Tensor:
#         self.init_param()
#         return x


class SingleInputNet(paddle.nn.Layer):
    """Simple CNN model."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(1, 10, kernel_size=5)
        self.conv2 = paddle.nn.Conv2D(10, 20, kernel_size=5)
        self.conv2_drop = paddle.nn.Dropout2D(p=0.3)
        self.fc1 = paddle.nn.Linear(in_features=320, out_features=50)
        self.fc2 = paddle.nn.Linear(in_features=50, out_features=10)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = paddle.nn.functional.relu(x=paddle.nn.functional.max_pool2d(x=
            self.conv1(x), kernel_size=2))
        x = paddle.nn.functional.relu(x=paddle.nn.functional.max_pool2d(x=
            self.conv2_drop(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 320)
        x = paddle.nn.functional.relu(x=self.fc1(x))
        x = self.fc2(x)
        return paddle.nn.functional.log_softmax(x=x, axis=1)


class MultipleInputNetDifferentDtypes(paddle.nn.Layer):
    """Model with multiple inputs containing different dtypes."""

    def __init__(self) ->None:
        super().__init__()
        self.fc1a = paddle.nn.Linear(in_features=300, out_features=50)
        self.fc1b = paddle.nn.Linear(in_features=50, out_features=10)
        self.fc2a = paddle.nn.Linear(in_features=300, out_features=50)
        self.fc2b = paddle.nn.Linear(in_features=50, out_features=10)

    def forward(self, x1: paddle.Tensor, x2: paddle.Tensor) ->paddle.Tensor:
        x1 = paddle.nn.functional.relu(x=self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.astype(paddle.float32)
        x2 = paddle.nn.functional.relu(x=self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = paddle.cat((x1, x2), 0)
        return paddle.nn.functional.log_softmax(x=x, axis=1)


class ScalarNet(paddle.nn.Layer):
    """Model that takes a scalar as a parameter."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(64, 64, 3, 1, 1)
        self.conv2 = paddle.nn.Conv2D(64, 32, 3, 1, 1)

    def forward(self, x: paddle.Tensor, scalar: float) ->paddle.Tensor:
        out = x
        if scalar == 5:
            out = self.conv1(out)
        else:
            out = self.conv2(out)
        return out


class LSTMNet(paddle.nn.Layer):
    """Batch-first LSTM model."""

    def __init__(self, vocab_size: int=20, embed_dim: int=300, hidden_dim:
        int=512, num_layers: int=2) ->None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = paddle.nn.Embedding(vocab_size, embed_dim)
        self.encoder = paddle.nn.LSTM(input_size=embed_dim, hidden_size=
            hidden_dim, num_layers=num_layers, time_major=not False)
        self.decoder = paddle.nn.Linear(in_features=hidden_dim,
            out_features=vocab_size)

    def forward(self, x: paddle.Tensor) ->tuple[paddle.Tensor, paddle.Tensor]:
        embed = self.embedding(x)
        out, hidden = self.encoder(embed)
        out = self.decoder(out)
        out = out.view(-1, out.size(2))
        return out, hidden


class RecursiveNet(paddle.nn.Layer):
    """Model that uses a layer recursively in computation."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(64, 64, 3, 1, 1)

    def forward(self, x: paddle.Tensor, args1: Any=None, args2: Any=None
        ) ->paddle.Tensor:
        del args1, args2
        out = x
        for _ in range(3):
            out = self.conv1(out)
            out = self.conv1(out)
        return out


class CustomParameter(paddle.nn.Layer):
    """Model that defines a custom parameter."""

    def __init__(self, input_size: int, attention_size: int) ->None:
        super().__init__()
        self.weight = paddle.nn.parameter.Parameter(paddle.ones((
            attention_size, input_size)), True)
        paddle.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        del x
        return self.weight


class ParameterListModel(paddle.nn.Layer):
    """ParameterList of custom parameters."""

    def __init__(self) ->None:
        super().__init__()
        self.weights = paddle.nn.ParameterList(parameters=[paddle.nn.
            parameter.Parameter(weight) for weight in paddle.Tensor(100, 
            300).split([100, 200], dim=1)])

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        _ = self.weights
        return x


class SiameseNets(paddle.nn.Layer):
    """Model with MaxPool and ReLU layers."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(1, 64, 10)
        self.conv2 = paddle.nn.Conv2D(64, 128, 7)
        self.conv3 = paddle.nn.Conv2D(128, 128, 4)
        self.conv4 = paddle.nn.Conv2D(128, 256, 4)
        self.pooling = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.fc1 = paddle.nn.Linear(in_features=256, out_features=4096)
        self.fc2 = paddle.nn.Linear(in_features=4096, out_features=1)
        self.dropout = paddle.nn.Dropout(p=0.5)

    def forward(self, x1: paddle.Tensor, x2: paddle.Tensor) ->paddle.Tensor:
        x1 = self.pooling(paddle.nn.functional.relu(x=self.conv1(x1)))
        x1 = self.pooling(paddle.nn.functional.relu(x=self.conv2(x1)))
        x1 = self.pooling(paddle.nn.functional.relu(x=self.conv3(x1)))
        x1 = self.pooling(paddle.nn.functional.relu(x=self.conv4(x1)))
        x2 = self.pooling(paddle.nn.functional.relu(x=self.conv1(x2)))
        x2 = self.pooling(paddle.nn.functional.relu(x=self.conv2(x2)))
        x2 = self.pooling(paddle.nn.functional.relu(x=self.conv3(x2)))
        x2 = self.pooling(paddle.nn.functional.relu(x=self.conv4(x2)))
        batch_size = x1.size(0)
        x1 = x1.view(batch_size, -1)
        x2 = x2.view(batch_size, -1)
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)
        metric = paddle.abs(x=x1 - x2)
        similarity = paddle.nn.functional.sigmoid(self.fc2(self.dropout(
            metric)))
        return similarity


class FunctionalNet(paddle.nn.Layer):
    """Model that uses many functional paddle layers."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(1, 32, 5, 1)
        self.conv2 = paddle.nn.Conv2D(32, 64, 5, 1)
        self.dropout1 = paddle.nn.Dropout2D(p=0.4)
        self.dropout2 = paddle.nn.Dropout2D(p=0.5)
        self.fc1 = paddle.nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = paddle.nn.Linear(in_features=1024, out_features=10)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.conv1(x)
        x = paddle.nn.functional.relu(x=x)
        x = paddle.nn.functional.max_pool2d(x=x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = paddle.nn.functional.relu(x=x)
        x = paddle.nn.functional.max_pool2d(x=x, kernel_size=2, stride=2)
        x = x.view(-1, 2048)
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x=x)
        x = self.dropout1(x)
        x = self.fc2(x)
        output = paddle.nn.functional.log_softmax(x=x, axis=1)
        return output


class ReturnDictLayer(paddle.nn.Layer):
    """Model that returns a dict in forward()."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(1, 10, kernel_size=5)
        self.conv2 = paddle.nn.Conv2D(10, 20, kernel_size=5)
        self.fc1 = paddle.nn.Linear(in_features=320, out_features=50)
        self.fc2 = paddle.nn.Linear(in_features=50, out_features=10)

    def forward(self, x: paddle.Tensor) ->dict[str, paddle.Tensor]:
        activation_dict = {}
        x = self.conv1(x)
        activation_dict['conv1'] = x
        x = paddle.nn.functional.relu(x=paddle.nn.functional.max_pool2d(x=x,
            kernel_size=2))
        x = self.conv2(x)
        activation_dict['conv2'] = x
        x = paddle.nn.functional.relu(x=paddle.nn.functional.max_pool2d(x=x,
            kernel_size=2))
        x = x.view(-1, 320)
        x = paddle.nn.functional.relu(x=self.fc1(x))
        activation_dict['fc1'] = x
        x = self.fc2(x)
        activation_dict['fc2'] = x
        x = paddle.nn.functional.log_softmax(x=x, axis=1)
        activation_dict['output'] = x
        return activation_dict


class ReturnDict(paddle.nn.Layer):
    """Model that uses a ReturnDictLayer."""

    def __init__(self) ->None:
        super().__init__()
        self.return_dict = ReturnDictLayer()

    def forward(self, x: paddle.Tensor, y: Any) ->dict[str, paddle.Tensor]:
        del y
        activation_dict: dict[str, paddle.Tensor] = self.return_dict(x)
        return activation_dict


class DictParameter(paddle.nn.Layer):
    """Model that takes in a dict in forward()."""

    def __init__(self) ->None:
        super().__init__()
        self.constant = 5

    def forward(self, x: dict[int, paddle.Tensor], scale_factor: int
        ) ->paddle.Tensor:
        return scale_factor * (x[256] + x[512][0]) * self.constant


class ModuleDictModel(paddle.nn.Layer):
    """Model that uses a ModuleDict."""

    def __init__(self) ->None:
        super().__init__()
        self.choices = paddle.nn.LayerDict(sublayers={'conv': paddle.nn.
            Conv2D(10, 10, 3), 'pool': paddle.nn.MaxPool2D(kernel_size=3)})
        self.activations = paddle.nn.LayerDict(sublayers={'lrelu': paddle.
            nn.LeakyReLU(), 'prelu': paddle.nn.PReLU()})

    def forward(self, x: paddle.Tensor, layer_type: str, activation_type: str
        ) ->paddle.Tensor:
        x = self.choices[layer_type](x)
        x = self.activations[activation_type](x)
        return x


class ObjectWithTensors:
    """A class with a 'tensors'-attribute."""

    def __init__(self, tensors: (paddle.Tensor | Sequence[Any])) ->None:
        self.tensors = tensors


class HighlyNestedDictModel(paddle.nn.Layer):
    """Model that returns a highly nested dict."""

    def __init__(self) ->None:
        super().__init__()
        self.lin1 = paddle.nn.Linear(in_features=10, out_features=10)
        self.lin2 = paddle.nn.Linear(in_features=10, out_features=10)

    def forward(self, x: paddle.Tensor) ->dict[str, tuple[dict[str, list[
        ObjectWithTensors]]]]:
        x = self.lin1(x)
        x = self.lin2(x)
        x = paddle.compat.softmax(x, dim=0)
        return {'foo': ({'bar': [ObjectWithTensors(x)]},)}


class IntWithGetitem(int):
    """An int with a __getitem__ method."""

    def __init__(self, tensor: paddle.Tensor) ->None:
        super().__init__()
        self.tensor = tensor

    def __int__(self) ->IntWithGetitem:
        return self

    def __getitem__(self, val: int) ->paddle.Tensor:
        return self.tensor * val


class EdgecaseInputOutputModel(paddle.nn.Layer):
    """
    For testing LayerInfo.calculate_size.extract_tensor:

    case hasattr(inputs, "__getitem__") but not
    isinstance(inputs, (list, tuple, dict)).

    case not inputs.
    """

    def __init__(self) ->None:
        super().__init__()
        self.linear = paddle.nn.Linear(in_features=3, out_features=1)

    def forward(self, input_list: dict[str, paddle.Tensor]) ->dict[str,
        IntWithGetitem]:
        device = paddle.device('cuda' if paddle.cuda.is_available() else 'cpu')
        x = input_list['foo'] if input_list else paddle.ones(3).to(device)
        x = self.linear(x)
        return {'foo': IntWithGetitem(x)}


class NamedTuple(paddle.nn.Layer):
    """Model that takes in a NamedTuple as input."""
    Point = namedtuple('Point', ['x', 'y'])

    def forward(self, x: Any, y: Any, z: Any) ->Any:
        return self.Point(x, y).x + paddle.ones(z.x)


class NumpyModel(paddle.nn.Layer):
    """Model that takes a np.ndarray."""

    def __init__(self) ->None:
        super().__init__()
        self.lin = paddle.nn.Linear(in_features=3, out_features=3)

    def forward(self, inp: np.ndarray[Any, Any]) ->Any:
        assert isinstance(inp, np.ndarray)
        x = paddle.from_numpy(inp)
        x = self.lin(x)
        return x.cpu().detach().numpy()


class LayerWithRidiculouslyLongNameAndDoesntDoAnything(paddle.nn.Layer):
    """Model with a very long name."""

    def __init__(self) ->None:
        super().__init__()
        self.identity = paddle.nn.Identity()

    def forward(self, x: Any) ->Any:
        return self.identity(x)


class EdgeCaseModel(paddle.nn.Layer):
    """Model that throws an exception when used."""

    def __init__(self, throw_error: bool=False, return_str: bool=False,
        return_class: bool=False) ->None:
        super().__init__()
        self.throw_error = throw_error
        self.return_str = return_str
        self.return_class = return_class
        self.conv1 = paddle.nn.Conv2D(1, 10, kernel_size=5)
        self.model = LayerWithRidiculouslyLongNameAndDoesntDoAnything()

    def forward(self, x: paddle.Tensor) ->Any:
        x = self.conv1(x)
        x = self.model('string output' if self.return_str else x)
        if self.throw_error:
            x = self.conv1(x)
        if self.return_class:
            x = self.model(EdgeCaseModel)
        return x


# class PackPaddedLSTM(paddle.nn.Layer):
#     """LSTM model with pack_padded layers."""

#     def __init__(self, vocab_size: int=60, embedding_size: int=128,
#         output_size: int=18, hidden_size: int=32):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.embedding = paddle.nn.Embedding(vocab_size, embedding_size)
#         self.lstm = paddle.nn.LSTM(input_size=embedding_size, hidden_size=
#             self.hidden_size, num_layers=1, time_major=not False)
#         self.hidden2out = paddle.nn.Linear(in_features=self.hidden_size,
#             out_features=output_size)
#         self.dropout_layer = paddle.nn.Dropout(p=0.2)

#     def forward(self, batch: paddle.Tensor, lengths: paddle.Tensor
#         ) ->paddle.Tensor:
#         hidden1 = paddle.ones(1, batch.size(-1), self.hidden_size, device=
#             batch.place)
#         hidden2 = paddle.ones(1, batch.size(-1), self.hidden_size, device=
#             batch.place)
#         embeds = self.embedding(batch)
#         packed_input = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths)
#         _, (ht, _) = self.lstm(packed_input, (hidden1, hidden2))
#         output = self.dropout_layer(ht[-1])
#         output = self.hidden2out(output)
#         output = paddle.nn.functional.log_softmax(x=output, axis=1)
#         return cast(paddle.Tensor, output)


class ContainerModule(paddle.nn.Layer):
    """Model using ModuleList."""

    def __init__(self) ->None:
        super().__init__()
        self._layers = paddle.nn.LayerList()
        self._layers.append(paddle.nn.Linear(in_features=5, out_features=5))
        self._layers.append(ContainerChildModule())
        self._layers.append(paddle.nn.Linear(in_features=5, out_features=5))
        self._layers.append(None)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        out = x
        for layer in self._layers:
            if layer is not None:
                out = layer(out)
        return out


class ContainerChildModule(paddle.nn.Layer):
    """Model using Sequential in different ways."""

    def __init__(self) ->None:
        super().__init__()
        self._sequential = paddle.nn.Sequential(paddle.nn.Linear(
            in_features=5, out_features=5), paddle.nn.Linear(in_features=5,
            out_features=5))
        self._between = paddle.nn.Linear(in_features=5, out_features=5)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        out = self._sequential(x)
        out = self._between(out)
        for layer in self._sequential:
            out = layer(out)
        out = self._sequential(x)
        for layer in self._sequential:
            out = layer(out)
        return cast(paddle.Tensor, out)


class EmptyModule(paddle.nn.Layer):
    """A module that has no layers"""

    def __init__(self) ->None:
        super().__init__()
        out_0 = paddle.rand(shape=[3, 3])
        out_0.stop_gradient = not True
        self.parameter = out_0
        self.example_input_array = paddle.zeros(1, 2, 3, 4, 5)

    def forward(self) ->dict[str, Any]:
        return {'loss': self.parameter.sum()}


class AutoEncoder(paddle.nn.Layer):
    """Autoencoder module"""

    def __init__(self) ->None:
        super().__init__()
        self.encoder = paddle.nn.Sequential(paddle.nn.Conv2D(3, 16, 3,
            padding=1), paddle.nn.ReLU())
        self.pool = paddle.nn.MaxPool2D(kernel_size=2, stride=2,
            return_mask=True)
        self.unpool = paddle.nn.MaxUnPool2D(2, 2)
        self.decode = paddle.nn.Sequential(paddle.nn.Conv2D(16, 3, 3,
            padding=1), paddle.nn.ReLU())

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.encoder(x)
        unpooled_shape = x.size()
        x, indices = self.pool(x)
        x = self.unpool(x, indices=indices, output_size=unpooled_shape)
        x = self.decode(x)
        return x


class PartialJITModel(paddle.nn.Layer):
    """Partial JIT model."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(1, 10, kernel_size=5)
        self.conv2 = paddle.nn.Conv2D(10, 20, kernel_size=5)
        self.conv2_drop = paddle.nn.Dropout2D(p=0.3)
        self.fc1 = paddle.jit.to_static(function=paddle.nn.Linear(
            in_features=320, out_features=50))
        self.fc2 = paddle.jit.to_static(function=paddle.nn.Linear(
            in_features=50, out_features=10))

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = paddle.nn.functional.relu(x=paddle.nn.functional.max_pool2d(x=
            self.conv1(x), kernel_size=2))
        x = paddle.nn.functional.relu(x=paddle.nn.functional.max_pool2d(x=
            self.conv2_drop(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 320)
        x = paddle.nn.functional.relu(x=self.fc1(x))
        x = self.fc2(x)
        return paddle.nn.functional.log_softmax(x=x, axis=1)


class MixedTrainableParameters(paddle.nn.Layer):
    """Model with trainable and non-trainable parameters in the same layer."""

    def __init__(self) ->None:
        super().__init__()
        self.w = paddle.nn.parameter.Parameter(paddle.empty(10),
            requires_grad=True)
        self.b = paddle.nn.parameter.Parameter(paddle.empty(10),
            requires_grad=False)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        return self.w * x + self.b


class MixedTrainable(paddle.nn.Layer):
    """Model with fully, partial and non trainable modules."""

    def __init__(self) ->None:
        super().__init__()
        self.fully_trainable = paddle.nn.Conv1d(1, 1, 1)
        self.partially_trainable = paddle.nn.Conv1d(1, 1, 1, bias=True)
        assert self.partially_trainable.bias is not None
        self.partially_trainable.bias.stop_gradient = not False
        self.non_trainable = paddle.nn.Conv1d(1, 1, 1, 1, bias=True)
        self.non_trainable.weight.stop_gradient = not False
        assert self.non_trainable.bias is not None
        self.non_trainable.bias.stop_gradient = not False
        self.dropout = paddle.nn.Dropout()

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.fully_trainable(x)
        x = self.partially_trainable(x)
        x = self.non_trainable(x)
        x = self.dropout(x)
        return x


class ReuseLinear(paddle.nn.Layer):
    """Model that uses a reference to the same Linear layer over and over."""

    def __init__(self) ->None:
        super().__init__()
        linear = paddle.nn.Linear(in_features=10, out_features=10)
        model = []
        for _ in range(4):
            model += [linear, paddle.nn.ReLU()]
        self.model = paddle.nn.Sequential(*model)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        return cast(paddle.Tensor, self.model(x))


class ReuseLinearExtended(paddle.nn.Layer):
    """Model that uses a reference to the same Linear layer over and over."""

    def __init__(self) ->None:
        super().__init__()
        self.linear = paddle.nn.Linear(in_features=10, out_features=10)
        model = []
        for _ in range(4):
            model += [self.linear, paddle.nn.ReLU()]
        self.model = paddle.nn.Sequential(*model)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        return cast(paddle.Tensor, self.model(x))


class ReuseReLU(paddle.nn.Layer):
    """Model that uses a reference to the same ReLU layer over and over."""

    def __init__(self) ->None:
        super().__init__()
        activation = paddle.nn.ReLU()
        model = [paddle.nn.Pad2D(padding=3, mode='reflect'), paddle.nn.
            Conv2D(4, 1, kernel_size=1, padding=0), paddle.nn.BatchNorm2D(
            num_features=1), activation]
        for i in range(3):
            mult = 2 ** i
            model += [paddle.nn.Conv2D(mult, mult * 2, kernel_size=1,
                stride=2, padding=1), paddle.nn.BatchNorm2D(num_features=
                mult * 2), activation]
        self.model = paddle.nn.Sequential(*model)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        return cast(paddle.Tensor, self.model(x))


class PrunedLayerNameModel(paddle.nn.Layer):
    """Model that defines parameters with _orig and _mask as suffixes."""

    def __init__(self, input_size: int, attention_size: int) ->None:
        super().__init__()
        self.weight_orig = paddle.nn.parameter.Parameter(paddle.ones((
            attention_size, input_size)), True)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        del x
        return self.weight_orig


class FakePrunedLayerModel(paddle.nn.Layer):
    """Model that defines parameters with _orig and _mask as suffixes."""

    def __init__(self, input_size: int, attention_size: int) ->None:
        super().__init__()
        self.weight_orig = paddle.nn.parameter.Parameter(paddle.ones((
            attention_size, input_size)), True)
        self.weight_mask = paddle.nn.parameter.Parameter(paddle.zeros((
            attention_size, input_size)), True)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        del x
        return self.weight_orig


class RegisterParameter(paddle.nn.Sequential):
    """A model with one parameter."""
    weights: list[paddle.Tensor]

    def __init__(self, *blocks: paddle.nn.Layer) ->None:
        super().__init__(*blocks)
        self.add_parameter(name='weights', parameter=paddle.nn.parameter.
            Parameter(paddle.zeros(len(blocks)).to(paddle.float32)))

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        for k, block in enumerate(self):
            x += self.weights[k] * block(x)
        return x


class ParameterFCNet(paddle.nn.Layer):
    """FCNet using Parameters."""

    def __init__(self, input_dim: int=128, hidden_dim: int=64, output_dim:
        (int | None)=None) ->None:
        super().__init__()
        self.output_dim = output_dim
        self.a = paddle.nn.parameter.Parameter(paddle.randn(input_dim,
            hidden_dim))
        self.b = paddle.nn.parameter.Parameter(paddle.randn(hidden_dim))
        if output_dim is not None:
            self.fc2 = paddle.nn.Linear(in_features=hidden_dim,
                out_features=output_dim)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        h = paddle.mm(input=x, mat2=self.a) + self.b
        if self.output_dim is None:
            return h
        return cast(paddle.Tensor, self.fc2(h))


class InsideModel(paddle.nn.Layer):
    """Module with a parameter and an inner module with a Parameter."""


    class Inside(paddle.nn.Layer):
        """Inner module with a Parameter."""

        def __init__(self) ->None:
            super().__init__()
            self.l_1 = paddle.nn.Linear(in_features=1, out_features=1)
            self.param_1 = paddle.nn.parameter.Parameter(paddle.ones(1))

        def forward(self, x: paddle.Tensor) ->paddle.Tensor:
            return cast(paddle.Tensor, self.l_1(x) * self.param_1)

    def __init__(self) ->None:
        super().__init__()
        self.l_0 = paddle.nn.Linear(in_features=2, out_features=1)
        self.param_0 = paddle.nn.parameter.Parameter(paddle.ones(2))
        self.inside = InsideModel.Inside()

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        return cast(paddle.Tensor, self.inside(self.l_0(x)) * self.param_0)


class RecursiveWithMissingLayers(paddle.nn.Layer):
    """
    Module with more complex recursive layers, which activates add_missing_layers.
    """

    def __init__(self) ->None:
        super().__init__()
        self.out_conv0 = paddle.nn.Conv2D(3, 8, 5, padding='same')
        self.out_bn0 = paddle.nn.BatchNorm2D(num_features=8)
        self.block0 = paddle.nn.LayerDict()
        for i in range(1, 4):
            self.block0.add_sublayer(name=f'in_conv{i}', sublayer=paddle.nn
                .Conv2D(8, 8, 3, padding='same', dilation=2 ** i))
            self.block0.add_sublayer(name=f'in_bn{i}', sublayer=paddle.nn.
                BatchNorm2D(num_features=8))
        self.block1 = paddle.nn.LayerDict()
        for i in range(4, 7):
            self.block1.add_sublayer(name=f'in_conv{i}', sublayer=paddle.nn
                .Conv2D(8, 8, 3, padding='same', dilation=2 ** (7 - i)))
            self.block1.add_sublayer(name=f'in_bn{i}', sublayer=paddle.nn.
                BatchNorm2D(num_features=8))
        self.out_conv7 = paddle.nn.Conv2D(8, 1, 1, padding='same')
        self.out_bn7 = paddle.nn.BatchNorm2D(num_features=1)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.out_conv0(x)
        x = paddle.nn.functional.relu(x=self.out_bn0(x))
        for i in range(1, 4):
            x = self.block0[f'in_conv{i}'](x)
            x = paddle.nn.functional.relu(x=self.block0[f'in_bn{i}'](x))
        for i in range(4, 7):
            x = self.block1[f'in_conv{i}'](x)
            x = paddle.nn.functional.relu(x=self.block1[f'in_bn{i}'](x))
        x = self.out_conv7(x)
        x = paddle.nn.functional.relu(x=self.out_bn7(x))
        return x


class CNNModuleList(paddle.nn.Layer):
    """ModuleList with ConvLayers."""

    def __init__(self, conv_layer_cls: type[paddle.nn.Layer]) ->None:
        super().__init__()
        self.ml = paddle.nn.LayerList(sublayers=[conv_layer_cls() for i in
            range(5)])

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        for layer in self.ml:
            x = layer(x)
        return x


class ConvLayerA(paddle.nn.Layer):
    """ConvLayer with the same module instantiation order in forward()."""

    def __init__(self) ->None:
        super().__init__()
        self.conv = paddle.nn.Conv1d(1, 1, 1)
        self.relu = paddle.nn.ReLU()
        self.pool = paddle.nn.MaxPool1D(kernel_size=1)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        out = self.pool(out)
        return cast(paddle.Tensor, out)


class ConvLayerB(paddle.nn.Layer):
    """ConvLayer with a different module instantiation order in forward()."""

    def __init__(self) ->None:
        super().__init__()
        self.relu = paddle.nn.ReLU()
        self.conv = paddle.nn.Conv1d(1, 1, 1)
        self.pool = paddle.nn.MaxPool1D(kernel_size=1)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        out = self.pool(out)
        return cast(paddle.Tensor, out)


class SimpleRNN(paddle.nn.Layer):
    """Simple RNN"""

    def __init__(self, repeat_outside_loop: bool=False) ->None:
        super().__init__()
        self.hid_dim = 2
        self.input_dim = 3
        self.max_length = 4
        self.repeat_outside_loop = repeat_outside_loop
        self.lstm = LSTMCell(input_size=self.input_dim, hidden_size=self.
            hid_dim)
        self.activation = paddle.nn.Tanh()
        self.projection = paddle.nn.Linear(in_features=self.hid_dim,
            out_features=self.input_dim)

    def forward(self, token_embedding: paddle.Tensor) ->paddle.Tensor:
        b_size = token_embedding.size()[0]
        hx = paddle.randn(b_size, self.hid_dim, device=token_embedding.place)
        cx = paddle.randn(b_size, self.hid_dim, device=token_embedding.place)
        for _ in range(self.max_length):
            hx, cx = self.lstm(token_embedding, (hx, cx))
            hx = self.activation(hx)
        if self.repeat_outside_loop:
            hx = self.projection(hx)
            hx = self.activation(hx)
        return hx


class MultiDeviceModel(paddle.nn.Layer):
    """
    A model living on several devices.

    Follows the ToyModel from the Tutorial on parallelism:
    https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
    """

    def __init__(self, device1: (paddle.device | str), device2: (paddle.
        device | str)) ->None:
        super().__init__()
        self.device1 = device1
        self.device2 = device2
        self.net1 = paddle.nn.Linear(in_features=10, out_features=10).to(
            device1)
        self.relu = paddle.nn.ReLU()
        self.net2 = paddle.nn.Linear(in_features=10, out_features=5).to(device2
            )

    def forward(self, x: paddle.Tensor) ->Any:
        x = self.relu(self.net1(x.to(self.device1)))
        return self.net2(x.to(self.device2))
