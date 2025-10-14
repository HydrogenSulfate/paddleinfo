import sys

sys.path.append("/work/third_party_tools/paddleinfo")
import paddle
import pytest
from paddle_utils import *
from tests.fixtures.models import MultiDeviceModel, SingleInputNet
from paddleinfo import summary


@pytest.mark.skipif(not paddle.cuda.is_available(), reason="GPU must be enabled.")
class TestGPU:
    """GPU-only tests."""

    # @staticmethod
    # def test_single_layer_network_on_gpu() -> None:
    #     model = paddle.nn.Linear(in_features=2, out_features=5)
    #     # model.cuda()
    #     results = summary(model, input_size=(1, 2))
    #     assert results.total_params == 15
    #     assert results.trainable_params == 15

    # @staticmethod
    # def test_single_layer_network_on_gpu_device() -> None:
    #     model = paddle.nn.Linear(in_features=2, out_features=5)
    #     results = summary(model, input_size=(1, 2), device="cuda")
    #     assert results.total_params == 15
    #     assert results.trainable_params == 15

    # @staticmethod
    # def test_input_size_half_precision() -> None:
    #     test = (
    #         paddle.nn.Linear(in_features=2, out_features=5)
    #         # .half()
    #         .to(paddle.device("cuda"))
    #     )
    #     with pytest.raises(
    #         RuntimeError,
    #     ):
    #         summary(test, dtypes=[paddle.float16], input_size=(10, 2), device="cuda")

    # @staticmethod
    # def test_device() -> None:
    #     model = SingleInputNet()
    #     summary(model, input_size=(5, 1, 28, 28), device="cuda")
    #     model = SingleInputNet()
    #     input_data = paddle.randn(5, 1, 28, 28)
    #     summary(model, input_data=input_data)
    #     summary(model, input_data=input_data, device="cuda")
    #     summary(model, input_data=input_data.to("cuda"))
    #     summary(model, input_data=input_data.to("cuda"), device=paddle.device("cpu"))


@pytest.mark.skipif(
    not paddle.cuda.device_count() >= 2, reason="Only relevant for multi-GPU"
)
class TestMultiGPU:
    """multi-GPU-only tests"""

    @staticmethod
    def test_model_stays_on_device_if_gpu() -> None:
        model = paddle.nn.Linear(in_features=10, out_features=10).to("cuda:1")
        summary(model)
        model_parameter = model.parameters()[0]
        assert model_parameter.place == paddle.device("cuda:1")

    @staticmethod
    def test_different_model_parts_on_different_devices() -> None:
        model = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=10, out_features=10).to(1),
            paddle.nn.Linear(in_features=10, out_features=10).to(0),
        )
        summary(model)


# @pytest.mark.skipif(
#     not paddle.cuda.is_available(), reason="Need CUDA to test parallelism."
# )
# def test_device_parallelism() -> None:
#     model = MultiDeviceModel("cpu", "cuda")
#     input_data = paddle.randn(10)
#     summary(model, input_data=input_data)
#     assert not model.net1.parameters()[0].is_cuda
#     assert model.net2.parameters()[0].is_cuda
