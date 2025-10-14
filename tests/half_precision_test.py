import sys

sys.path.append("/work/third_party_tools/paddleinfo")
import paddle
import pytest
from paddle_utils import *
from tests.fixtures.models import LinearModel, LSTMNet, SingleInputNet
from paddleinfo import summary
from paddleinfo.model_statistics import ModelStatistics


# @pytest.mark.skipif(
#     not paddle.cuda.is_available(), reason="Cuda must be available to test half models."
# )
# class TestHalfPrecision:
#     """
#     Iterate on these tests by using the following Google Colab link:
#     https://colab.research.google.com/drive/1e_86DcAL6Q0r1OkjFcOuYKQRJaKsIro8
#     """

#     @staticmethod
#     def test_single_input_half() -> None:
#         model = SingleInputNet()
#         # model.half().cuda()
#         input_data = paddle.randn((2, 1, 28, 28), dtype=paddle.float16, device="cuda")
#         results = summary(model, input_data=input_data)
#         assert ModelStatistics.to_megabytes(results.total_param_bytes) == 0.04368
#         assert ModelStatistics.to_megabytes(results.total_output_bytes) == 0.0568

#     @staticmethod
#     def test_linear_model_half() -> None:
#         x = paddle.randn((64, 128))
#         model = LinearModel()
#         results = summary(model, input_data=x)
#         # model.half().cuda()
#         x = x.astype(paddle.float16).cuda()
#         results_half = summary(model, input_data=x)
#         assert ModelStatistics.to_megabytes(
#             results_half.total_param_bytes
#         ) == pytest.approx(ModelStatistics.to_megabytes(results.total_param_bytes) / 2)
#         assert ModelStatistics.to_megabytes(
#             results_half.total_output_bytes
#         ) == pytest.approx(ModelStatistics.to_megabytes(results.total_output_bytes) / 2)

#     @staticmethod
#     def test_lstm_half() -> None:
#         model = LSTMNet()
#         model.half()
#         results = summary(
#             model,
#             input_size=(1, 100),
#             dtypes=[paddle.long],
#             col_width=20,
#             col_names=("kernel_size", "output_size", "num_params", "mult_adds"),
#             row_settings=("var_names",),
#         )
#         assert ModelStatistics.to_megabytes(results.total_param_bytes) == 7.56916
#         assert ModelStatistics.to_megabytes(results.total_output_bytes) == 0.3328
