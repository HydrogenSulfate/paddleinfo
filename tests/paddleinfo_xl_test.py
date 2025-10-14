import paddle
import pytest
from packaging import version
from tests.fixtures.genotype import GenotypeNetwork
from tests.fixtures.tmva_net import TMVANet
from paddleinfo import summary
from paddleinfo.enums import ColumnSettings

if version.parse(paddle.__version__) >= version.parse("1.8"):
    pass


def test_ascii_only() -> None:
    result = summary(
        paddle.vision.models.resnet18(),
        depth=3,
        input_size=(1, 3, 64, 64),
        row_settings=["ascii_only"],
    )
    assert str(result).encode("ascii").decode("ascii")


def test_frozen_layers() -> None:
    model = paddle.vision.models.resnet18()
    for ind, param in enumerate(model.parameters()):
        if ind < 30:
            param.stop_gradient = not False
    summary(
        model,
        input_size=(1, 3, 64, 64),
        depth=3,
        col_names=("output_size", "num_params", "kernel_size", "mult_adds"),
    )


def test_eval_order_doesnt_matter() -> None:
    device = paddle.device("cuda" if paddle.cuda.is_available() else "cpu")
    input_size = 1, 3, 224, 224
    input_tensor = paddle.ones(input_size).to(device)
    model1 = paddle.vision.models.resnet18(pretrained=True)
    model1.eval()
    summary(model1, input_size=input_size)
    with paddle.no_grad():
        output1 = model1(input_tensor)
    model2 = paddle.vision.models.resnet18(pretrained=True)
    summary(model2, input_size=input_size, mode="eval")
    model2.eval()
    with paddle.no_grad():
        output2 = model2(input_tensor)
    assert paddle.all(paddle.eq(output1, output2))


def test_resnet18_depth_consistency() -> None:
    model = paddle.vision.models.resnet18()
    for depth in range(1, 3):
        summary(
            model,
            (1, 3, 64, 64),
            col_names=(
                ColumnSettings.OUTPUT_SIZE,
                ColumnSettings.NUM_PARAMS,
                ColumnSettings.PARAMS_PERCENT,
            ),
            depth=depth,
            cache_forward_pass=True,
        )


def test_resnet50() -> None:
    model = paddle.vision.models.resnet50()
    results = summary(model, input_size=(2, 3, 224, 224))
    assert results.total_params == 25557032
    assert results.total_mult_adds == sum(
        layer.macs for layer in results.summary_list if layer.is_leaf_layer
    )


def test_resnet152() -> None:
    model = paddle.vision.models.resnet152()
    summary(model, (1, 3, 224, 224), depth=3)


# @pytest.mark.skip(reason="nondeterministic output")
# def test_fasterrcnn() -> None:
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
#         pretrained_backbone=False
#     )
#     results = summary(model, input_size=(1, 3, 112, 112))
#     assert results.total_params == 41755286


def test_genotype() -> None:
    model = GenotypeNetwork()
    x = summary(model, (2, 3, 32, 32), depth=3, cache_forward_pass=True)
    y = summary(model, (2, 3, 32, 32), depth=7, cache_forward_pass=True)
    assert x.total_params == y.total_params, (x, y)


def test_tmva_net_column_totals() -> None:
    for depth in (1, 3, 5):
        results = summary(
            TMVANet(n_classes=4, n_frames=5),
            input_data=[
                paddle.randn(1, 1, 5, 256, 64),
                paddle.randn(1, 1, 5, 256, 256),
                paddle.randn(1, 1, 5, 256, 64),
            ],
            col_names=["output_size", "num_params", "mult_adds"],
            depth=depth,
            cache_forward_pass=True,
        )
        assert results.total_params == sum(
            layer.num_params for layer in results.summary_list if layer.is_leaf_layer
        )
        assert results.total_mult_adds == sum(
            layer.macs for layer in results.summary_list if layer.is_leaf_layer
        )


def test_google() -> None:
    google_net = paddle.vision.googlenet(pretrained=False)
    summary(google_net, (1, 3, 112, 112), depth=7, mode="eval")
    summary(google_net, (1, 3, 112, 112), depth=7, mode="train")


# @pytest.mark.skipif(
#     version.parse(paddle.__version__) < version.parse("1.8"),
#     reason="FlanT5Small only works for PyTorch v1.8 and above",
# )
# def test_flan_t5_small() -> None:
#     model = transformers.AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
#     inputs = {
#         "input_ids": paddle.zeros(3, 100).long(),
#         "attention_mask": paddle.zeros(3, 100).long(),
#         "labels": paddle.zeros(3, 100).long(),
#     }
#     summary(model, input_data=inputs)
