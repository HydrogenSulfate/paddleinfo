from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .enums import Units
from .formatting import CONVERSION_FACTORS, FormattingOptions

if TYPE_CHECKING:
    from .layer_info import LayerInfo


class ModelStatistics:
    """Class for storing results of the summary."""

    def __init__(
        self,
        summary_list: list[LayerInfo],
        input_size: Any,
        total_input_size: int,
        formatting: FormattingOptions,
    ) -> None:
        self.summary_list = summary_list
        self.input_size = input_size
        self.formatting = formatting
        self.total_input = total_input_size
        self.total_mult_adds = 0
        self.total_params, self.trainable_params = 0, 0
        self.total_param_bytes, self.total_output_bytes = 0, 0
        for layer_info in summary_list:
            if layer_info.is_leaf_layer:
                self.total_mult_adds += layer_info.macs
                if layer_info.num_params > 0:
                    self.total_output_bytes += layer_info.output_bytes * 2
                if layer_info.is_recursive:
                    continue
                self.total_params += max(layer_info.num_params, 0)
                self.total_param_bytes += layer_info.param_bytes
                self.trainable_params += max(layer_info.trainable_params, 0)
            else:
                if layer_info.is_recursive:
                    continue
                leftover_params = layer_info.leftover_params()
                leftover_trainable_params = layer_info.leftover_trainable_params()
                self.total_params += max(leftover_params, 0)
                self.trainable_params += max(leftover_trainable_params, 0)
        self.formatting.set_layer_name_width(summary_list)

    def __repr__(self) -> str:
        """Print results of the summary."""
        divider = "=" * self.formatting.get_total_width()
        total_params = ModelStatistics.format_output_num(
            self.total_params, self.formatting.params_count_units, False
        )
        trainable_params = ModelStatistics.format_output_num(
            self.trainable_params, self.formatting.params_count_units, False
        )
        non_trainable_params = ModelStatistics.format_output_num(
            self.total_params - self.trainable_params,
            self.formatting.params_count_units,
            False,
        )
        all_layers = self.formatting.layers_to_str(self.summary_list, self.total_params)
        summary_str = f"""{divider}
{self.formatting.header_row()}{divider}
{all_layers}{divider}
Total params{total_params}
Trainable params{trainable_params}
Non-trainable params{non_trainable_params}
"""
        if self.input_size:
            macs = ModelStatistics.format_output_num(
                self.total_mult_adds, self.formatting.macs_units, False
            )
            input_size = ModelStatistics.format_output_num(
                self.total_input, self.formatting.params_size_units, True
            )
            output_bytes = ModelStatistics.format_output_num(
                self.total_output_bytes, self.formatting.params_size_units, True
            )
            param_bytes = ModelStatistics.format_output_num(
                self.total_param_bytes, self.formatting.params_size_units, True
            )
            total_bytes = ModelStatistics.format_output_num(
                self.total_input + self.total_output_bytes + self.total_param_bytes,
                self.formatting.params_size_units,
                True,
            )
            summary_str += f"""Total mult-adds{macs}
{divider}
Input size{input_size}
Forward/backward pass size{output_bytes}
Params size{param_bytes}
Estimated Total Size{total_bytes}
"""
        summary_str += divider
        return summary_str

    @staticmethod
    def float_to_megabytes(num: int) -> float:
        """Converts a number (assume floats, 4 bytes each) to megabytes."""
        return num * 4 / 1000000.0

    @staticmethod
    def to_megabytes(num: int) -> float:
        """Converts bytes to megabytes."""
        return num / 1000000.0

    @staticmethod
    def to_readable(num: float, units: Units = Units.AUTO) -> tuple[Units, float]:
        """Converts a number to millions, billions, or trillions."""
        if units == Units.AUTO:
            if num >= 1000000000000.0:
                return Units.TERABYTES, num / 1000000000000.0
            if num >= 1000000000.0:
                return Units.GIGABYTES, num / 1000000000.0
            if num >= 1000000.0:
                return Units.MEGABYTES, num / 1000000.0
            if num >= 1000.0:
                return Units.KILOBYTES, num / 1000.0
            return Units.NONE, num
        return units, num / CONVERSION_FACTORS[units]

    @staticmethod
    def format_output_num(num: int, units: Units, is_bytes: bool) -> str:
        units_used, converted_num = ModelStatistics.to_readable(num, units)
        if isinstance(converted_num, float) and converted_num.is_integer():
            converted_num = int(converted_num)
        units_display = (
            ""
            if units_used == Units.NONE
            else f" ({units_used.value}{'B' if is_bytes else ''})"
        )
        fmt = "d" if isinstance(converted_num, int) else ".2f"
        return f"{units_display}: {converted_num:,{fmt}}"
