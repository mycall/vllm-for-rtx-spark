"""Exercise the FP4-Local output-head shape on the installed Windows vLLM wheel."""

import torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
)


def main():
    vocab, hidden = 248320, 2560
    device = torch.device("cuda:0")
    torch.manual_seed(54)
    packed = torch.randint(256, (vocab, hidden // 2), dtype=torch.uint8, device=device)
    # Alternate group scales across the input dimension to check scale repacking.
    group_scales = torch.tensor([1.0, 2.0], device=device).repeat(hidden // 32)
    scales = group_scales.to(torch.float8_e4m3fn).repeat(vocab, 1)
    global_scale = 0.5
    layer = torch.nn.Module()
    layer.input_size_per_partition = hidden
    layer.output_size_per_partition = vocab
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor(global_scale, device=device), requires_grad=False
    )
    prepare_fp4_layer_for_marlin(layer)

    # With one-hot inputs each output is one dequantized weight. This reference
    # reads the original E2M1 nibbles independently of Marlin's packed layout.
    values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    lut = torch.tensor(values + [-value for value in values], dtype=torch.bfloat16, device=device)

    def project(inputs):
        return apply_fp4_marlin_linear(
            inputs, layer.weight, layer.weight_scale, layer.weight_global_scale, None, vocab, hidden
        )

    for batch in (1, 4, 8):
        rows = torch.arange(batch, device=device)
        columns = rows + 127
        inputs = torch.zeros((batch, hidden), dtype=torch.bfloat16, device=device)
        inputs[rows, columns] = 1
        raw = packed[:, columns // 2].T
        codes = torch.where(columns[:, None] % 2 == 0, raw & 15, raw >> 4)
        expected = (lut[codes.long()] * group_scales[columns // 16, None] * global_scale).to(torch.bfloat16)
        torch.testing.assert_close(project(inputs), expected, atol=0, rtol=0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = project(inputs)
        inputs.neg_()
        graph.replay()
        torch.testing.assert_close(output, -expected, atol=0, rtol=0)
        print(f"Batch size {batch} matched all {vocab} logits per input in eager execution and graph replay.")


if __name__ == "__main__":
    main()
