# @Time   : 2025/9/12 下午10:21
# @Author : Chas
# @File   : test.py
# @desc   :
# Copyright (c) 2025 Chas-OUC. All Rights Reserved.
import torch

from ultralytics.nn.modules.NewAttention.FACAttention import FACAttention


def test_fac_attention():
    """测试FAC-Attention模块"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建测试输入
    batch_size, channels, height, width = 4, 160, 64, 64
    x = torch.randn(batch_size, channels, height, width).to(device)

    # 测试不同配置
    configs = [
        {'mode': 3, 'use_multi_scale': True, 'heads': 8, 'reduction': 16},
        {'mode': 1, 'use_multi_scale': False, 'heads': 8, 'reduction': 16},
        {'mode': 2, 'use_multi_scale': True, 'heads': 4, 'reduction': 8}
    ]

    for i, config in enumerate(configs):
        print(f"\nTesting config {i + 1}: {config}")

        # 正确的调用方式：所有参数都通过 config 字典传递
        model = FACAttention(
            channels=channels,
            **config
        ).to(device)

        # 前向传播
        with torch.no_grad():
            output = model(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

        # 检查输出是否合理
        assert output.shape == x.shape, "Output shape should match input shape"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

        print("✓ Test passed!")


# 如果你想要更清晰的测试，可以使用这种方式：
def test_fac_attention_alternative():
    """另一种测试方式"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size, channels, height, width = 4, 512, 32, 32
    x = torch.randn(batch_size, channels, height, width).to(device)

    # 测试1: 默认配置
    print("Testing default configuration...")
    model1 = FACAttention(channels=channels).to(device)
    output1 = model1(x)
    print(f"Default config - Params: {sum(p.numel() for p in model1.parameters()):,}")

    # 测试2: ECA + 多尺度
    print("Testing ECA + multi-scale configuration...")
    model2 = FACAttention(
        channels=channels,
        mode=1,
        use_multi_scale=True,
        heads=4,
        reduction=8
    ).to(device)
    output2 = model2(x)
    print(f"ECA config - Params: {sum(p.numel() for p in model2.parameters()):,}")

    # 测试3: SE + 单尺度
    print("Testing SE + single-scale configuration...")
    model3 = FACAttention(
        channels=channels,
        mode=2,
        use_multi_scale=False,
        heads=8,
        reduction=16
    ).to(device)
    output3 = model3(x)
    print(f"SE config - Params: {sum(p.numel() for p in model3.parameters()):,}")

    # 验证所有输出形状一致
    assert output1.shape == output2.shape == output3.shape == x.shape
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_fac_attention()
    # test_fac_attention_alternative()
