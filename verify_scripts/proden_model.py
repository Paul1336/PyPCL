"""12-layer ConvNet (Laine & Aila, 2017) architecture, exactly as specified in
PRODEN's own paper (Lv, Xu, Feng, Niu, Geng, Sugiyama, ICML 2020) for the
CIFAR-10 experiments -- confirmed by direct PDF text extraction from
PRODEN.pdf (see verify_scripts/proden_verify.py docstring for the quoted
passage). This architecture is PRODEN-specific, so it lives in this
PRODEN-scoped file rather than a shared verify_scripts/models.py (other
paper-verification scripts are being added to verify_scripts/ in parallel
and should not collide on a generic filename).

Paper's exact architecture description (quoted):
    "0th (input) layer: (32*32*3)- 1st to 4th layers: [C(3*3, 128)]*3-Max
    Pooling- 5th to 8th layers: [C(3*3, 256)]*3-Max Pooling- 9th to 11th
    layers: C(3*3, 512)-C(3*3, 256)-C(3*3, 128)- 12th layers: Average
    Pooling-10 where C(3*3, 128) means 128 channels of 3*3 convolutions
    followed by Leaky-ReLU (LReLU) active function (Maas et al., 2013)"

i.e. (counting the two max-pool layers and the final avgpool+linear as
"layers" too, which is how the paper gets to "12 layers"):
    input (32x32x3)
    -> [Conv3x3(128) -> LReLU] x3
    -> MaxPool2x2                                  (layer 4)
    -> [Conv3x3(256) -> LReLU] x3
    -> MaxPool2x2                                  (layer 8)
    -> Conv3x3(512) -> LReLU
    -> Conv3x3(256) -> LReLU
    -> Conv3x3(128) -> LReLU                       (layers 9-11)
    -> GlobalAveragePool -> Linear(128, 10)        (layer 12)

ASSUMPTIONS (paper text does not spell these out for the ConvNet; flagged
per the task instructions rather than silently guessed):
  - BatchNorm after every conv, before the LReLU. The paper's Appendix E.1
    only explicitly mentions "Batch normalization ... applied before hidden
    layers" for the MNIST MLP model, not for the CIFAR-10 ConvNet. BatchNorm
    is standard convention for this well-known Laine & Aila "conv-large"
    architecture family (used with BN in the original Temporal Ensembling
    paper and in follow-on noisy-/partial-label work that reuses this exact
    architecture), so it is included here. Can be toggled off via
    PRODENConvNet(use_batchnorm=False) if a later comparison run wants the
    literal bare-conv variant.
  - LeakyReLU negative_slope=0.1. The paper cites Maas et al. 2013 (original
    LReLU paper, alpha=0.01 default) but names the specific architecture
    "ConvNet (Laine & Aila, 2017)", whose own paper specifies LReLU(0.1) for
    this exact conv-large network. 0.1 is used here as the better-sourced
    value for the *architecture actually being cited*, not PyTorch's
    LeakyReLU default (0.01).
  - Padding=1 (SAME padding) on every 3x3 conv, implied by the fact that the
    paper never mentions spatial downsampling except at the two named
    Max-Pooling layers -- consistent with the standard Laine & Aila conv-large
    architecture, which uses SAME-padded 3x3 convs throughout.
"""

import torch.nn as nn


class PRODENConvNet(nn.Module):
    """Paper-exact 12-layer ConvNet for PRODEN's CIFAR-10 experiments.

    Args:
        num_classes: output classes (10 for CIFAR-10).
        in_channels: input image channels (3 for CIFAR-10 RGB).
        leaky_slope: LeakyReLU negative slope (see module docstring assumption).
        use_batchnorm: include BatchNorm2d after each conv (see module docstring
            assumption -- default True, standard for this architecture family).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3,
                 leaky_slope: float = 0.1, use_batchnorm: bool = True):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        def conv_unit(c_in: int, c_out: int) -> nn.Sequential:
            layers = [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
            return nn.Sequential(*layers)

        # 1st-4th layers: [C(3x3,128)]x3 - MaxPool
        self.stage1 = nn.Sequential(
            conv_unit(in_channels, 128),
            conv_unit(128, 128),
            conv_unit(128, 128),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 5th-8th layers: [C(3x3,256)]x3 - MaxPool
        self.stage2 = nn.Sequential(
            conv_unit(128, 256),
            conv_unit(256, 256),
            conv_unit(256, 256),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 9th-11th layers: C(3x3,512)-C(3x3,256)-C(3x3,128)
        self.stage3 = nn.Sequential(
            conv_unit(256, 512),
            conv_unit(512, 256),
            conv_unit(256, 128),
        )

        # 12th layer: Average Pooling - 10 (global avg pool + linear classifier)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.fc(x)
