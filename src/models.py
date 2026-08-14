import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.resnet = models.resnet18(num_classes=num_classes, weights=None)

        # Modify the first convolutional layer for small images (e.g., CIFAR)
        # and/or a non-RGB channel count (e.g. grayscale MNIST-family).
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Re-initialize the weights of the modified layer.
        nn.init.kaiming_normal_(
            self.resnet.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x):
        return self.resnet(x)


def create_model(num_classes, in_channels=3):
    """Creates a ResNet-18 model."""
    model = ResNet18(num_classes=num_classes, in_channels=in_channels)
    return model


class MLPClassifier(nn.Module):
    """Simple MLP backbone for non-image (tabular / pre-extracted-feature)
    datasets, where a CNN's spatial-convolution assumption doesn't apply."""

    def __init__(self, input_dim, num_classes, hidden_dims=(128, 128)):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def create_mlp_model(input_dim, num_classes, hidden_dims=(128, 128)):
    return MLPClassifier(input_dim, num_classes, hidden_dims=hidden_dims)


def create_model_for_spec(spec, num_classes):
    """Dispatches to the right backbone for a DatasetSpec (see
    src/pipeline/datasets/specs.py). `spec=None` means the default/original
    CIFAR-100-subset path -- kept for exact backward compatibility."""
    if spec is None:
        return create_model(num_classes)
    if spec.backbone == 'mlp':
        if spec.input_dim is None:
            raise ValueError(f"DatasetSpec '{spec.name}' has backbone='mlp' but no input_dim set")
        return create_mlp_model(spec.input_dim, num_classes)
    return create_model(num_classes, in_channels=spec.in_channels)
