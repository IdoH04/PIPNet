import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features, ConvNextMultiStage
import torch
from torch import Tensor


class PIPNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 classification_layer: nn.Module
                 ):
        super().__init__()
        assert num_classes > 0

        # Original PIP-Net branch
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

        self.has_penultimate_branch = hasattr(args, "penultimate_channels")

        if self.has_penultimate_branch:
            pen_channels = args.penultimate_channels
            pen_num_prototypes = getattr(args, "penultimate_num_features", 0)

            if pen_num_prototypes == 0:
                pen_num_prototypes = pen_channels
                self.add_on_penultimate = nn.Sequential(
                    nn.Softmax(dim=1)
                )
                print("Stage3 explanatory prototypes:", pen_num_prototypes, flush=True)
            else:
                self.add_on_penultimate = nn.Sequential(
                    nn.Conv2d(
                        in_channels=pen_channels,
                        out_channels=pen_num_prototypes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True
                    ),
                    nn.Softmax(dim=1)
                )
                print("Stage3 explanatory prototypes set from",
                      pen_channels, "to", pen_num_prototypes,
                      flush=True)

            self.penultimate_num_prototypes = pen_num_prototypes

            self.pool_penultimate = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(1, 1)),
                nn.Flatten()
            )

            self.classifier_penultimate = NonNegLinear(
                pen_num_prototypes,
                num_classes,
                bias=args.bias
            )

            # Freeze stage3 explanatory branch so it cannot affect training.
            for p in self.add_on_penultimate.parameters():
                p.requires_grad = False
            for p in self.pool_penultimate.parameters():
                p.requires_grad = False
            for p in self.classifier_penultimate.parameters():
                p.requires_grad = False
        else:
            self.add_on_penultimate = None
            self.pool_penultimate = None
            self.classifier_penultimate = None
            self.penultimate_num_prototypes = 0

       
        self.proto_penultimate = None
        self.pooled_penultimate = None
        self.out_penultimate = None

    def _compute_penultimate_for_visualization(self, stage3_features):
        if self.add_on_penultimate is None:
            return None, None, None

        with torch.no_grad():
            stage3_features = stage3_features.detach()
            proto_pen = self.add_on_penultimate(stage3_features)
            pooled_pen = self.pool_penultimate(proto_pen)
            out_pen = self.classifier_penultimate(pooled_pen)

        self.proto_penultimate = proto_pen
        self.pooled_penultimate = pooled_pen
        self.out_penultimate = out_pen

        return proto_pen, pooled_pen, out_pen

    def forward(self, xs, inference=False, compute_penultimate=False):
        if compute_penultimate or inference:
            stages = self._net(xs, return_stage="both")
            stage3_features = stages["penultimate"]
            final_features = stages["final"]

            self._compute_penultimate_for_visualization(stage3_features)
        else:
            final_features = self._net(xs)
            self.proto_penultimate = None
            self.pooled_penultimate = None
            self.out_penultimate = None

        proto_features = self._add_on(final_features)
        pooled = self._pool(proto_features)

        if inference:
            clamped_pooled = torch.where(pooled < 0.1, 0., pooled)
            out = self._classification(clamped_pooled)
            return proto_features, clamped_pooled, out
        else:
            out = self._classification(pooled)
            return proto_features, pooled, out

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'convnext_tiny_26': convnext_tiny_26_features,
                                 'convnext_tiny_13': convnext_tiny_13_features,
                                 'convnext_tiny_multistage': ConvNextMultiStage}


# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,), requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.relu(self.weight), self.bias)


def _infer_feature_channels(features, args):
    features_name = str(features).upper()
    if 'next' in args.net:
        features_name = str(args.net).upper()

    # Important: explicitly ask multistage backbone for both stages
    if args.net == "convnext_tiny_multistage":
        was_training = features.training
        features.eval()

        with torch.no_grad():
            dummy = torch.randn(1, 3, args.image_size, args.image_size)
            out = features(dummy, return_stage="both")

        if was_training:
            features.train()

        args.penultimate_channels = out["penultimate"].shape[1]
        first_add_on_layer_in_channels = out["final"].shape[1]

        print("Penultimate channels:", args.penultimate_channels, flush=True)
        print("Final channels:", first_add_on_layer_in_channels, flush=True)

        return first_add_on_layer_in_channels

    # Original PIP-Net behavior
    if features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    else:
        raise Exception('other base architecture NOT implemented')

    return first_add_on_layer_in_channels

def get_network(num_classes: int, args: argparse.Namespace):
    features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)

    first_add_on_layer_in_channels = _infer_feature_channels(features, args)

    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        print("Number of prototypes: ", num_prototypes, flush=True)
        add_on_layers = nn.Sequential(
            nn.Softmax(dim=1),  # softmax over prototypes for each patch
        )
    else:
        num_prototypes = args.num_features
        print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_prototypes,
              ". Extra 1x1 conv layer added. Not recommended.", flush=True)
        add_on_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=first_add_on_layer_in_channels,
                out_channels=num_prototypes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Softmax(dim=1),
        )

    pool_layer = nn.Sequential(
        nn.AdaptiveMaxPool2d(output_size=(1, 1)),
        nn.Flatten()
    )

    if args.bias:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)
    else:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)

    return features, add_on_layers, pool_layer, classification_layer, num_prototypes
