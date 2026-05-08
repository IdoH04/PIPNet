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

        # Stage 3 -> Stage 4 gating controls.
        # If use_stage3_gating is False, normal net(xs) remains the original PIP-Net path.
        self.use_stage3_gating = bool(getattr(args, "use_stage3_gating", False))
        # Default: detach Stage 3 pooled scores before the gate. This lets Stage 3
        # influence final evidence without letting the gating loss reshape earlier ConvNeXt layers.
        self.stage3_gate_detach = not bool(getattr(args, "stage3_gate_allow_backbone_grad", False))
        self.stage3_gate_bias = float(getattr(args, "stage3_gate_bias", 3.0))
        self.stage3_gate_values = None
        self.pooled_final_raw = None
        self.pooled_final_gated = None

        # Stage 3 additive-evidence controls.
        # If enabled, prediction becomes:
        #   out = out_final + alpha * out_stage3
        # where out_stage3 is produced by classifier_penultimate(pooled_stage3).
        self.use_stage3_additive_evidence = bool(getattr(args, "use_stage3_additive_evidence", False))
        self.stage3_additive_detach = not bool(getattr(args, "stage3_additive_allow_backbone_grad", False))
        self.stage3_additive_alpha_init = float(getattr(args, "stage3_additive_alpha", 0.1))
        self.stage3_additive_learnable_alpha = bool(getattr(args, "stage3_additive_learnable_alpha", False))

        if self.stage3_additive_learnable_alpha:
            alpha = min(max(self.stage3_additive_alpha_init, 1e-6), 1.0 - 1e-6)
            self.stage3_additive_alpha_logit = nn.Parameter(
                torch.logit(torch.tensor(alpha, dtype=torch.float32))
            )
        else:
            self.register_buffer(
                "stage3_additive_alpha_buffer",
                torch.tensor(self.stage3_additive_alpha_init, dtype=torch.float32)
            )

        self.out_final_only = None
        self.out_stage3_additive = None
        self.stage3_additive_alpha_value = None

        # Stage 3 -> Stage 4 matrix shaping controls.
        # This learns a matrix M that predicts/supports final Stage 4 prototype
        # presence from Stage 3 prototype presence:
        #   predicted_final_support = sigmoid(M(pooled_stage3))
        # A shaping loss then encourages final pooled Stage 4 evidence to be
        # predictable from Stage 3 evidence.
        self.use_stage3_matrix_shaping = bool(getattr(args, "use_stage3_matrix_shaping", False))
        self.stage3_matrix_detach = not bool(getattr(args, "stage3_matrix_allow_backbone_grad", False))
        self.stage3_matrix_detach_final = bool(getattr(args, "stage3_matrix_detach_final", False))
        self.stage3_matrix_loss_weight = float(getattr(args, "stage3_matrix_loss_weight", 0.1))
        self.stage3_matrix_bias = float(getattr(args, "stage3_matrix_bias", 0.0))
        self.stage3_matrix_predicted_final = None
        self.stage3_matrix_shaping_loss = None

        #Stage 3 branch for explanations
        self.has_penultimate_branch = hasattr(args, "penultimate_channels")

        if self.has_penultimate_branch:
            pen_channels = args.penultimate_channels

            # Stage3 prototypes = softmax over stage3 feature channels.
            self.add_on_penultimate = nn.Sequential(
                nn.Softmax(dim=1)
            )

            self.penultimate_num_prototypes = pen_channels

            self.pool_penultimate = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(1, 1)),
                nn.Flatten()
            )

            self.classifier_penultimate = NonNegLinear(
                pen_channels,
                num_classes,
                bias=args.bias
            )

            # Learned mapping from lower-level Stage 3 prototype presence scores
            # to gates over final Stage 4 prototype evidence.
            self.stage3_to_final_gate = nn.Linear(
                pen_channels,
                num_prototypes,
                bias=True
            )
            # Near-identity initialization: sigmoid(3.0) ~= 0.95, so the model
            # starts close to normal PIP-Net and learns deviations only if useful.
            nn.init.zeros_(self.stage3_to_final_gate.weight)
            nn.init.constant_(self.stage3_to_final_gate.bias, self.stage3_gate_bias)

            self.stage3_to_final_matrix = nn.Linear(
                pen_channels,
                num_prototypes,
                bias=True
            )
            nn.init.zeros_(self.stage3_to_final_matrix.weight)
            nn.init.constant_(self.stage3_to_final_matrix.bias, self.stage3_matrix_bias)

            # Dont train stage3 classifier branch, only use for explanations
            for p in self.classifier_penultimate.parameters():
                p.requires_grad = False

            print("Stage3 explanatory prototypes:", pen_channels, flush=True)
            if self.use_stage3_additive_evidence:
                print("Stage3 additive evidence enabled. alpha =", self.stage3_additive_alpha_init, "detach =", self.stage3_additive_detach, flush=True)
            if self.use_stage3_matrix_shaping:
                print("Stage3 matrix shaping enabled. weight =", self.stage3_matrix_loss_weight, "stage3_detach =", self.stage3_matrix_detach, "final_detach =", self.stage3_matrix_detach_final, flush=True)

        else:
            self.add_on_penultimate = None
            self.pool_penultimate = None
            self.classifier_penultimate = None
            self.stage3_to_final_gate = None
            self.stage3_to_final_matrix = None
            self.penultimate_num_prototypes = 0

    def _compute_penultimate_branch(self, stage3_features, train_stage3=False, allow_grad=False):
        """Compute Stage 3 prototype maps/scores.

        Modes:
        - train_stage3=True: old auxiliary classifier-only path; detaches Stage 3 features.
        - allow_grad=True: used by Stage 3 -> Stage 4 gating; gradients may flow through
          pooled Stage 3 scores unless stage3_gate_detach=True.
        - default: visualization-only; no gradients.
        """
        if self.add_on_penultimate is None:
            return None, None, None

        if train_stage3:
            stage3_features = stage3_features.detach()
            proto_pen = self.add_on_penultimate(stage3_features)
            pooled_pen = self.pool_penultimate(proto_pen)
            out_pen = self.classifier_penultimate(pooled_pen)

        elif allow_grad:
            # Detach Stage 3 features unless at least one active Stage 3 attachment
            # explicitly allows backbone gradients.
            detach_stage3 = True
            if self.use_stage3_gating and not self.stage3_gate_detach:
                detach_stage3 = False
            if self.use_stage3_additive_evidence and not self.stage3_additive_detach:
                detach_stage3 = False
            if self.use_stage3_matrix_shaping and not self.stage3_matrix_detach:
                detach_stage3 = False

            if detach_stage3:
                stage3_features = stage3_features.detach()

            proto_pen = self.add_on_penultimate(stage3_features)
            pooled_pen = self.pool_penultimate(proto_pen)
            # Keep this available for visualization/debugging. The Stage 3 classifier
            # is not used by the gated/matrix score unless explicitly requested.
            out_pen = self.classifier_penultimate(pooled_pen.detach())

        else:
            with torch.no_grad():
                stage3_features = stage3_features.detach()
                proto_pen = self.add_on_penultimate(stage3_features)
                pooled_pen = self.pool_penultimate(proto_pen)
                out_pen = self.classifier_penultimate(pooled_pen)

        self.proto_penultimate = proto_pen
        self.pooled_penultimate = pooled_pen
        self.out_penultimate = out_pen

        return proto_pen, pooled_pen, out_pen

    def _apply_stage3_gate(self, pooled_final, pooled_penultimate):
        """Use Stage 3 prototype presence scores to gate final prototype evidence."""
        if self.stage3_to_final_gate is None:
            raise RuntimeError(
                "use_stage3_gating=True was requested, but this PIPNet instance "
                "does not have a Stage 3 gating layer. Use convnext_tiny_multistage."
            )

        gate_input = pooled_penultimate.detach() if self.stage3_gate_detach else pooled_penultimate
        gate = torch.sigmoid(self.stage3_to_final_gate(gate_input))
        pooled_gated = pooled_final * gate

        self.stage3_gate_values = gate
        self.pooled_final_raw = pooled_final
        self.pooled_final_gated = pooled_gated
        return pooled_gated

    def _stage3_alpha(self):
        """Return Stage 3 additive-evidence scale alpha."""
        if self.stage3_additive_learnable_alpha:
            return torch.sigmoid(self.stage3_additive_alpha_logit)
        return self.stage3_additive_alpha_buffer.to(next(self.parameters()).device)

    def _combine_final_and_stage3_scores(self, out_final, pooled_penultimate):
        """Add Stage 3 class evidence to the final Stage 4 class evidence."""
        if self.classifier_penultimate is None:
            raise RuntimeError(
                "use_stage3_additive_evidence=True was requested, but this PIPNet "
                "instance does not have a Stage 3 classifier. Use convnext_tiny_multistage."
            )

        stage3_input = pooled_penultimate.detach() if self.stage3_additive_detach else pooled_penultimate
        out_stage3 = self.classifier_penultimate(stage3_input)
        alpha = self._stage3_alpha()
        out = out_final + alpha * out_stage3

        self.out_final_only = out_final
        self.out_stage3_additive = out_stage3
        self.stage3_additive_alpha_value = alpha.detach()
        return out

    def _compute_stage3_matrix_shaping_loss(self, pooled_final, pooled_penultimate):
        """Compute matrix shaping loss from Stage 3 pooled scores to Stage 4 pooled scores.

        predicted_final_support has the same shape as pooled_final. By default:
        - Stage 3 pooled scores are detached, so the matrix and final Stage 4 evidence
          are trained without reshaping the earlier backbone through this loss.
        - final pooled scores are NOT detached, so this actually shapes Stage 4 evidence
          to be predictable/supported by Stage 3 evidence.

        Use --stage3_matrix_allow_backbone_grad to let gradients flow into Stage 3.
        Use --stage3_matrix_detach_final to train only the matrix predictor without
        shaping Stage 4 pooled scores.
        """
        if self.stage3_to_final_matrix is None:
            raise RuntimeError(
                "use_stage3_matrix_shaping=True was requested, but this PIPNet "
                "instance does not have a Stage 3 -> Stage 4 matrix. Use convnext_tiny_multistage."
            )

        matrix_input = pooled_penultimate.detach() if self.stage3_matrix_detach else pooled_penultimate
        predicted_final_support = torch.sigmoid(self.stage3_to_final_matrix(matrix_input))
        target_final = pooled_final.detach() if self.stage3_matrix_detach_final else pooled_final

        # MSE keeps this as a soft support/predictability constraint instead of a
        # hard gate. The training loop multiplies this by stage3_matrix_loss_weight.
        loss = F.mse_loss(predicted_final_support, target_final)

        self.stage3_matrix_predicted_final = predicted_final_support
        self.stage3_matrix_shaping_loss = loss
        return loss

    def forward(self, xs, inference=False, compute_penultimate=False, train_stage3=False):
        if train_stage3:
            if self.add_on_penultimate is None:
                raise RuntimeError(
                    "train_stage3=True was requested, but this PIPNet instance "
                    "does not have a penultimate/stage3 branch."
                )

            with torch.no_grad():
                stages = self._net(xs, return_stage="both")
                stage3_features = stages["penultimate"]

            proto_pen, pooled_pen, out_pen = self._compute_penultimate_branch(
                stage3_features,
                train_stage3=True
            )
            return proto_pen, pooled_pen, out_pen

        # Only request both stages when needed. This keeps normal final-only PIP-Net
        # behavior intact unless Stage 3 visualization or Stage 3 gating is enabled.
        need_stage3 = self.has_penultimate_branch and (compute_penultimate or self.use_stage3_gating or self.use_stage3_additive_evidence or self.use_stage3_matrix_shaping)

        if need_stage3:
            stages = self._net(xs, return_stage="both")
            stage3_features = stages["penultimate"]
            final_features = stages["final"]

            # For visualization-only calls, keep Stage 3 detached/no-grad.
            # For gating calls, allow gradients through the gate and optionally into Stage 3.
            allow_stage3_grad = bool(self.use_stage3_gating or self.use_stage3_additive_evidence or self.use_stage3_matrix_shaping)
            _, pooled_pen, _ = self._compute_penultimate_branch(
                stage3_features,
                train_stage3=False,
                allow_grad=allow_stage3_grad
            )
        else:
            final_features = self._net(xs)
            pooled_pen = None
            self.proto_penultimate = None
            self.pooled_penultimate = None
            self.out_penultimate = None
            self.stage3_gate_values = None
            self.pooled_final_raw = None
            self.pooled_final_gated = None
            self.out_final_only = None
            self.out_stage3_additive = None
            self.stage3_additive_alpha_value = None
            self.stage3_matrix_predicted_final = None
            self.stage3_matrix_shaping_loss = None

        proto_features = self._add_on(final_features)
        pooled = self._pool(proto_features)
        self.pooled_final_raw = pooled

        if self.use_stage3_matrix_shaping:
            if pooled_pen is None:
                raise RuntimeError("Stage 3 matrix shaping requires penultimate pooled scores.")
            self._compute_stage3_matrix_shaping_loss(pooled, pooled_pen)
        else:
            self.stage3_matrix_predicted_final = None
            self.stage3_matrix_shaping_loss = None

        if self.use_stage3_gating:
            if pooled_pen is None:
                raise RuntimeError("Stage 3 gating requires penultimate pooled scores.")
            pooled_for_classification = self._apply_stage3_gate(pooled, pooled_pen)
        else:
            pooled_for_classification = pooled
            self.pooled_final_gated = None

        if inference:
            clamped_pooled = torch.where(
                pooled_for_classification < 0.1,
                torch.zeros_like(pooled_for_classification),
                pooled_for_classification
            )
            out_final = self._classification(clamped_pooled)
            if self.use_stage3_additive_evidence:
                if pooled_pen is None:
                    raise RuntimeError("Stage 3 additive evidence requires penultimate pooled scores.")
                pooled_pen_for_score = torch.where(
                    pooled_pen < 0.1,
                    torch.zeros_like(pooled_pen),
                    pooled_pen
                )
                out = self._combine_final_and_stage3_scores(out_final, pooled_pen_for_score)
            else:
                out = out_final
                self.out_final_only = out_final
                self.out_stage3_additive = None
                self.stage3_additive_alpha_value = None
            return proto_features, clamped_pooled, out
        else:
            out_final = self._classification(pooled_for_classification)
            if self.use_stage3_additive_evidence:
                if pooled_pen is None:
                    raise RuntimeError("Stage 3 additive evidence requires penultimate pooled scores.")
                out = self._combine_final_and_stage3_scores(out_final, pooled_pen)
            else:
                out = out_final
                self.out_final_only = out_final
                self.out_stage3_additive = None
                self.stage3_additive_alpha_value = None
            return proto_features, pooled_for_classification, out

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

    if args.net == "convnext_tiny_multistage":
        was_training = features.training
        features.eval()

        # Preserve RNG state so channel inference does not change classifier
        # initialization or dataloader/augmentation randomness relative to baseline.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        with torch.no_grad():
            dummy = torch.zeros(1, 3, args.image_size, args.image_size)
            out = features(dummy, return_stage="both")

        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

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
