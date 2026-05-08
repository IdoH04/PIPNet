import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim

"""
    Utility functions for handling parsed arguments

"""
def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Train a PIP-Net')
    parser.add_argument('--dataset',
                        type=str,
                        default='CUB-200-2011',
                        help='Data set on PIP-Net should be trained')
    parser.add_argument('--validation_size',
                        type=float,
                        default=0.,
                        help='Split between training and validation set. Can be zero when there is a separate test or validation directory. Should be between 0 and 1. Used for partimagenet (e.g. 0.2)')
    parser.add_argument('--net',
                        type=str,
                        default='convnext_tiny_26',
                        help='Base network used as backbone of PIP-Net. Default is convnext_tiny_26 with adapted strides to output 26x26 latent representations. Other option is convnext_tiny_13 that outputs 13x13 (smaller and faster to train, less fine-grained). Pretrained network on iNaturalist is only available for resnet50_inat. Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, convnext_tiny_26 and convnext_tiny_13.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size when training the model using minibatch gradient descent. Batch size is multiplied with number of available GPUs')
    parser.add_argument('--batch_size_pretrain',
                        type=int,
                        default=128,
                        help='Batch size when pretraining the prototypes (first training stage)')
    parser.add_argument('--epochs',
                        type=int,
                        default=60,
                        help='The number of epochs PIP-Net should be trained (second training stage)')
    parser.add_argument('--epochs_pretrain',
                        type=int,
                        default = 10,
                        help='Number of epochs to pre-train the prototypes (first training stage). Recommended to train at least until the align loss < 1'
                        )
    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        help='The optimizer that should be used when training PIP-Net')
    parser.add_argument('--lr',
                        type=float,
                        default=0.05, 
                        help='The optimizer learning rate for training the weights from prototypes to classes')
    parser.add_argument('--lr_block',
                        type=float,
                        default=0.0005, 
                        help='The optimizer learning rate for training the last conv layers of the backbone')
    parser.add_argument('--lr_net',
                        type=float,
                        default=0.0005, 
                        help='The optimizer learning rate for the backbone. Usually similar as lr_block.') 
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./runs/run_pipnet',
                        help='The directory in which train progress should be logged')
    parser.add_argument('--num_features',
                        type=int,
                        default = 0,
                        help='Number of prototypes. When zero (default) the number of prototypes is the number of output channels of backbone. If this value is set, then a 1x1 conv layer will be added. Recommended to keep 0, but can be increased when number of classes > num output channels in backbone.')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Input images will be resized to --image_size x --image_size (square). Code only tested with 224x224, so no guarantees that it works for different sizes.')
    parser.add_argument('--state_dict_dir_net',
                        type=str,
                        default='',
                        help='The directory containing a state dict with a pretrained PIP-Net. E.g., ./runs/run_pipnet/checkpoints/net_pretrained')
    parser.add_argument('--freeze_epochs',
                        type=int,
                        default = 10,
                        help='Number of epochs where pretrained features_net will be frozen while training classification layer (and last layer(s) of backbone)'
                        )
    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='visualization_results',
                        help='Directoy for saving the prototypes and explanations')
    parser.add_argument('--disable_pretrained',
                        action='store_true',
                        help='When set, the backbone network is initialized with random weights instead of being pretrained on another dataset).'
                        )
    parser.add_argument('--weighted_loss',
                        action='store_true',
                        help='Flag that weights the loss based on the class balance of the dataset. Recommended to use when data is imbalanced. ')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed. Note that there will still be differences between runs due to nondeterminism. See https://pytorch.org/docs/stable/notes/randomness.html')
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='',
                        help='ID of gpu. Can be separated with comma')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Num workers in dataloaders.')
    parser.add_argument('--bias',
                        action='store_true',
                        help='Flag that indicates whether to include a trainable bias in the linear classification layer.'
                        )
    parser.add_argument('--use_stage3_gating',
                        action='store_true',
                        help='Use Stage 3 prototype presence scores to gate final Stage 4 prototype evidence. Requires --net convnext_tiny_multistage.'
                        )
    parser.add_argument('--stage3_gate_allow_backbone_grad',
                        action='store_true',
                        help='If set, gradients from the final gated classification loss may flow back through the Stage 3 branch. Default is safer: detach Stage 3 pooled scores before the gate.'
                        )
    parser.add_argument('--stage3_gate_bias',
                        type=float,
                        default=3.0,
                        help='Initial bias for the Stage 3 -> Stage 4 gate. 3.0 gives sigmoid(3) ~= 0.95, close to identity.'
                        )
    parser.add_argument('--stage3_gate_lr_multiplier',
                        type=float,
                        default=1.0,
                        help='Multiplier on lr_block for the Stage 3 -> Stage 4 gate optimizer group.'
                        )

    parser.add_argument('--use_stage3_additive_evidence',
                        action='store_true',
                        help='Add Stage 3 class evidence directly to the final Stage 4 class score. Requires --net convnext_tiny_multistage.'
                        )
    parser.add_argument('--stage3_additive_allow_backbone_grad',
                        action='store_true',
                        help='Allow gradients from the Stage 3 additive evidence term to flow into the Stage 3/backbone path. By default Stage 3 pooled scores are detached.'
                        )
    parser.add_argument('--stage3_additive_alpha',
                        type=float,
                        default=0.1,
                        help='Scale alpha for additive Stage 3 evidence: out = out_final + alpha * out_stage3.'
                        )
    parser.add_argument('--stage3_additive_learnable_alpha',
                        action='store_true',
                        help='Make the Stage 3 additive-evidence scale alpha learnable via a sigmoid-constrained parameter.'
                        )
    parser.add_argument('--stage3_additive_lr_multiplier',
                        type=float,
                        default=1.0,
                        help='Learning-rate multiplier for Stage 3 additive parameters, including classifier_penultimate and optional alpha.'
                        )
    parser.add_argument('--use_stage3_matrix_shaping',
                        action='store_true',
                        help='Use a learned Stage 3 -> Stage 4 matrix and shaping loss so final Stage 4 prototype scores are predictable from Stage 3 prototype scores.'
                        )
    parser.add_argument('--stage3_matrix_allow_backbone_grad',
                        action='store_true',
                        help='Allow the Stage 3 -> Stage 4 matrix shaping loss to flow into Stage 3/backbone features. Default detaches Stage 3 pooled scores.'
                        )
    parser.add_argument('--stage3_matrix_detach_final',
                        action='store_true',
                        help='Detach final Stage 4 pooled scores in the matrix shaping loss. This trains only the matrix predictor instead of shaping Stage 4 evidence.'
                        )
    parser.add_argument('--stage3_matrix_loss_weight',
                        type=float,
                        default=0.01,
                        help='Weight for the Stage 3 -> Stage 4 matrix shaping loss added to the original PIP-Net loss.'
                        )
    parser.add_argument('--stage3_matrix_bias',
                        type=float,
                        default=-3.0,
                        help='Initial bias for the Stage 3 -> Stage 4 matrix. 0.0 gives sigmoid(0)=0.5 initial predicted support.'
                        )
    parser.add_argument('--stage3_matrix_lr_multiplier',
                        type=float,
                        default=1.0,
                        help='Learning-rate multiplier for the Stage 3 -> Stage 4 matrix parameters.'
                        )
    parser.add_argument('--extra_test_image_folder',
                        type=str,
                        default='./experiments',
                        help='Folder with images that PIP-Net will predict and explain, that are not in the training or test set. E.g. images with 2 objects or OOD image. Images should be in subfolder. E.g. images in ./experiments/images/, and argument --./experiments')

    args = parser.parse_args()
    if len(args.log_dir.split('/'))>2:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)



    if args.use_stage3_gating and args.net != 'convnext_tiny_multistage':
        raise ValueError('--use_stage3_gating requires --net convnext_tiny_multistage')

    if args.use_stage3_additive_evidence and args.net != 'convnext_tiny_multistage':
        raise ValueError('--use_stage3_additive_evidence requires --net convnext_tiny_multistage')

    if args.use_stage3_matrix_shaping and args.net != 'convnext_tiny_multistage':
        raise ValueError('--use_stage3_matrix_shaping requires --net convnext_tiny_multistage')

    if sum([
        bool(args.use_stage3_gating),
        bool(args.use_stage3_additive_evidence),
        bool(args.use_stage3_matrix_shaping)
    ]) > 1:
        raise ValueError(
            'Use only one Stage 3 attachment method at a time: gating, additive evidence, or matrix shaping.'
        )

    if args.stage3_additive_alpha < 0.0:
        raise ValueError('--stage3_additive_alpha must be non-negative')

    if args.stage3_matrix_loss_weight < 0.0:
        raise ValueError('--stage3_matrix_loss_weight must be non-negative')

    return args


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)                                                                               
    
def get_optimizer_nn(net, args: argparse.Namespace) -> torch.optim.Optimizer:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []
    # set up optimizer
    if 'resnet50' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            elif 'layer2' in name:
                params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                param.requires_grad = False
                # params_backbone.append(param)
    
    elif 'convnext' in args.net:
        print("chosen network is convnext", flush=True)
        for name,param in net.module._net.named_parameters():
            if 'features.7.2' in name: 
                params_to_train.append(param)
            elif 'features.7' in name or 'features.6' in name:
                params_to_freeze.append(param)
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone.append(param)
    else:
        print("Network is not ResNet or ConvNext.", flush=True)     
    classification_weight = []
    classification_bias = []
    for name, param in net.module._classification.named_parameters():
        if 'weight' in name:
            classification_weight.append(param)
        elif 'multiplier' in name:
            param.requires_grad = False
        else:
            if args.bias:
                classification_bias.append(param)
    
    stage3_gate_params = []
    if bool(getattr(args, 'use_stage3_gating', False)):
        if hasattr(net.module, 'stage3_to_final_gate') and net.module.stage3_to_final_gate is not None:
            stage3_gate_params = list(net.module.stage3_to_final_gate.parameters())
        else:
            raise ValueError('--use_stage3_gating requires a network with a Stage 3 branch, e.g. --net convnext_tiny_multistage')

    stage3_matrix_params = []
    if bool(getattr(args, 'use_stage3_matrix_shaping', False)):
        if hasattr(net.module, 'stage3_to_final_matrix') and net.module.stage3_to_final_matrix is not None:
            stage3_matrix_params = list(net.module.stage3_to_final_matrix.parameters())
        else:
            raise ValueError('--use_stage3_matrix_shaping requires a Stage 3 matrix branch')

    stage3_additive_params = []
    if bool(getattr(args, 'use_stage3_additive_evidence', False)):
        if hasattr(net.module, 'classifier_penultimate') and net.module.classifier_penultimate is not None:
            stage3_additive_params = list(net.module.classifier_penultimate.parameters())
            if getattr(net.module, 'stage3_additive_learnable_alpha', False) and hasattr(net.module, 'stage3_additive_alpha_logit'):
                stage3_additive_params.append(net.module.stage3_additive_alpha_logit)
        else:
            raise ValueError('--use_stage3_additive_evidence requires a Stage 3 classifier branch')


    paramlist_net = [
            {"params": params_backbone, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
            {"params": params_to_freeze, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": net.module._add_on.parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay}]

    if stage3_gate_params:
        paramlist_net.append({
            "params": stage3_gate_params,
            "lr": args.lr_block * args.stage3_gate_lr_multiplier,
            "weight_decay_rate": args.weight_decay
        })

    if stage3_matrix_params:
        paramlist_net.append({
            "params": stage3_matrix_params,
            "lr": args.lr_block * args.stage3_matrix_lr_multiplier,
            "weight_decay_rate": args.weight_decay
        })

    if stage3_additive_params:
        paramlist_net.append({
            "params": stage3_additive_params,
            "lr": args.lr_block * args.stage3_additive_lr_multiplier,
            "weight_decay_rate": args.weight_decay
        })
            
    paramlist_classifier = [
            {"params": classification_weight, "lr": args.lr, "weight_decay_rate": args.weight_decay},
            {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},
    ]
          
    if args.optimizer == 'Adam':
        optimizer_net = torch.optim.AdamW(paramlist_net,lr=args.lr,weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier,lr=args.lr,weight_decay=args.weight_decay)
        return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone
    else:
        raise ValueError("this optimizer type is not implemented")

