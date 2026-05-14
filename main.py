from pipnet.pipnet import PIPNet, get_network
from util.log import Log
import torch.nn as nn
import torch.nn.functional as F
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
from util.func import init_weights_xavier
from pipnet.train import train_pipnet
from pipnet.test import eval_pipnet, get_thresholds, eval_ood
from util.eval_cub_csv import eval_prototypes_cub_parts_csv, get_topk_cub, get_proto_patches_cub
import torch
from util.vis_pipnet import visualize, visualize_topk, visualize_both_stages
from util.visualize_prediction import vis_pred, vis_pred_experiments
from util.distribution_matching import compute_class_centroids
import sys, os
import random
import numpy as np
from shutil import copy
import matplotlib.pyplot as plt
from copy import deepcopy


def set_stage3_gate_enabled(net, enabled):
    """Turn Stage 3 -> Stage 4 gating on/off without changing command-line args."""
    model = net.module if hasattr(net, "module") else net
    if hasattr(model, "use_stage3_gating"):
        model.use_stage3_gating = bool(enabled)


def set_stage3_matrix_enabled(net, enabled):
    """Turn Stage 3 -> Stage 4 matrix shaping on/off without changing command-line args."""
    model = net.module if hasattr(net, "module") else net
    if hasattr(model, "use_stage3_matrix_shaping"):
        model.use_stage3_matrix_shaping = bool(enabled)


def set_stage3_stage4_distribution_loss_active(net, active):
    """Turn Version A distribution loss computation on/off."""
    model = net.module if hasattr(net, "module") else net
    if hasattr(model, "stage3_stage4_distribution_loss_active"):
        model.stage3_stage4_distribution_loss_active = bool(active)


def set_stage3_gate_trainable(net, trainable):
    """Control whether the learned Stage 3 -> Stage 4 gate can update."""
    model = net.module if hasattr(net, "module") else net
    if hasattr(model, "stage3_to_final_gate") and model.stage3_to_final_gate is not None:
        for p in model.stage3_to_final_gate.parameters():
            p.requires_grad = bool(trainable)


def set_stage3_matrix_trainable(net, trainable):
    """Control whether the learned Stage 3 -> Stage 4 matrix can update."""
    model = net.module if hasattr(net, "module") else net
    if hasattr(model, "stage3_to_final_matrix") and model.stage3_to_final_matrix is not None:
        for p in model.stage3_to_final_matrix.parameters():
            p.requires_grad = bool(trainable)


def set_stage3_additive_trainable(net, trainable):
    model = net.module if hasattr(net, "module") else net

    if hasattr(model, "classifier_penultimate") and model.classifier_penultimate is not None:
        for p in model.classifier_penultimate.parameters():
            p.requires_grad = bool(trainable)

    if hasattr(model, "stage3_additive_alpha_logit"):
        model.stage3_additive_alpha_logit.requires_grad = bool(trainable)



def train_stage3_classifier_only(net, dataloader, device, args, num_epochs=5, lr=0.001):
    print("\nTraining stage3 classifier only...", flush=True)

    if not hasattr(net.module, "classifier_penultimate") or net.module.classifier_penultimate is None:
        print("No stage3/penultimate classifier exists; skipping stage3 training.", flush=True)
        return

    for p in net.module.parameters():
        p.requires_grad = False

    for p in net.module.classifier_penultimate.parameters():
        p.requires_grad = False

    net.module.classifier_penultimate.weight.requires_grad = True
    stage3_params = [net.module.classifier_penultimate.weight]

    if net.module.classifier_penultimate.bias is not None:
        net.module.classifier_penultimate.bias.requires_grad = True
        stage3_params.append(net.module.classifier_penultimate.bias)

    if hasattr(net.module.classifier_penultimate, "normalization_multiplier"):
        net.module.classifier_penultimate.normalization_multiplier.requires_grad = False

    optimizer_stage3 = torch.optim.Adam(stage3_params, lr=lr)
    criterion_stage3 = nn.NLLLoss(reduction='mean').to(device)

    for epoch in range(1, num_epochs + 1):
        # Keep the backbone in eval mode so stage3 classifier training cannot
        # alter/dropout/normalization behavior. The classifier still gets grads.
        net.eval()
        net.module.classifier_penultimate.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch in dataloader:
            if len(batch) == 3:
                xs, _, ys = batch
            else:
                xs, ys = batch

            xs = xs.to(device)
            ys = ys.to(device)

            optimizer_stage3.zero_grad(set_to_none=True)

            # Computes only the stage3 branch. The final PIP-Net branch is not
            # evaluated, so this cannot destabilize final/stage4 training.
            _, _, out_stage3 = net(xs, train_stage3=True)
            log_probs_stage3 = F.log_softmax(out_stage3, dim=1)

            loss = criterion_stage3(log_probs_stage3, ys)
            loss.backward()
            optimizer_stage3.step()

            with torch.no_grad():
                net.module.classifier_penultimate.weight.copy_(
                    torch.clamp(net.module.classifier_penultimate.weight.data - 1e-3, min=0.)
                )
                if net.module.classifier_penultimate.bias is not None:
                    net.module.classifier_penultimate.bias.copy_(
                        torch.clamp(net.module.classifier_penultimate.bias.data, min=0.)
                    )

            total_loss += loss.item() * xs.shape[0]
            preds = torch.argmax(out_stage3, dim=1)
            total_correct += torch.sum(preds == ys).item()
            total_seen += xs.shape[0]

        mean_loss = total_loss / total_seen
        acc = total_correct / total_seen

        print(
            "Stage3 Epoch",
            epoch,
            "loss:",
            round(mean_loss, 4),
            "acc:",
            round(acc, 4),
            flush=True
        )

    for p in net.module.classifier_penultimate.parameters():
        p.requires_grad = False

    net.eval()
    print("Finished training stage3 classifier.", flush=True)


def visualize_stage3_snapshot(net, projectloader, classes, device, args, foldername, k=10):
    """Save top-k Stage 3 prototype visualizations at a named training snapshot.
    """
    if not hasattr(net.module, "classifier_penultimate") or net.module.classifier_penultimate is None:
        print(f"Skipping {foldername}: no stage3/penultimate branch exists.", flush=True)
        return None

    print(f"\nSaving Stage3 snapshot: {foldername}", flush=True)
    was_training = net.training
    net.eval()

    with torch.no_grad():
        topks = visualize_topk(
            net,
            projectloader,
            len(classes),
            device,
            foldername,
            args,
            k=k,
            branch="stage3"
        )

    if was_training:
        net.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return topks

def run_pipnet(args=None):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args = args or get_args()
    assert args.batch_size > 1

    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    # Log the run arguments
    save_args(args, log.metadata_dir)
    
    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    
    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids)==0:
            device = torch.device('cuda')
            print("CUDA device set without id specification", flush=True)
            device_ids.append(torch.cuda.current_device())
        else:
            print("This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.", flush=True)
            device_str = ''
            for d in device_ids:
                device_str+=str(d)
                device_str+=","
            device = torch.device('cuda:'+str(device_ids[0]))
    else:
        device = torch.device('cpu')
     
    # Log which device was actually used
    print("Device used: ", device, "with id", device_ids, flush=True)
    
    # Obtain the dataset and dataloaders
    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)
    if len(classes)<=20:
        if args.validation_size == 0.:
            print("Classes: ", testloader.dataset.class_to_idx, flush=True)
        else:
            print("Classes: ", str(classes), flush=True)
    
    # Create a convolutional network based on arguments and add 1x1 conv layer
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(len(classes), args)
   
    # Create a PIP-Net
    net = PIPNet(num_classes=len(classes),
                    num_prototypes=num_prototypes,
                    feature_net = feature_net,
                    args = args,
                    add_on_layers = add_on_layers,
                    pool_layer = pool_layer,
                    classification_layer = classification_layer
                    )
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids = device_ids)    

    # Keep Stage 3 gating disabled during prototype pretraining. Pretraining should
    # stay close to original PIP-Net and should not train the gate through tanh loss.
    requested_stage3_gating = bool(getattr(args, 'use_stage3_gating', False))
    requested_stage3_additive = bool(getattr(args, 'use_stage3_additive_evidence', False))
    requested_stage3_matrix = bool(getattr(args, 'use_stage3_matrix_shaping', False))
    requested_distribution_loss = bool(getattr(args, 'use_stage3_stage4_distribution_loss', False))
    set_stage3_gate_enabled(net, False)
    set_stage3_matrix_enabled(net, False)
    set_stage3_stage4_distribution_loss_active(net, False)
    set_stage3_gate_trainable(net, False)
    set_stage3_matrix_trainable(net, False)
    set_stage3_additive_trainable(net, False)
    
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)   

    # Initialize or load model
    with torch.no_grad():
        if args.state_dict_dir_net != '':
            epoch = 0
            checkpoint = torch.load(args.state_dict_dir_net,map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            print("Pretrained network loaded", flush=True)
            net.module._multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict']) 
            except:
                pass
            if torch.mean(net.module._classification.weight).item() > 1.0 and torch.mean(net.module._classification.weight).item() < 3.0 and torch.count_nonzero(torch.relu(net.module._classification.weight-1e-5)).float().item() > 0.8*(num_prototypes*len(classes)): #assume that the linear classification layer is not yet trained (e.g. when loading a pretrained backbone only)
                print("We assume that the classification layer is not yet trained. We re-initialize it...", flush=True)
                torch.nn.init.normal_(net.module._classification.weight, mean=1.0,std=0.1)
                if net.module.classifier_penultimate is not None:
                    torch.nn.init.normal_(net.module.classifier_penultimate.weight, mean=1.0, std=0.1)
                    if hasattr(net.module.classifier_penultimate, "normalization_multiplier"):
                        torch.nn.init.constant_(net.module.classifier_penultimate.normalization_multiplier, val=2.)
                        net.module.classifier_penultimate.normalization_multiplier.requires_grad = False
                    if net.module.classifier_penultimate.bias is not None:
                        torch.nn.init.constant_(net.module.classifier_penultimate.bias, val=0.)
                torch.nn.init.constant_(net.module._multiplier, val=2.)
                print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item(), flush=True)
                if args.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.)
            # else: #uncomment these lines if you want to load the optimizer too
            #     if 'optimizer_classifier_state_dict' in checkpoint.keys():
            #         optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
            
        else:
            net.module._add_on.apply(init_weights_xavier)
            torch.nn.init.normal_(net.module._classification.weight, mean=1.0,std=0.1)

            if net.module.classifier_penultimate is not None:
                torch.nn.init.normal_(net.module.classifier_penultimate.weight, mean=1.0, std=0.1)
                if hasattr(net.module.classifier_penultimate, "normalization_multiplier"):
                    torch.nn.init.constant_(net.module.classifier_penultimate.normalization_multiplier, val=2.)
                    net.module.classifier_penultimate.normalization_multiplier.requires_grad = False
                if net.module.classifier_penultimate.bias is not None:
                    torch.nn.init.constant_(net.module.classifier_penultimate.bias, val=0.)

            if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val=0.)
            torch.nn.init.constant_(net.module._multiplier, val=2.)
            net.module._multiplier.requires_grad = False

            print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item(), flush=True)
    
    # Define classification loss function and scheduler
    criterion = nn.NLLLoss(reduction='mean').to(device)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader_pretraining)*args.epochs_pretrain, eta_min=args.lr_block/100., last_epoch=-1)

    # Forward one batch through the backbone to get the latent output size
    xs1, _, _ = next(iter(trainloader))
    xs1 = xs1.to(device)
    proto_features, _, _ = net(xs1)
    wshape = proto_features.shape[-1]
    args.wshape = wshape #needed for calculating image patch size
    print("Output shape: ", proto_features.shape, flush=True)

    # Snapshot 1: Stage3 before any PIP-Net training.
    visualize_stage3_snapshot(
        net,
        projectloader,
        classes,
        device,
        args,
        'stage3_snapshot_00_before_training_topk',
        k=10
    )
    
    if net.module._num_classes == 2:
        # Create a csv log for storing the test accuracy, F1-score, mean train accuracy and mean loss for each epoch
        log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 'test_f1', 'almost_sim_nonzeros', 'local_size_all_classes','almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
        print("Your dataset only has two classes. Is the number of samples per class similar? If the data is imbalanced, we recommend to use the --weighted_loss flag to account for the imbalance.", flush=True)
    else:
        # Create a csv log for storing the test accuracy (top 1 and top 5), mean train accuracy and mean loss for each epoch
        log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 'test_top5_acc', 'almost_sim_nonzeros', 'local_size_all_classes','almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
    
    
    lrs_pretrain_net = []
    # # PRETRAINING PROTOTYPES PHASE
    for epoch in range(1, args.epochs_pretrain+1):
        # Original PIP-Net freezing logic for prototype pretraining.
        # Do not train the Stage 3 gate during pretraining.
        set_stage3_gate_enabled(net, False)
        set_stage3_matrix_enabled(net, False)
        set_stage3_stage4_distribution_loss_active(net, False)
        set_stage3_gate_trainable(net, False)
        set_stage3_matrix_trainable(net, False)
        set_stage3_additive_trainable(net, False)
        for param in params_to_train:
            param.requires_grad = True
        for param in net.module._add_on.parameters():
            param.requires_grad = True
        for param in net.module._classification.parameters():
            param.requires_grad = False
        for param in params_to_freeze:
            param.requires_grad = True # can be set to False when you want to freeze more layers
        for param in params_backbone:
            param.requires_grad = False #can be set to True when you want to train whole backbone (e.g. if dataset is very different from ImageNet)

        print("\nPretrain Epoch", epoch, "with batch size", trainloader_pretraining.batch_size, flush=True)
        
        # Pretrain prototypes
        train_info = train_pipnet(net, trainloader_pretraining, optimizer_net, optimizer_classifier, scheduler_net, None, criterion, epoch, args.epochs_pretrain, device, pretrain=True, finetune=False)
        lrs_pretrain_net+=train_info['lrs_net']
        plt.clf()
        plt.plot(lrs_pretrain_net)
        plt.savefig(os.path.join(args.log_dir,'lr_pretrain_net.png'))
        log.log_values('log_epoch_overview', epoch, "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", train_info['loss'])
    
    if args.state_dict_dir_net == '':
        net.eval()
        torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_pretrained'))
        net.train()
    if args.epochs_pretrain > 0:
        # Snapshot 2: Stage3 after prototype pretraining.
        visualize_stage3_snapshot(
            net,
            projectloader,
            classes,
            device,
            args,
            'stage3_snapshot_01_after_pretraining_topk',
            k=10
        )

        with torch.no_grad():
            if 'convnext' in args.net:
                topks = visualize_both_stages(net, projectloader, len(classes), device, 'visualised_pretrained_prototypes_topk', args, topk=True)
        
    # SECOND TRAINING PHASE
    # Enable gated evidence for the supervised/classification phase only.
    set_stage3_gate_enabled(net, requested_stage3_gating)
    set_stage3_matrix_enabled(net, requested_stage3_matrix)
    set_stage3_stage4_distribution_loss_active(net, requested_distribution_loss)

    # re-initialize optimizers and schedulers for second training phase
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)            
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader)*args.epochs, eta_min=args.lr_net/100.)
    # scheduler for the classification layer is with restarts, such that the model can re-active zeroed-out prototypes. Hence an intuitive choice. 
    if args.epochs<=30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=5, eta_min=0.001, T_mult=1)
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=10, eta_min=0.001, T_mult=1)
    for param in net.module.parameters():
        param.requires_grad = False
    for param in net.module._classification.parameters():
        param.requires_grad = True
    set_stage3_gate_trainable(net, False)
    set_stage3_matrix_trainable(net, False)
    set_stage3_additive_trainable(net, False)
    
    frozen = True
    lrs_net = []
    lrs_classifier = []
   
    for epoch in range(1, args.epochs + 1):                      
        epochs_to_finetune = 3 #during finetuning, only train classification layer and freeze rest. usually done for a few epochs (at least 1, more depends on size of dataset)
        if epoch <= epochs_to_finetune and (args.epochs_pretrain > 0 or args.state_dict_dir_net != ''):
            for param in net.module._add_on.parameters():
                param.requires_grad = False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            set_stage3_gate_trainable(net, False)
            set_stage3_matrix_trainable(net, False)
            set_stage3_additive_trainable(net, False)
            finetune = True
        
        else: 
            finetune=False          
            if frozen:
                # unfreeze backbone
                if epoch>(args.freeze_epochs):
                    for param in net.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_freeze:
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = True
                    set_stage3_gate_trainable(net, requested_stage3_gating)
                    set_stage3_matrix_trainable(net, requested_stage3_matrix)
                    frozen = False
                # freeze first layers of backbone, train rest
                else:
                    for param in params_to_freeze:
                        param.requires_grad = True #Can be set to False if you want to train fewer layers of backbone
                    for param in net.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = False
                    set_stage3_gate_trainable(net, requested_stage3_gating)
                    set_stage3_matrix_trainable(net, requested_stage3_matrix)
                    set_stage3_additive_trainable(net, requested_stage3_additive)
        
        print("\n Epoch", epoch, "frozen:", frozen, flush=True)            
        if (epoch==args.epochs or epoch%30==0) and args.epochs>1:
            # SET SMALL WEIGHTS TO ZERO
            with torch.no_grad():
                torch.set_printoptions(profile="full")
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 0.001, min=0.)) 
                print("Classifier weights: ", net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)], (net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)]).shape, flush=True)
                if args.bias:
                    print("Classifier bias: ", net.module._classification.bias, flush=True)
                torch.set_printoptions(profile="default")

        if requested_distribution_loss:
            # Version A: recompute class centroids once before each supervised epoch,
            # matching MCPNet's idea of refreshing class-specific distributions.
            was_training = net.training
            net.eval()
            centroid_loader = trainloader_normal if getattr(args, 'stage3_stage4_distribution_centroid_loader', 'train_normal') == 'train_normal' else projectloader
            with torch.no_grad():
                centroids = compute_class_centroids(
                    net,
                    centroid_loader,
                    len(classes),
                    device,
                    include_stage3=bool(getattr(args, 'stage3_stage4_distribution_include_stage3', True)),
                    include_stage4=bool(getattr(args, 'stage3_stage4_distribution_include_stage4', True)),
                    normalize_parts=bool(getattr(args, 'stage3_stage4_distribution_normalize_parts', True)),
                )
            net.module.stage3_stage4_distribution_centroids = centroids.detach()
            if was_training:
                net.train()
            print("Distribution centroids updated for epoch", epoch, "shape:", tuple(centroids.shape), flush=True)

        train_info = train_pipnet(net, trainloader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, args.epochs, device, pretrain=False, finetune=finetune)
        lrs_net+=train_info['lrs_net']
        lrs_classifier+=train_info['lrs_class']
        # Evaluate model
        eval_info = eval_pipnet(net, testloader, epoch, device, log)
        log.log_values('log_epoch_overview', epoch, eval_info['top1_accuracy'], eval_info['top5_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], train_info['train_accuracy'], train_info['loss'])
            
        with torch.no_grad():
            net.eval()
            torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained'))

            if epoch%30 == 0:
                net.eval()
                torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_%s'%str(epoch)))            
        
            # save learning rate in figure
            plt.clf()
            plt.plot(lrs_net)
            plt.savefig(os.path.join(args.log_dir,'lr_net.png'))
            plt.clf()
            plt.plot(lrs_classifier)
            plt.savefig(os.path.join(args.log_dir,'lr_class.png'))
                
    net.eval()
    torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_last'))

    # Snapshot 3: Stage3 after full final-branch PIP-Net training, before the
    visualize_stage3_snapshot(
        net,
        projectloader,
        classes,
        device,
        args,
        'stage3_snapshot_02_after_final_training_topk',
        k=10
    )

    if (not getattr(args, "use_stage3_additive_evidence", False)) and (not getattr(args, "use_stage3_matrix_shaping", False)) and hasattr(net.module, "classifier_penultimate") and net.module.classifier_penultimate is not None:
        train_stage3_classifier_only(
            net,
            trainloader_normal,
            device,
            args,
            num_epochs=5,
            lr=0.001
        )
        net.eval()
        torch.save(
            {
                'model_state_dict': net.state_dict(),
                'optimizer_net_state_dict': optimizer_net.state_dict(),
                'optimizer_classifier_state_dict': optimizer_classifier.state_dict()
            },
            os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_last_with_stage3')
        )

        # # Snapshot 4: Same stage3 prototype maps after the stage3 classifier-only
        # visualize_stage3_snapshot(
        #     net,
        #     projectloader,
        #     classes,
        #     device,
        #     args,
        #     'stage3_snapshot_03_after_stage3_classifier_topk',
        #     k=10
        # )

    topks = visualize_topk(
        net,
        projectloader,
        len(classes),
        device,
        'visualised_prototypes_topk',
        args,
        branch="final"
    )

    visualize_topk(
        net,
        projectloader,
        len(classes),
        device,
        'visualised_prototypes_topk',
        args,
        branch="stage3"
    )    
    # set weights of prototypes that are never really found in projection set to 0
    set_to_zero = []
    if topks:
        for prot in topks.keys():
            found = False
            for (i_id, score) in topks[prot]:
                if score > 0.1:
                    found = True
            if not found:
                torch.nn.init.zeros_(net.module._classification.weight[:,prot])
                set_to_zero.append(prot)
        print("Weights of prototypes", set_to_zero, "are set to zero because it is never detected with similarity>0.1 in the training set", flush=True)
        eval_info = eval_pipnet(net, testloader, "notused"+str(args.epochs), device, log)
        log.log_values('log_epoch_overview', "notused"+str(args.epochs), eval_info['top1_accuracy'], eval_info['top5_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], "n.a.", "n.a.")

    print("classifier weights: ", net.module._classification.weight, flush=True)
    print("Classifier weights nonzero: ", net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)], (net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)]).shape, flush=True)
    print("Classifier bias: ", net.module._classification.bias, flush=True)
    # Print weights and relevant prototypes per class
    for c in range(net.module._classification.weight.shape[0]):
        relevant_ps = []
        proto_weights = net.module._classification.weight[c,:]
        for p in range(net.module._classification.weight.shape[1]):
            if proto_weights[p]> 1e-3:
                relevant_ps.append((p, proto_weights[p].item()))
        if args.validation_size == 0.:
            print("Class", c, "(", list(testloader.dataset.class_to_idx.keys())[list(testloader.dataset.class_to_idx.values()).index(c)],"):","has", len(relevant_ps),"relevant prototypes: ", relevant_ps, flush=True)

    # Evaluate prototype purity        
    if args.dataset == 'CUB-200-2011':
        projectset_img0_path = projectloader.dataset.samples[0][0]
        project_path = os.path.split(os.path.split(projectset_img0_path)[0])[0].split("dataset")[0]
        parts_loc_path = os.path.join(project_path, "parts/part_locs.txt")
        parts_name_path = os.path.join(project_path, "parts/parts.txt")
        imgs_id_path = os.path.join(project_path, "images.txt")
        cubthreshold = 0.5 

        net.eval()
        print("\n\nEvaluating cub prototypes for training set", flush=True)        
        csvfile_topk = get_topk_cub(net, projectloader, 10, 'train_'+str(epoch), device, args)
        eval_prototypes_cub_parts_csv(csvfile_topk, parts_loc_path, parts_name_path, imgs_id_path, 'train_topk_'+str(epoch), args, log)
        
        csvfile_all = get_proto_patches_cub(net, projectloader, 'train_all_'+str(epoch), device, args, threshold=cubthreshold)
        eval_prototypes_cub_parts_csv(csvfile_all, parts_loc_path, parts_name_path, imgs_id_path, 'train_all_thres'+str(cubthreshold)+'_'+str(epoch), args, log)
        
        print("\n\nEvaluating cub prototypes for test set", flush=True)
        csvfile_topk = get_topk_cub(net, test_projectloader, 10, 'test_'+str(epoch), device, args)
        eval_prototypes_cub_parts_csv(csvfile_topk, parts_loc_path, parts_name_path, imgs_id_path, 'test_topk_'+str(epoch), args, log)
        cubthreshold = 0.5
        csvfile_all = get_proto_patches_cub(net, test_projectloader, 'test_'+str(epoch), device, args, threshold=cubthreshold)
        eval_prototypes_cub_parts_csv(csvfile_all, parts_loc_path, parts_name_path, imgs_id_path, 'test_all_thres'+str(cubthreshold)+'_'+str(epoch), args, log)
        
    # visualize predictions 
    visualize_both_stages(
    net,
    projectloader,
    len(classes),
    device,
    'visualised_prototypes',
    args
    )
    testset_img0_path = test_projectloader.dataset.samples[0][0]
    test_path = os.path.split(os.path.split(testset_img0_path)[0])[0]
    vis_pred(net, test_path, classes, device, args) 
    if args.extra_test_image_folder != '':
        if os.path.exists(args.extra_test_image_folder):   
            vis_pred_experiments(net, args.extra_test_image_folder, classes, device, args)


    # EVALUATE OOD DETECTION
    ood_datasets = ["CARS", "CUB-200-2011", "pets"]
    for percent in [95.]:
        print("\nOOD Evaluation for epoch", epoch,"with percent of", percent, flush=True)
        _, _, _, class_thresholds = get_thresholds(net, testloader, epoch, device, percent, log)
        print("Thresholds:", class_thresholds, flush=True)
        # Evaluate with in-distribution data
        id_fraction = eval_ood(net, testloader, epoch, device, class_thresholds)
        print("ID class threshold ID fraction (TPR) with percent",percent,":", id_fraction, flush=True)
        
        # Evaluate with out-of-distribution data
        for ood_dataset in ood_datasets:
            if ood_dataset != args.dataset:
                print("\n OOD dataset: ", ood_dataset,flush=True)
                ood_args = deepcopy(args)
                ood_args.dataset = ood_dataset
                _, _, _, _, _,ood_testloader, _, _ = get_dataloaders(ood_args, device)
                
                id_fraction = eval_ood(net, ood_testloader, epoch, device, class_thresholds)
                print(args.dataset, "- OOD", ood_dataset, "class threshold ID fraction (FPR) with percent",percent,":", id_fraction, flush=True)                

    print("Done!", flush=True)

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print_dir = os.path.join(args.log_dir,'out.txt')
    tqdm_dir = os.path.join(args.log_dir,'tqdm.txt')
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = open(print_dir, 'w')
    sys.stderr = open(tqdm_dir, 'w')
    run_pipnet(args)
    
    sys.stdout.close()
    sys.stderr.close()
