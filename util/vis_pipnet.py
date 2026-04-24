from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
import torch.utils.data
import os
from PIL import Image, ImageDraw as D
import torchvision.transforms as transforms
import torchvision
from util.func import get_patch_size
import random


def _unwrap_net(net):
    return net.module if hasattr(net, "module") else net


def _get_branch_outputs(net, xs, branch="final"):
    """Return prototype maps, pooled scores, logits, classifier weights, num prototypes."""
    model = _unwrap_net(net)

    if branch in ["stage3", "penultimate"]:
        # Computes stage3 only for visualization.
        net(xs, compute_penultimate=True)

        proto = model.proto_penultimate
        pooled = model.pooled_penultimate
        out = model.out_penultimate

        if proto is None or pooled is None or out is None:
            raise RuntimeError(
                "Stage3 outputs were not computed. Make sure PIPNet.forward supports "
                "compute_penultimate=True and ConvNextMultiStage supports return_stage='both'."
            )

        classification_weights = model.classifier_penultimate.weight
        num_prototypes = model.penultimate_num_prototypes
        return proto, pooled, out, classification_weights, num_prototypes

    # Original stage4 behavior.
    proto, pooled, out = net(xs, inference=True)
    classification_weights = model._classification.weight
    num_prototypes = model._num_prototypes
    return proto, pooled, out, classification_weights, num_prototypes


@torch.no_grad()
def visualize_topk(net, projectloader, num_classes, device, foldername, args: argparse.Namespace, k=10, branch="final"):
    print(f"Visualizing {branch} prototypes for topk...", flush=True)

    save_dir = os.path.join(args.log_dir, foldername, branch)
    os.makedirs(save_dir, exist_ok=True)

    patchsize, skip = get_patch_size(args)
    imgs = projectloader.dataset.imgs

    net.eval()

    topks = {}
    num_prototypes = None
    classification_weights = None

    img_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        mininterval=50.,
        desc=f'Collecting topk {branch}',
        ncols=0
    )

    for i, (xs, ys) in img_iter:
        xs, ys = xs.to(device), ys.to(device)

        proto, pooled, out, classification_weights, num_prototypes = _get_branch_outputs(
            net, xs, branch=branch
        )

        # Original visualization assumes projectloader batch size = 1.
        pooled = pooled.squeeze(0)

        for p in range(pooled.shape[0]):
            c_weight = torch.max(classification_weights[:, p])

            # Stage3 classifier is frozen/random unless you train it, so do not
            # filter stage3 prototypes by classifier weight.
            if branch in ["stage3", "penultimate"]:
                use_proto = True
            else:
                use_proto = c_weight > 1e-3

            if use_proto:
                score = pooled[p].item()

                if p not in topks:
                    topks[p] = []

                if len(topks[p]) < k:
                    topks[p].append((i, score))
                else:
                    topks[p] = sorted(topks[p], key=lambda tup: tup[1], reverse=True)
                    if topks[p][-1][1] < score:
                        topks[p][-1] = (i, score)
                    elif topks[p][-1][1] == score:
                        if random.choice([0, 1]) > 0:
                            topks[p][-1] = (i, score)

    alli = []
    prototypes_not_used = []

    for p in topks.keys():
        found = False
        for idx, score in topks[p]:
            alli.append(idx)
            if score > 0.1:
                found = True
        if not found:
            prototypes_not_used.append(p)

    print(
        len(prototypes_not_used),
        f"{branch} prototypes do not have any similarity score > 0.1. Will be ignored.",
        flush=True
    )

    saved = {p: 0 for p in range(num_prototypes)}
    tensors_per_prototype = {p: [] for p in range(num_prototypes)}

    img_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        mininterval=50.,
        desc=f'Visualizing topk {branch}',
        ncols=0
    )

    for i, (xs, ys) in img_iter:
        if i not in alli:
            continue

        xs, ys = xs.to(device), ys.to(device)

        for p in topks.keys():
            if p in prototypes_not_used:
                continue

            for idx, score in topks[p]:
                if idx != i:
                    continue

                proto, pooled, out, classification_weights, num_prototypes = _get_branch_outputs(
                    net, xs, branch=branch
                )

                # proto shape: [1, num_prototypes, H, W]
                proto_single = proto.squeeze(0)

                proto_map = proto_single[p]
                flat_idx = torch.argmax(proto_map)
                h_idx = flat_idx // proto_map.shape[1]
                w_idx = flat_idx % proto_map.shape[1]

                img_to_open = imgs[i]
                if isinstance(img_to_open, tuple) or isinstance(img_to_open, list):
                    img_to_open = img_to_open[0]

                image = transforms.Resize(size=(args.image_size, args.image_size))(
                    Image.open(img_to_open).convert("RGB")
                )
                img_tensor = transforms.ToTensor()(image).unsqueeze_(0)

                h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(
                    args.image_size,
                    proto_single.shape,
                    patchsize,
                    skip,
                    h_idx,
                    w_idx
                )

                img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                saved[p] += 1
                tensors_per_prototype[p].append(img_tensor_patch)

    all_tensors = []

    for p in range(num_prototypes):
        if saved[p] > 0:
            _, patch_h, patch_w = tensors_per_prototype[p][0].shape
            label = "S3 P" + str(p) if branch in ["stage3", "penultimate"] else "S4 P" + str(p)

            txtimage = Image.new("RGB", (patch_w, patch_h), (0, 0, 0))
            draw = D.Draw(txtimage)
            draw.text((patch_w // 2, patch_h // 2), label, anchor='mm', fill="white")
            txttensor = transforms.ToTensor()(txtimage)
            tensors_per_prototype[p].append(txttensor)

            try:
                grid = torchvision.utils.make_grid(tensors_per_prototype[p], nrow=k + 1, padding=1)
                torchvision.utils.save_image(grid, os.path.join(save_dir, "grid_topk_%s.png" % str(p)))

                if saved[p] >= k:
                    all_tensors += tensors_per_prototype[p]
            except Exception as e:
                print(f"Could not save {branch} prototype {p}: {e}", flush=True)

    if len(all_tensors) > 0:
        grid = torchvision.utils.make_grid(all_tensors, nrow=k + 1, padding=1)
        torchvision.utils.save_image(grid, os.path.join(save_dir, "grid_topk_all.png"))
    else:
        print(f"No {branch} prototype grids saved.", flush=True)

    return topks


def visualize(net, projectloader, num_classes, device, foldername, args: argparse.Namespace, branch="final"):
    print(f"Visualizing {branch} prototypes...", flush=True)

    save_dir = os.path.join(args.log_dir, foldername, branch)
    os.makedirs(save_dir, exist_ok=True)

    patchsize, skip = get_patch_size(args)
    imgs = projectloader.dataset.imgs

    if len(imgs) / num_classes < 10:
        skip_img = 10
    elif len(imgs) / num_classes < 50:
        skip_img = 5
    else:
        skip_img = 2

    print("Every", skip_img, "is skipped in order to speed up visualization", flush=True)

    net.eval()

    saved = None
    tensors_per_prototype = None
    seen_max = None
    num_prototypes = None

    img_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        mininterval=100.,
        desc=f'Visualizing {branch}',
        ncols=0
    )

    images_seen_before = 0

    for i, (xs, ys) in img_iter:
        if i % skip_img == 0:
            images_seen_before += xs.shape[0]
            continue

        xs, ys = xs.to(device), ys.to(device)

        proto, pooled, out, classification_weights, num_prototypes = _get_branch_outputs(
            net, xs, branch=branch
        )

        if saved is None:
            saved = {p: 0 for p in range(num_prototypes)}
            tensors_per_prototype = {p: [] for p in range(num_prototypes)}
            seen_max = {p: 0. for p in range(num_prototypes)}

        max_per_prototype, max_idx_per_prototype = torch.max(proto, dim=0)
        max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
        max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1)

        for p in range(num_prototypes):
            c_weight = torch.max(classification_weights[:, p])

            if branch in ["stage3", "penultimate"]:
                use_proto = True
            else:
                use_proto = c_weight > 0

            if not use_proto:
                continue

            h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
            w_idx = max_idx_per_prototype_w[p]
            idx_to_select = max_idx_per_prototype[p, h_idx, w_idx].item()
            found_max = max_per_prototype[p, h_idx, w_idx].item()

            if found_max > seen_max[p]:
                seen_max[p] = found_max

            if found_max > 0.5:
                img_to_open = imgs[images_seen_before + idx_to_select]
                imglabel = None

                if isinstance(img_to_open, tuple) or isinstance(img_to_open, list):
                    imglabel = img_to_open[1]
                    img_to_open = img_to_open[0]

                image = transforms.Resize(size=(args.image_size, args.image_size))(
                    Image.open(img_to_open).convert("RGB")
                )
                img_tensor = transforms.ToTensor()(image).unsqueeze_(0)

                h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(
                    args.image_size,
                    proto.shape,
                    patchsize,
                    skip,
                    h_idx,
                    w_idx
                )

                img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]

                saved[p] += 1
                tensors_per_prototype[p].append((img_tensor_patch, found_max))

                proto_dir = os.path.join(save_dir, "prototype_%s" % str(p))
                os.makedirs(proto_dir, exist_ok=True)

                draw = D.Draw(image)
                draw.rectangle(
                    [(w_coor_min, h_coor_min), (w_coor_max, h_coor_max)],
                    outline='yellow',
                    width=2
                )

                base = str(img_to_open).split('/')[-1].split('.jpg')[0]
                image.save(
                    os.path.join(
                        proto_dir,
                        "%s_p%s_%s_%s_%s_rect.png" % (
                            branch,
                            str(p),
                            str(imglabel),
                            str(round(found_max, 2)),
                            base
                        )
                    )
                )

        images_seen_before += len(ys)

    if saved is None:
        print(f"No {branch} prototypes processed.", flush=True)
        return

    for p in range(num_prototypes):
        if saved[p] > 0:
            try:
                sorted_by_second = sorted(tensors_per_prototype[p], key=lambda tup: tup[1], reverse=True)
                sorted_ps = [i[0] for i in sorted_by_second]
                grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                torchvision.utils.save_image(grid, os.path.join(save_dir, "grid_%s.png" % str(p)))
            except RuntimeError:
                pass


def visualize_stage3(net, projectloader, num_classes, device, foldername, args: argparse.Namespace):
    return visualize(net, projectloader, num_classes, device, foldername, args, branch="stage3")


def visualize_topk_stage3(net, projectloader, num_classes, device, foldername, args: argparse.Namespace, k=10):
    return visualize_topk(net, projectloader, num_classes, device, foldername, args, k=k, branch="stage3")


def visualize_both_stages(net, projectloader, num_classes, device, foldername, args: argparse.Namespace, topk=False, k=10):
    if topk:
        visualize_topk(net, projectloader, num_classes, device, foldername, args, k=k, branch="final")
        visualize_topk(net, projectloader, num_classes, device, foldername, args, k=k, branch="stage3")
    else:
        visualize(net, projectloader, num_classes, device, foldername, args, branch="final")
        visualize(net, projectloader, num_classes, device, foldername, args, branch="stage3")


def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip, h_idx, w_idx):
    # Supports both [B, C, H, W] and [C, H, W].
    if len(softmaxes_shape) == 4:
        latent_h = softmaxes_shape[2]
        latent_w = softmaxes_shape[3]
    else:
        latent_h = softmaxes_shape[1]
        latent_w = softmaxes_shape[2]

    if latent_h == 26 and latent_w == 26:
        h_coor_min = max(0, (h_idx - 1) * skip + 4)
        if h_idx < latent_h - 1:
            h_coor_max = h_coor_min + patchsize
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patchsize

        w_coor_min = max(0, (w_idx - 1) * skip + 4)
        if w_idx < latent_w - 1:
            w_coor_max = w_coor_min + patchsize
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patchsize
    else:
        h_coor_min = h_idx * skip
        h_coor_max = min(img_size, h_idx * skip + patchsize)
        w_coor_min = w_idx * skip
        w_coor_max = min(img_size, w_idx * skip + patchsize)

    if h_idx == latent_h - 1:
        h_coor_max = img_size
    if w_idx == latent_w - 1:
        w_coor_max = img_size
    if h_coor_max == img_size:
        h_coor_min = img_size - patchsize
    if w_coor_max == img_size:
        w_coor_min = img_size - patchsize

    return int(h_coor_min), int(h_coor_max), int(w_coor_min), int(w_coor_max)
