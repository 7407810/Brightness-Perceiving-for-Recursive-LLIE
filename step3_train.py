import argparse
import glob
import os
from typing import List, Tuple

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

import dataloader
import loss_function
import model
import psnr_ssim


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight') and m.weight is not None:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_histogram_batch(v_channel: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Args:
        v_channel: [B, 1, H, W], range [0, 1]
    Returns:
        hist: [B, 256]
    """
    device = v_channel.device
    hists = []
    for i in range(v_channel.size(0)):
        hist = torch.histc(v_channel[i].detach() * 255.0, bins=256, min=0, max=255)
        if normalize:
            hist = hist / (hist.sum() + 1e-6)
        hists.append(hist)
    return torch.stack(hists, dim=0).to(device)


def bpnet_forward(class_net: nn.Module, hist: torch.Tensor) -> torch.Tensor:
    """
    Try several common tensor shapes for BP-Net input.
    This still assumes BP-Net has been changed to the paper-style histogram input.
    """
    candidates = [
        hist,                          # [B, 256]
        hist.unsqueeze(1),             # [B, 1, 256]
        hist.unsqueeze(-1).unsqueeze(-1),  # [B, 256, 1, 1]
    ]

    last_err = None
    for x in candidates:
        try:
            out = class_net(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.view(out.size(0), -1)
            return out[:, 0]
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Unable to forward BP-Net with histogram input. "
        f"Please check whether model.light_class() has been rewritten to accept a 256-bin histogram. "
        f"Last error: {last_err}"
    )


def scale_bp_output(raw_pred: torch.Tensor, rho_min: int, rho_max: int) -> torch.Tensor:
    """
    Paper logic: BP(x) in [rho_min, rho_max].
    If raw_pred is in [0, 1], map it into [rho_min, rho_max].
    Otherwise clamp it directly.
    """
    if raw_pred.min().item() >= 0.0 and raw_pred.max().item() <= 1.0:
        pred = raw_pred * float(rho_max - rho_min) + float(rho_min)
    else:
        pred = raw_pred
    return pred.clamp(float(rho_min), float(rho_max))


@torch.no_grad()
def estimate_pseudo_label(
    act_net: nn.Module,
    in_v_single: torch.Tensor,
    target_brightness: float,
    rho_max: int,
    level4_min_brightness: float,
) -> int:
    """
    Step-2/Step-3 pseudo-label rule aligned with the paper:
    - if input belongs to Level-4 ([0.45, 0.6]), Label_p = 1
    - else recursively apply ACT-Net until brightness reaches ~0.6, capped by rho_max
    """
    original_mean = float(in_v_single.mean().item())
    if original_mean >= level4_min_brightness:
        return 1

    current = in_v_single
    iter_num = 0
    while float(current.mean().item()) < target_brightness and iter_num < rho_max:
        current, _ = act_net(current)
        iter_num += 1

    return max(1, min(iter_num, rho_max))


def run_act_recursively(
    act_net: nn.Module,
    in_v: torch.Tensor,
    pred_iters: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        in_v: [B, 1, H, W]
        pred_iters: [B], integer tensor in [rho_min, rho_max]
    Returns:
        enhanced_v: [B, 1, H, W]
        map_tensor: concatenated mapping features from all executed iterations, [B, C_total, H, W]
    """
    enhanced_v_list: List[torch.Tensor] = []
    map_list: List[torch.Tensor] = []

    for b in range(in_v.size(0)):
        current_v = in_v[b:b + 1]
        current_maps = []
        n_iter = int(pred_iters[b].item())

        for _ in range(n_iter):
            current_v, current_map = act_net(current_v)
            current_maps.append(current_map)

        enhanced_v_list.append(current_v)
        map_list.append(torch.cat(current_maps, dim=1))

    enhanced_v = torch.cat(enhanced_v_list, dim=0)
    map_tensor = torch.cat(map_list, dim=0)
    return enhanced_v, map_tensor


@torch.no_grad()
def recursive_enhance_with_bpnet(
    act_net: nn.Module,
    class_net: nn.Module,
    input_rgb: torch.Tensor,
    rho_min: int,
    rho_max: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    in_hsv = kornia.color.rgb_to_hsv(input_rgb)
    h, s, in_v = torch.split(in_hsv, 1, dim=1)

    hist = build_histogram_batch(in_v, normalize=True)
    raw_pred = bpnet_forward(class_net, hist)
    pred_iters_float = scale_bp_output(raw_pred, rho_min, rho_max)
    pred_iters = torch.round(pred_iters_float).long().clamp(rho_min, rho_max)

    enhanced_v, _ = run_act_recursively(act_net, in_v, pred_iters)
    enhance_img = torch.cat((h, s, enhanced_v), dim=1)
    enhance_img = kornia.color.hsv_to_rgb(enhance_img).clamp(0.0, 1.0)
    return enhance_img, pred_iters


@torch.no_grad()
def validate(config, act_net, class_net, device):
    act_net.eval()
    class_net.eval()

    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']
    test_list = []
    for ext in image_extensions:
        test_list.extend(glob.glob(os.path.join(config.val_images_path, ext)))

    for image in test_list:
        data_lowlight = Image.open(image).convert('RGB')
        data_lowlight = np.asarray(data_lowlight, dtype=np.float32) / 255.0
        data_lowlight = torch.from_numpy(data_lowlight).permute(2, 0, 1).unsqueeze(0).to(device)

        enhance_img, pred_iters = recursive_enhance_with_bpnet(
            act_net=act_net,
            class_net=class_net,
            input_rgb=data_lowlight,
            rho_min=config.rho_min,
            rho_max=config.rho_max,
        )

        image_name = os.path.basename(image)
        torchvision.utils.save_image(enhance_img, os.path.join(config.results_folder, image_name))

    list_psnr = []
    list_ssim = []
    for image in test_list:
        image_name = os.path.basename(image)
        result_path = os.path.join(config.results_folder, image_name)
        gt_path = os.path.join(config.val_gt_path, image_name)

        if os.path.exists(result_path) and os.path.exists(gt_path):
            img_result = cv2.imread(result_path)
            img_gt = cv2.imread(gt_path)
            if img_result is not None and img_gt is not None:
                list_psnr.append(psnr_ssim.psnr(img_result, img_gt))
                list_ssim.append(psnr_ssim.ssim(img_result, img_gt))

    now_psnr = float(np.mean(list_psnr)) if list_psnr else 0.0
    now_ssim = float(np.mean(list_ssim)) if list_ssim else 0.0
    return now_psnr, now_ssim


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and config.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    act_net = model.attention_enhance_light().to(device)
    class_net = model.light_class().to(device)

    # Step-3 starts from pre-trained ACT-Net and BP-Net.
    if config.load_pretrain:
        act_state = torch.load(config.pretrain_dir, map_location=device)
        bp_state = torch.load(config.class_pretrain_dir, map_location=device)
        act_net.load_state_dict(act_state)
        class_net.load_state_dict(bp_state)
    else:
        raise ValueError('Step-3 joint training requires pre-trained ACT-Net and BP-Net checkpoints.')

    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Paper losses: Lexp + Lcol + Ltvm + Lp
    L_color = loss_function.L_color()
    L_exp = loss_function.L_exp(16, config.target_brightness)
    L_TV = loss_function.L_TV()
    L1 = nn.L1Loss(reduction='mean')

    optimizer_0 = torch.optim.Adam(act_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer_1 = torch.optim.Adam(class_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    max_psnr = -1e9

    for epoch in range(config.num_epochs):
        # Step-3 is joint fine-tuning, so both networks should be in train mode.
        act_net.train()
        class_net.train()

        epoch_loss = 0.0

        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.to(device, non_blocking=True)

            in_hsv = kornia.color.rgb_to_hsv(img_lowlight)
            h, s, in_v = torch.split(in_hsv, 1, dim=1)
            hist = build_histogram_batch(in_v, normalize=True)

            # BP-Net predicts the recursive factor from brightness histogram.
            raw_pred = bpnet_forward(class_net, hist)
            pred_iters_float = scale_bp_output(raw_pred, config.rho_min, config.rho_max)
            pred_iters = torch.round(pred_iters_float).long().clamp(config.rho_min, config.rho_max)

            # ACT-Net is unrolled according to the predicted recursive factor.
            enhanced_v, map_tensor = run_act_recursively(act_net, in_v, pred_iters)
            enhance_img = torch.cat((h, s, enhanced_v), dim=1)
            enhance_img = kornia.color.hsv_to_rgb(enhance_img).clamp(0.0, 1.0)

            # Pseudo label for BP-Net. We generate it with the current ACT-Net under no_grad,
            # so the label remains a supervision signal rather than a differentiable target.
            with torch.no_grad():
                pseudo_labels = []
                for b in range(in_v.size(0)):
                    label = estimate_pseudo_label(
                        act_net=act_net,
                        in_v_single=in_v[b:b + 1],
                        target_brightness=config.target_brightness,
                        rho_max=config.rho_max,
                        level4_min_brightness=config.level4_min_brightness,
                    )
                    pseudo_labels.append(label)
                pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.float32, device=device)

            # Paper total loss:
            # Ltotal = lambda_exp * Lexp + lambda_col * Lcol + lambda_tvm * Ltvm + lambda_p * Lp
            loss_exp = torch.mean(L_exp(enhance_img))
            loss_col = torch.mean(L_color(enhance_img))
            loss_tvm = L_TV(map_tensor)
            loss_p = L1(pred_iters_float, pseudo_labels)

            loss = (
                config.lambda_exp * loss_exp +
                config.lambda_col * loss_col +
                config.lambda_tvm * loss_tvm +
                config.lambda_p * loss_p
            )

            optimizer_0.zero_grad()
            optimizer_1.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(act_net.parameters(), config.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(class_net.parameters(), config.grad_clip_norm)
            optimizer_0.step()
            optimizer_1.step()

            epoch_loss += loss.item()

            if (iteration + 1) % config.display_iter == 0:
                print(
                    f"Epoch [{epoch + 1}/{config.num_epochs}] Iter [{iteration + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f} | "
                    f"Lexp: {loss_exp.item():.6f} | "
                    f"Lcol: {loss_col.item():.6f} | "
                    f"Ltvm: {loss_tvm.item():.6f} | "
                    f"Lp: {loss_p.item():.6f} | "
                    f"Pseudo: {pseudo_labels.detach().cpu().tolist()} | "
                    f"Pred: {pred_iters_float.detach().cpu().tolist()} | "
                    f"Used iters: {pred_iters.detach().cpu().tolist()}"
                )

        avg_epoch_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] average total loss: {avg_epoch_loss:.6f}")

        if (epoch + 1) % config.snapshot_iter == 0:
            torch.save(act_net.state_dict(), os.path.join(config.snapshots_folder, f'ACTNet_epoch_{epoch + 1}.pth'))
            torch.save(class_net.state_dict(), os.path.join(config.snapshots_folder, f'BPNet_epoch_{epoch + 1}.pth'))

        if config.do_validation:
            now_psnr, now_ssim = validate(config, act_net, class_net, device)
            if now_psnr > max_psnr:
                torch.save(act_net.state_dict(), os.path.join(config.snapshots_folder, 'ACTNet_best.pth'))
                torch.save(class_net.state_dict(), os.path.join(config.snapshots_folder, 'BPNet_best.pth'))
                max_psnr = now_psnr
            print(f"Validation PSNR: {now_psnr:.4f}, SSIM: {now_ssim:.4f}, Best PSNR: {max_psnr:.4f}")
        else:
            torch.save(act_net.state_dict(), os.path.join(config.snapshots_folder, 'ACTNet_latest.pth'))
            torch.save(class_net.state_dict(), os.path.join(config.snapshots_folder, 'BPNet_latest.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--lowlight_images_path', type=str, default='./data/step_3/train/')
    parser.add_argument('--val_images_path', type=str, default='./data/step_3/val/')
    parser.add_argument('--val_gt_path', type=str, default='./data/step_3/val_gt/')
    parser.add_argument('--results_folder', type=str, default='./data/step_3/results/')
    parser.add_argument('--snapshots_folder', type=str, default='./data/step_3/save_model/')
    parser.add_argument('--pretrain_dir', type=str, default='./data/step_1/save_model/ACTNet_best.pth')
    parser.add_argument('--class_pretrain_dir', type=str, default='./data/step_2/save_model/BPNet_best.pth')

    # Training setup, aligned with the paper as much as possible
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=50)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--do_validation', type=bool, default=True)

    # Paper-specific settings
    parser.add_argument('--target_brightness', type=float, default=0.6)
    parser.add_argument('--level4_min_brightness', type=float, default=0.45)
    parser.add_argument('--rho_min', type=int, default=1)
    parser.add_argument('--rho_max', type=int, default=10)

    # Total loss weights in the paper
    parser.add_argument('--lambda_exp', type=float, default=1.0)
    parser.add_argument('--lambda_col', type=float, default=0.5)
    parser.add_argument('--lambda_tvm', type=float, default=200.0)
    parser.add_argument('--lambda_p', type=float, default=0.001)

    config = parser.parse_args()

    ensure_dir(config.snapshots_folder)
    ensure_dir(config.results_folder)

    train(config)
