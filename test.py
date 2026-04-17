import os
import glob
import argparse
from typing import List, Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
import torchvision
from PIL import Image

import model
import psnr_ssim


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_images(folder: str) -> List[str]:
    exts = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def load_rgb_as_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return ten


def build_histogram_batch(v_channel: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    v_channel: [B, 1, H, W], range [0, 1]
    returns:  [B, 256]
    """
    device = v_channel.device
    hists = []
    for i in range(v_channel.size(0)):
        hist = torch.histc(v_channel[i].detach() * 255.0, bins=256, min=0, max=255)
        if normalize:
            hist = hist / (hist.sum() + 1e-6)
        hists.append(hist)
    return torch.stack(hists, dim=0).to(device)


def bpnet_forward(class_net: torch.nn.Module, hist: torch.Tensor) -> torch.Tensor:
    """
    Compatible with several possible histogram input shapes.
    This assumes model.light_class() has already been modified
    to accept a 256-bin histogram, as required by the paper-aligned step2/step3 code.
    """
    candidates = [
        hist,                              # [B, 256]
        hist.unsqueeze(1),                 # [B, 1, 256]
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
        'Unable to forward BP-Net with histogram input. '
        'Please check whether model.light_class() has been rewritten to accept a 256-bin histogram. '
        f'Last error: {last_err}'
    )


def scale_bp_output(raw_pred: torch.Tensor, rho_min: int, rho_max: int) -> torch.Tensor:
    if raw_pred.min().item() >= 0.0 and raw_pred.max().item() <= 1.0:
        pred = raw_pred * float(rho_max - rho_min) + float(rho_min)
    else:
        pred = raw_pred
    return pred.clamp(float(rho_min), float(rho_max))


def run_act_recursively(act_net: torch.nn.Module, in_v: torch.Tensor, pred_iters: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    in_v: [B, 1, H, W]
    pred_iters: [B] long tensor
    """
    enhanced_v_list = []
    map_list = []

    for b in range(in_v.size(0)):
        current_v = in_v[b:b + 1]
        current_map = None
        n_iter = int(pred_iters[b].item())

        for _ in range(n_iter):
            current_v, m = act_net(current_v)
            current_map = m

        enhanced_v_list.append(current_v)
        if current_map is not None:
            map_list.append(current_map)

    enhanced_v = torch.cat(enhanced_v_list, dim=0)
    map_tensor = torch.cat(map_list, dim=0) if len(map_list) > 0 else None
    return enhanced_v, map_tensor


@torch.no_grad()
def enhance_step1_only(act_net: torch.nn.Module, input_rgb: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Step-1 ACT-Net pretraining/testing: single enhancement pass.
    """
    in_hsv = kornia.color.rgb_to_hsv(input_rgb)
    h, s, in_v = torch.split(in_hsv, 1, dim=1)
    enhanced_v, _ = act_net(in_v)
    enhanced_rgb = torch.cat((h, s, enhanced_v), dim=1)
    enhanced_rgb = kornia.color.hsv_to_rgb(enhanced_rgb).clamp(0.0, 1.0)
    return enhanced_rgb, 1


@torch.no_grad()
def enhance_with_bpnet(act_net: torch.nn.Module, class_net: torch.nn.Module, input_rgb: torch.Tensor,
                       rho_min: int, rho_max: int) -> Tuple[torch.Tensor, int]:
    in_hsv = kornia.color.rgb_to_hsv(input_rgb)
    h, s, in_v = torch.split(in_hsv, 1, dim=1)

    hist = build_histogram_batch(in_v, normalize=True)
    raw_pred = bpnet_forward(class_net, hist)
    pred_iters_float = scale_bp_output(raw_pred, rho_min, rho_max)
    pred_iters = torch.round(pred_iters_float).long().clamp(rho_min, rho_max)

    enhanced_v, _ = run_act_recursively(act_net, in_v, pred_iters)
    enhanced_rgb = torch.cat((h, s, enhanced_v), dim=1)
    enhanced_rgb = kornia.color.hsv_to_rgb(enhanced_rgb).clamp(0.0, 1.0)
    return enhanced_rgb, int(pred_iters[0].item())


def evaluate_metrics(result_dir: str, gt_dir: str) -> Tuple[float, float]:
    test_list = list_images(result_dir)
    list_psnr = []
    list_ssim = []

    for result_path in test_list:
        image_name = os.path.basename(result_path)
        gt_path = os.path.join(gt_dir, image_name)
        if not os.path.exists(gt_path):
            continue

        img_result = cv2.imread(result_path)
        img_gt = cv2.imread(gt_path)
        if img_result is None or img_gt is None:
            continue

        list_psnr.append(psnr_ssim.psnr(img_result, img_gt))
        list_ssim.append(psnr_ssim.ssim(img_result, img_gt))

    mean_psnr = float(np.mean(list_psnr)) if list_psnr else 0.0
    mean_ssim = float(np.mean(list_ssim)) if list_ssim else 0.0
    return mean_psnr, mean_ssim


def test(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and config.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    ensure_dir(config.results_folder)

    act_net = model.attention_enhance_light().to(device)
    act_state = torch.load(config.pretrain_dir, map_location=device)
    act_net.load_state_dict(act_state)
    act_net.eval()

    class_net = None
    if config.mode == 'final':
        if not config.class_pretrain_dir:
            raise ValueError('mode=final requires --class_pretrain_dir')
        class_net = model.light_class().to(device)
        bp_state = torch.load(config.class_pretrain_dir, map_location=device)
        class_net.load_state_dict(bp_state)
        class_net.eval()

    test_list = list_images(config.test_images_path)
    if len(test_list) == 0:
        raise FileNotFoundError(f'No images found in {config.test_images_path}')

    print(f'Test mode: {config.mode}')
    print(f'Found {len(test_list)} images')

    for image_path in test_list:
        input_rgb = load_rgb_as_tensor(image_path, device)

        if config.mode == 'step1':
            enhanced_rgb, used_iters = enhance_step1_only(act_net, input_rgb)
        else:
            enhanced_rgb, used_iters = enhance_with_bpnet(
                act_net=act_net,
                class_net=class_net,
                input_rgb=input_rgb,
                rho_min=config.rho_min,
                rho_max=config.rho_max,
            )

        out_name = os.path.basename(image_path)
        save_path = os.path.join(config.results_folder, out_name)
        torchvision.utils.save_image(enhanced_rgb, save_path)

        out_hsv = kornia.color.rgb_to_hsv(enhanced_rgb)
        _, _, out_v = torch.split(out_hsv, 1, dim=1)
        print(f'{out_name} | used_iters={used_iters} | mean_V={out_v.mean().item():.6f}')

    if config.gt_path and os.path.isdir(config.gt_path):
        mean_psnr, mean_ssim = evaluate_metrics(config.results_folder, config.gt_path)
        print(f'PSNR: {mean_psnr:.4f}')
        print(f'SSIM: {mean_ssim:.4f}')
    else:
        print('GT folder not provided or not found, skipped PSNR/SSIM evaluation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='final', choices=['step1', 'final'],
                        help='step1: test ACT-Net only with one pass; final: test ACT-Net + BP-Net recursively')
    parser.add_argument('--test_images_path', type=str, default='./data/test_in/')
    parser.add_argument('--results_folder', type=str, default='./data/test_out/')
    parser.add_argument('--gt_path', type=str, default='')

    parser.add_argument('--pretrain_dir', type=str, default='./data/step_3/save_model/ACTNet_best.pth',
                        help='ACTNet checkpoint path')
    parser.add_argument('--class_pretrain_dir', type=str, default='./data/step_3/save_model/BPNet_best.pth',
                        help='BPNet checkpoint path, required when mode=final')

    parser.add_argument('--rho_min', type=int, default=1)
    parser.add_argument('--rho_max', type=int, default=10)
    parser.add_argument('--gpu_id', type=str, default='0')

    config = parser.parse_args()
    test(config)