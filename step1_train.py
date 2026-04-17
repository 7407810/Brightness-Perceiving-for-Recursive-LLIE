import os
import glob
import argparse

import cv2
import numpy as np
from PIL import Image

import torch
import torch.optim
import torchvision
import kornia

import dataloader
import model
import loss_function
import psnr_ssim


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'Conv2d_cd':
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def enhance_once(light_enhance_net, in_v):
    """Paper-aligned Step 1: ACT-Net is manually trained with recursive times = 1."""
    enhanced_v, enhance_map = light_enhance_net(in_v)
    return enhanced_v, enhance_map


def evaluate(light_enhance_net, config):
    light_enhance_net.eval()
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']
    test_list = []
    for ext in image_extensions:
        test_list.extend(glob.glob(os.path.join(config.val_images_path, ext)))

    with torch.no_grad():
        for image in test_list:
            data_lowlight = Image.open(image).convert('RGB')
            data_lowlight = np.asarray(data_lowlight, dtype=np.float32) / 255.0
            data_lowlight = torch.from_numpy(data_lowlight).float().cuda()
            data_lowlight = data_lowlight.permute(2, 0, 1).unsqueeze(0)

            in_hsv = kornia.color.rgb_to_hsv(data_lowlight)
            h, s, in_v = torch.split(in_hsv, 1, dim=1)

            # Step 1 validation is also kept single-pass to match the paper's ACT-Net pretraining stage.
            enhanced_v, _ = enhance_once(light_enhance_net, in_v)

            enhance_img = torch.cat((h, s, enhanced_v), dim=1)
            enhance_img = kornia.color.hsv_to_rgb(enhance_img)

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
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    light_enhance_net = model.attention_enhance_light().cuda()
    light_enhance_net.apply(weights_init)

    if config.load_pretrain:
        state_dict = torch.load(config.pretrain_dir, map_location='cpu')
        light_enhance_net.load_state_dict(state_dict)

    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    l_color = loss_function.L_color()
    l_exp = loss_function.L_exp(16, config.exposure_target)
    l_tv = loss_function.L_TV()

    optimizer = torch.optim.Adam(
        light_enhance_net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    max_psnr = -1.0

    for epoch in range(config.num_epochs):
        light_enhance_net.train()

        epoch_total = 0.0
        epoch_tv = 0.0
        epoch_exp = 0.0
        epoch_col = 0.0
        num_batches = 0

        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.cuda(non_blocking=True)
            in_hsv = kornia.color.rgb_to_hsv(img_lowlight)
            h, s, in_v = torch.split(in_hsv, 1, dim=1)

            # Paper-aligned Step 1: one enhancement step only.
            enhanced_v, enhance_map = enhance_once(light_enhance_net, in_v)

            enhance_img = torch.cat((h, s, enhanced_v), dim=1)
            enhance_img = kornia.color.hsv_to_rgb(enhance_img)

            loss_tv = config.lambda_tv * l_tv(enhance_map)
            loss_col = config.lambda_col * torch.mean(l_color(enhance_img))
            loss_exp = config.lambda_exp * torch.mean(l_exp(enhance_img))
            loss = loss_tv + loss_exp + loss_col

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(light_enhance_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            epoch_total += loss.item()
            epoch_tv += loss_tv.item()
            epoch_exp += loss_exp.item()
            epoch_col += loss_col.item()
            num_batches += 1

            if ((iteration + 1) % config.display_iter) == 0:
                print(
                    f"Epoch [{epoch + 1}/{config.num_epochs}] "
                    f"Iter [{iteration + 1}/{len(train_loader)}] "
                    f"loss={loss.item():.6f} "
                    f"tv={loss_tv.item():.6f} "
                    f"exp={loss_exp.item():.6f} "
                    f"col={loss_col.item():.6f}"
                )

        if num_batches > 0:
            print(
                f"Epoch [{epoch + 1}/{config.num_epochs}] summary: "
                f"loss={epoch_total / num_batches:.6f}, "
                f"tv={epoch_tv / num_batches:.6f}, "
                f"exp={epoch_exp / num_batches:.6f}, "
                f"col={epoch_col / num_batches:.6f}"
            )

        now_psnr, now_ssim = evaluate(light_enhance_net, config)
        if now_psnr > max_psnr:
            torch.save(light_enhance_net.state_dict(), os.path.join(config.snapshots_folder, 'ACTNet_best.pth'))
            max_psnr = now_psnr

        if ((epoch + 1) % config.snapshot_iter) == 0:
            torch.save(light_enhance_net.state_dict(), os.path.join(config.snapshots_folder, f'ACTNet_epoch_{epoch + 1}.pth'))

        print(f"Validation: now_psnr={now_psnr:.4f}, now_ssim={now_ssim:.4f}, max_psnr={max_psnr:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=str, default='0')

    # paths
    parser.add_argument('--lowlight_images_path', type=str, default='./data/step_1/train/')
    parser.add_argument('--val_images_path', type=str, default='./data/step_1/val/')
    parser.add_argument('--val_gt_path', type=str, default='./data/step_1/val_gt/')
    parser.add_argument('--results_folder', type=str, default='./data/step_1/results/')
    parser.add_argument('--snapshots_folder', type=str, default='./data/step_1/save_model/')

    # optimization
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)

    # paper-aligned loss settings
    parser.add_argument('--exposure_target', type=float, default=0.6)
    parser.add_argument('--lambda_exp', type=float, default=1.0)
    parser.add_argument('--lambda_col', type=float, default=0.5)
    parser.add_argument('--lambda_tv', type=float, default=200.0)

    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default='')

    config = parser.parse_args()

    os.makedirs(config.snapshots_folder, exist_ok=True)
    os.makedirs(config.results_folder, exist_ok=True)

    train(config)