import argparse
import os
import warnings

import lpips
import numpy as np
import torch
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataloader import get_filelists, SEN12MSCR_Dataset
from loss.Charbonnier_Loss import L1_Charbonnier_loss
from loss.tv_loss import TVLoss
from models.meta.model import AttnCGAN_CR
from utils.tools import ssim

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='batch size used for training')
parser.add_argument('--inputs_dir', type=str, default='K:/dataset/ensemble/dsen2')
parser.add_argument('--inputs_dir2', type=str, default='K:/dataset/ensemble/clf')
parser.add_argument('--cloudy_dir', type=str, default='K:/dataset/selected_data_folder/s2_cloudy')
parser.add_argument('--targets_dir', type=str, default='K:/dataset/selected_data_folder/s2_cloudFree')
parser.add_argument('--sar_dir', type=str, default='K:/dataset/selected_data_folder/s1')
parser.add_argument('--data_list_filepath', type=str,
                    default='E:/Development Program/Pycharm Program/ECANet/csv/datasetfilelist.csv')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
parser.add_argument('--lr_step', type=int, default=2, help='lr decay rate')
parser.add_argument('--lr_start_epoch_decay', type=int, default=1, help='epoch to start lr decay')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--save_model_dir', type=str, default='./weights/meta_cbam.pth',
                    help='directory used to store trained networks')
parser.add_argument('--is_test', type=bool, default=False)
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--checkpoint', type=str, default="./checkpoint")
parser.add_argument('--frozen', type=bool, default=False)
parser.add_argument('--speed', type=bool, default=False)
parser.add_argument('--input_channels', type=int, default=13)
parser.add_argument('--use_sar', type=bool, default=True)
parser.add_argument('--use_rgb', type=bool, default=True)
parser.add_argument('--use_input2', type=bool, default=True)
parser.add_argument('--load_weights', type=bool, default=True)
parser.add_argument('--weights_path', type=str, default='checkpoint/checkpoint_visual.pth')
parser.add_argument('--weight_decay', type=float, default=0.0001)
opts = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
])

csv_filepath = opts.data_list_filepath
inputs_dir = opts.inputs_dir
inputs_dir2 = opts.inputs_dir2
targets_dir = opts.targets_dir
output_channels = 3
if opts.use_rgb:
    output_channels = 3
else:
    output_channels = 13
train_filelist, val_filelist, _ = get_filelists(csv_filepath)
train_dataset = SEN12MSCR_Dataset(train_filelist, inputs_dir, targets_dir, sar_dir=opts.sar_dir,
                                  inputs_dir2=inputs_dir2, use_attention=True, cloudy_dir=opts.cloudy_dir)
val_dataset = SEN12MSCR_Dataset(val_filelist, inputs_dir, targets_dir, sar_dir=opts.sar_dir, inputs_dir2=inputs_dir2,
                                use_attention=True, cloudy_dir=opts.cloudy_dir)

train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if opts.use_sar and opts.use_input2 is False:
    print('create unet_new inc=13')
    meta_learner = AttnCGAN_CR(2, 13, 3, 1).to(device)
elif opts.use_sar and opts.use_input2:
    print('create unet_new inc=26')
    meta_learner = AttnCGAN_CR(2, 13, 3, 3, bilinear=True).to(device)
else:
    raise NotImplemented

if opts.load_weights and opts.weights_path is not None:
    weights = torch.load(opts.weights_path)
    try:
        meta_learner.load_state_dict(weights, strict=False)
    except:
        pass

if len(opts.gpu_ids) > 1:
    print("Parallel training!")
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    meta_learner = nn.DataParallel(meta_learner)

optimizer = optim.Adam(meta_learner.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
criterion_L1 = L1_Charbonnier_loss().to(device)
criterion_L2 = nn.MSELoss().to(device)
criterion_TV = TVLoss(weight=1.0).to(device)
criterion_vgg = lpips.LPIPS(net='alex').to(device)
num_epochs = opts.epoch
log_step = opts.log_freq


def lr_lambda(ep):
    initial_lr = 1e-4
    final_lr = 1e-5
    lr_decay = final_lr / initial_lr
    return 1 - (1 - lr_decay) * (ep / (num_epochs - 1))


scheduler = LambdaLR(optimizer, lr_lambda)

print('Start Training!')

meta_learner.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    running_loss_rgb = 0.0
    running_loss_s2 = 0.0
    running_loss_TV = 0.0
    running_loss_vgg = 0.0
    running_loss_attn = 0.0
    running_loss_roi = 0.0
    running_loss_sar = 0.0
    running_ssim = 0.0
    running_psnr = 0.0
    original_ssim = 0.0
    original_ssim2 = 0.0
    original_vgg = 0.0
    original_vgg2 = 0.0
    for i, images in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs = images["input"].to(device)
        cloudy = images["cloudy"].to(device)
        targets = images["target"].to(device)
        mask = images["mask"].to(device)
        inputs2 = torch.zeros(inputs.shape)
        if opts.use_sar is not None and opts.use_input2 is None:
            sars = images["sar"].to(device)
            outputs = meta_learner(sars, inputs)
        elif opts.use_sar is not None and opts.use_input2 is not None:
            sars = images["sar"].to(device)
            inputs2 = images["input2"].to(device)
            outputs = meta_learner(inputs, inputs2, sars, cloudy)
        else:
            outputs = meta_learner(inputs)
        outputs, out_s2, out_sar, out_attn = outputs[0], outputs[1], outputs[2], outputs[3]
        if opts.use_rgb:
            targets_rgb = targets[:, 1:4, :, :]
        else:
            targets_rgb = targets
        attn_weights = mask.repeat(1, outputs.size(1), 1, 1)
        loss_RGB = criterion_L1(outputs, targets_rgb)
        loss_S2 = criterion_L1(out_s2, targets)
        loss_sar = criterion_L1(out_sar, targets)
        loss_ROI = criterion_L1(outputs * attn_weights, targets_rgb * attn_weights)
        loss_attn = criterion_L2(out_attn[:, 0, :, :], mask)
        loss_TV = criterion_TV(outputs)
        loss_vgg = criterion_vgg(outputs, targets_rgb)
        loss = loss_RGB * 100 + loss_S2 * 5 + loss_TV * 5 + loss_sar * 5 + loss_ROI * 100 + loss_vgg * 5 + loss_attn * 5
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss_rgb += loss_RGB.item() * 100
        running_loss_s2 += loss_S2.item() * 5
        running_loss_sar += loss_sar.item() * 5
        running_loss_TV += loss_TV.item() * 100  # Scale=100
        running_loss_roi += loss_ROI.item() * 100
        running_loss_attn += loss_attn.item() * 5
        running_loss_vgg += loss_vgg.item() * 5
        outputs_np = outputs.cpu().detach().numpy()
        targets_np = targets_rgb.cpu().detach().numpy()

        batch_ssim = ssim(outputs, targets_rgb)
        ori_ssim = 0.0
        ori_ssim2 = 0.0
        if opts.use_rgb:
            ori_ssim = ssim(inputs[:, 1:4, :, :], targets_rgb)
            ori_ssim2 = ssim(inputs2[:, 1:4, :, :], targets_rgb)
            ori_vgg = criterion_vgg(inputs[:, 1:4, :, :], targets_rgb).item() * 10
            ori_vgg2 = criterion_vgg(inputs2[:, 1:4, :, :], targets_rgb).item() * 10
        else:
            ori_ssim = ssim(inputs, targets_rgb)
            ori_ssim2 = ssim(inputs2, targets_rgb)
            ori_vgg = criterion_vgg(inputs, targets_rgb) * 10
            ori_vgg2 = criterion_vgg(inputs2, targets_rgb) * 10

        batch_psnr = np.mean([psnr(targets_np[b], outputs_np[b]) for b in range(outputs_np.shape[0])])
        running_ssim += batch_ssim
        running_psnr += batch_psnr
        original_ssim += ori_ssim
        original_ssim2 += ori_ssim2
        original_vgg += ori_vgg
        original_vgg2 += ori_vgg2
        if (i + 1) % log_step == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], [{i + 1}/{len(train_dataloader)}], "
                  f"Loss: {running_loss / log_step:.4f}, "
                  f"L_RGB: {running_loss_rgb / log_step:.4f}, "
                  f"L_ROI: {running_loss_roi / log_step:.4f}, "
                  f"L_LPIPS: {running_loss_vgg / log_step:.4f}, "
                  f"L_S2: {running_loss_s2 / log_step:.4f}, "
                  f"L_sar: {running_loss_sar / log_step:.4f}, "
                  f"L_attn: {running_loss_attn / log_step:.4f}, "
                  f"L_TV: {running_loss_TV / log_step:.4f}, "
                  f"SSIM: {running_ssim / log_step:.4f}, "
                  f"PSNR: {running_psnr / log_step:.4f}, "
                  f"m1: {original_ssim / log_step:.4f}, {original_vgg / log_step:.4f}, "
                  f"m2: {original_ssim2 / log_step:.4f}, {original_vgg2 / log_step:.4f}")
            running_loss = 0.0
            running_loss_vgg = 0.0
            running_loss_rgb = 0.0
            running_loss_s2 = 0.0
            running_loss_TV = 0.0
            running_loss_attn = 0.0
            running_loss_roi = 0.0
            running_loss_sar = 0.0
            running_ssim = 0.0
            running_psnr = 0.0
            original_ssim = 0.0
            original_ssim2 = 0.0
            original_vgg = 0.0
            original_vgg2 = 0.0
    scheduler.step()

    print('start val')
    running_loss = 0.0
    running_ssim = 0.0
    running_psnr = 0.0
    running_val_loss = 0.0
    original_ssim = 0.0
    original_ssim2 = 0.0
    original_vgg = 0.0
    original_vgg2 = 0.0
    meta_learner.eval()
    with torch.no_grad():
        val_loss, val_ssim, val_psnr = 0.0, 0.0, 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_ori_ssim = 0.0
        total_ori_ssim2 = 0.0
        total_original_vgg = 0.0
        total_original_vgg2 = 0.0
        for i, images in enumerate(val_dataloader):
            inputs = images["input"].to(device)
            targets = images["target"].to(device)
            cloudy = images["cloudy"].to(device)
            mask = images["mask"].to(device)
            inputs2 = torch.zeros(inputs.shape)
            if opts.use_sar is not None and opts.use_input2 is None:
                sars = images["sar"].to(device)
                outputs = meta_learner(sars, inputs)
            elif opts.use_sar is not None and opts.use_input2 is not None:
                sars = images["sar"].to(device)
                inputs2 = images["input2"].to(device)
                outputs = meta_learner(inputs, inputs2, sars, cloudy)
            else:
                outputs = meta_learner(inputs)
            outputs = outputs[0]
            if opts.use_rgb:
                targets_rgb = targets[:, 1:4, :, :]
            else:
                targets_rgb = targets

            loss = criterion_L1(outputs, targets_rgb)

            running_val_loss = loss.item()
            running_loss += running_val_loss
            val_loss += running_val_loss

            outputs_np = outputs.cpu().numpy()
            targets_np = targets_rgb.cpu().numpy()

            val_ssim = ssim(outputs, targets_rgb)
            ori_ssim = 0.0
            ori_ssim2 = 0.0
            if opts.use_rgb:
                ori_ssim = ssim(inputs[:, 1:4, :, :], targets_rgb)
                ori_ssim2 = ssim(inputs2[:, 1:4, :, :], targets_rgb)
                ori_vgg = criterion_vgg(inputs[:, 1:4, :, :], targets_rgb).item() * 10
                ori_vgg2 = criterion_vgg(inputs2[:, 1:4, :, :], targets_rgb).item() * 10
            else:
                ori_ssim = ssim(inputs, targets_rgb)
                ori_ssim2 = ssim(inputs2, targets_rgb)
                ori_vgg = criterion_vgg(inputs, targets_rgb) * 10
                ori_vgg2 = criterion_vgg(inputs2, targets_rgb) * 10
            val_psnr = np.mean([psnr(targets_np[b], outputs_np[b]) for b in range(outputs_np.shape[0])])

            total_psnr += val_psnr
            total_ssim += val_ssim
            total_ori_ssim += ori_ssim
            total_ori_ssim2 += ori_ssim2
            running_psnr += val_psnr
            running_ssim += val_ssim
            original_ssim += ori_ssim
            original_ssim2 += ori_ssim2
            original_vgg += ori_vgg
            original_vgg2 += ori_vgg2
            total_original_vgg += ori_vgg
            total_original_vgg2 += ori_vgg2
            if (i + 1) % log_step == 0:
                print(f"VAL: Step [{i + 1}/{len(val_dataloader)}], "
                      f"Loss: {running_loss / log_step:.4f}, "
                      f"SSIM: {running_ssim / log_step:.4f}, "
                      f"PSNR: {running_psnr / log_step:.4f}, "
                      f"m1: {original_ssim / log_step:.4f}, {original_vgg / log_step:.4f}, "
                      f"m2: {original_ssim2 / log_step:.4f}, {original_vgg2 / log_step:.4f}")
                running_loss = 0.0
                running_psnr = 0.0
                running_ssim = 0.0
                original_ssim = 0.0
                original_ssim2 = 0.0
                original_vgg = 0.0
                original_vgg2 = 0.0
        print(f"Validation Results - Epoch: {epoch + 1}, Loss: {val_loss / len(val_dataloader):.4f}, "
              f"SSIM: {total_ssim / len(val_dataloader):.4f}, "
              f"PSNR: {total_psnr / len(val_dataloader):.4f}, "
              f"m1: {original_ssim / len(val_dataloader):.4f}, {total_original_vgg / len(val_dataloader):.4f}, "
              f"m2: {original_ssim2 / len(val_dataloader):.4f}, {total_original_vgg2 / len(val_dataloader):.4f}")

    meta_learner.train()

    if epoch % opts.save_freq == 0:
        torch.save(meta_learner.state_dict(), os.path.join(opts.checkpoint, f'checkpoint_{epoch}.pth'))

torch.save(meta_learner.state_dict(), opts.save_model_dir)
