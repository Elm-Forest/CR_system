import os.path
import random
import shutil
import warnings
from matplotlib.cm import jet
import lpips
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

from dataset.data_generator import DataGenerator
from models.UnCRtainTS.test import pred_uncrtain
from models.dsen2cr.dsen2cr_network import DSen2CR_model
from models.glf_cr.net_CR_RDN import RDN_residual_CR
from utils.common import *
from utils.execute_utils import build_data, get_attention_mask, load_model, get_normalized_data, get_rgb_preview, \
    get_preview
from utils.tools import ssim, save_tensor_to_geotiff

warnings.filterwarnings('ignore')


def predict_meta(image_name='default'):
    device = torch.device('cuda:0')
    weights_path = META_WEIGHTS
    input_path = os.path.join(DEFAULT_DSEN_OUTPUT_DIR, image_name + '.tif')
    input_path2 = os.path.join(DEFAULT_GLF_OUTPUT_DIR, image_name + '.tif')
    input_path3 = os.path.join(DEFAULT_UNCRTAIN_OUTPUT_DIR, image_name + '.tif')
    target_path = os.path.join(SOURCE_DIR, 'target', image_name + '.tif')
    cloudy_path = os.path.join(SOURCE_DIR, 's2', image_name + '.tif')
    sar_path = os.path.join(SOURCE_DIR, 's1', image_name + '.tif')
    images = build_data(input_path, target_path, cloudy_path, sar_path, input_path2, ALLPROCESS, input_path3)
    inputs = images["input"]
    inputs2 = images["input2"]
    targets = images["target"]
    cloudy = images['cloudy']
    sar = images['sar']
    meta_learner = load_model(weights_path, device=device)
    meta_learner.eval()
    out = meta_learner(inputs.to(device).unsqueeze(dim=0), inputs2.to(device).unsqueeze(dim=0),
                       sar.to(device).unsqueeze(dim=0), cloudy.to(device).unsqueeze(dim=0))
    outputs, out_s2, out_sar, out_attn = out[0], out[1], out[2], out[3]
    outputs_rgb = outputs.cpu().detach() * 10000
    out_attn = out_attn.squeeze(0).squeeze(0).cpu().detach()
    maskdir = os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + MASK_GT_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    plt.imshow(out_attn)
    plt.axis('off')
    plt.savefig(maskdir, dpi=600, bbox_inches='tight', pad_inches=0, interpolation='nearest', cmap='jet')
    outputs_rgb = get_normalized_data(outputs_rgb.squeeze(0).numpy(), 2)
    criterion_vgg = lpips.LPIPS(net='alex').to(device)
    ssim_value = ssim(torch.from_numpy(outputs_rgb).unsqueeze(0), targets[1:4, :, :].unsqueeze(0), window_size=5).item()
    pnsr_value = psnr(outputs_rgb, targets[1:4, :, :].detach().numpy())
    lpips_value = criterion_vgg(torch.from_numpy(outputs_rgb).to(device),
                                targets[1:4, :, :].to(device).unsqueeze(0)).item()
    loss_L1 = F.l1_loss(torch.from_numpy(outputs_rgb).unsqueeze(0).to(device),
                        targets[1:4, :, :].to(device).unsqueeze(0)).item()
    output_R_channel = outputs_rgb[2, :, :]
    output_G_channel = outputs_rgb[1, :, :]
    output_B_channel = outputs_rgb[0, :, :]
    output_rgb = get_rgb_preview(output_R_channel, output_G_channel, output_B_channel, brighten_limit=2000)
    imgdir = os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + PRED_POSTFIX + OUTPUT_IMAGES_POSTFIX)

    plt.imshow(output_rgb)
    plt.axis('off')
    plt.savefig(imgdir, dpi=600, bbox_inches='tight', pad_inches=0)

    # out_sar = out[-1].cpu().detach() * 10000
    # outputs_rgb = get_normalized_data(out_sar.squeeze(0).numpy(), 2)
    # output_R_channel = outputs_rgb[3, :, :]
    # output_G_channel = outputs_rgb[2, :, :]
    # output_B_channel = outputs_rgb[1, :, :]
    # output_rgb = get_rgb_preview(output_R_channel, output_G_channel, output_B_channel, brighten_limit=2000)
    # sardir = os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + '_sar_rgb' + OUTPUT_IMAGES_POSTFIX)
    # plt.imshow(output_rgb)
    # plt.axis('off')
    # plt.savefig(sardir, dpi=600, bbox_inches='tight', pad_inches=0)

    # out_sar = out[-1].cpu().detach() * 10000
    # outputs_rgb = get_normalized_data(out_sar[:, 1:4, :, :].squeeze(0).numpy(), 2)
    # output_R_channel = outputs_rgb[2, :, :]
    # output_G_channel = outputs_rgb[1, :, :]
    # output_B_channel = outputs_rgb[0, :, :]
    # output_rgb = get_rgb_preview(output_R_channel, output_G_channel, output_B_channel, brighten_limit=2000)
    # sardir = os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + '_s2_st1_rgb' + OUTPUT_IMAGES_POSTFIX)
    # plt.imshow(output_rgb)
    # plt.axis('off')
    # plt.savefig(sardir, dpi=600, bbox_inches='tight', pad_inches=0)
    return {
        'image_dir': imgdir,
        'mask_dir': maskdir,
        'ssim': round(ssim_value, 4),
        'psnr': round(pnsr_value, 4),
        'lpips': round(lpips_value, 4),
        'l1': round(loss_L1, 4)
    }


def predict_glf(image_name='default'):
    device = torch.device('cuda:0')
    weights_path = GLF_WEIGHTS
    target_path = os.path.join(SOURCE_DIR, 'target', image_name + '.tif')
    cloudy_path = os.path.join(SOURCE_DIR, 's2', image_name + '.tif')
    sar_path = os.path.join(SOURCE_DIR, 's1', image_name + '.tif')
    images = build_data(None, target_path, cloudy_path, sar_path, None, PREPROCESS)
    cloudy = images['cloudy']
    sar = images['sar']
    checkpoint = torch.load(weights_path)
    skip_keys = [
        "RDBs.0.convs.1.attn_mask", "RDBs.0.convs.3.attn_mask",
        "RDBs.1.convs.1.attn_mask", "RDBs.1.convs.3.attn_mask",
        "RDBs.2.convs.1.attn_mask", "RDBs.2.convs.3.attn_mask",
        "RDBs.3.convs.1.attn_mask", "RDBs.3.convs.3.attn_mask",
        "RDBs.4.convs.1.attn_mask", "RDBs.4.convs.3.attn_mask",
        "RDBs.5.convs.1.attn_mask", "RDBs.5.convs.3.attn_mask"
    ]
    CR_net = RDN_residual_CR(256).to(device)
    try:
        CR_net.load_state_dict(checkpoint['network'], strict=True)
    except Exception as e:
        # print('Some problems happened during loading Weights: ', e)
        try:
            CR_net.load_state_dict(checkpoint['network'], strict=False)
        except:
            checkpoint_state_dict = checkpoint['network']
            filtered_checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k not in skip_keys}
            CR_net.load_state_dict(filtered_checkpoint_state_dict, strict=False)
    print('Finished loading Pretraining/Checkpoints Weights')
    CR_net.eval()
    for _, param in CR_net.named_parameters():
        param.requires_grad = False
    pred_clf = CR_net(cloudy.unsqueeze(0).to(device), sar.unsqueeze(0).to(device))
    outputs = pred_clf
    imgdir = os.path.join(DEFAULT_GLF_OUTPUT_DIR, image_name + '.tif')
    save_tensor_to_geotiff(outputs[0, :, :, :].cpu().numpy(), imgdir)
    return {'image_dir': imgdir}


def predict_dsen(image_name='default'):
    weights_path = DSEN_WEIGHTS
    random_seed_general = 42
    random.seed(random_seed_general)
    np.random.seed(random_seed_general)
    tf.set_random_seed(random_seed_general)
    input_shape = ((13, 256, 256), (2, 256, 256))
    scale = 2000
    model, shape_n = DSen2CR_model(input_shape,
                                   batch_per_gpu=1,
                                   num_layers=16,
                                   feature_size=256,
                                   use_cloud_mask=True,
                                   include_sar_input=True)
    model.load_weights(weights_path)
    params = {'input_dim': input_shape,
              'batch_size': 1,
              'shuffle': False,
              'scale': scale,
              'include_target': True,
              'data_augmentation': False,
              'random_crop': False,
              'crop_size': 256,
              'clip_min': [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  #
              'clip_max': [[0, 0],
                           [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                           [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]],
              'input_data_folder': SOURCE_DIR,
              'use_cloud_mask': True,
              'max_val_sar': 2,
              'cloud_threshold': 0.2}
    predict_filelist = [['1', 's1', 'target', 's2', f'{image_name}.tif']]
    predict_generator = DataGenerator(predict_filelist, **params)
    data, y = next(iter(predict_generator))
    predicted = model.predict_on_batch(data)
    data_image = next(iter(predicted))
    data_image *= scale
    original_channels_output = data_image[:13, :, :]
    image_path = os.path.join(DEFAULT_DSEN_OUTPUT_DIR, image_name + '.tif')
    save_tensor_to_geotiff(original_channels_output, path_out=image_path)
    return {'image_dir': image_path}


def predict_uncrtain(image_name='default'):
    target_path = os.path.join(SOURCE_DIR, 'target', image_name + '.tif')
    cloudy_path = os.path.join(SOURCE_DIR, 's2', image_name + '.tif')
    sar_path = os.path.join(SOURCE_DIR, 's1', image_name + '.tif')
    images = build_data(None, target_path, cloudy_path, sar_path, None, PREPROCESS)
    cloudy = images['cloudy']
    targets = images["target"]
    sar = images['sar']
    outputs = pred_uncrtain(cloudy, sar, targets)
    imgdir = os.path.join(DEFAULT_UNCRTAIN_OUTPUT_DIR, image_name + '.tif')
    save_tensor_to_geotiff(outputs[0, :, :, :].cpu().detach().numpy(), imgdir)
    return {'image_dir': imgdir}


def preprocess_images(image_name):
    target_path = os.path.join(SOURCE_DIR, 'target', image_name + '.tif')
    cloudy_path = os.path.join(SOURCE_DIR, 's2', image_name + '.tif')
    sar_path = os.path.join(SOURCE_DIR, 's1', image_name + '.tif')
    images = build_data(None, target_path, cloudy_path, sar_path, None, PREPROCESS)

    targets = images["target"]
    cloudy_ori = images["cloudy_ori"]
    cloudy = images['cloudy']
    mask = get_attention_mask(cloudy_path)
    mask = mask['mask']

    targets_R_channel = targets[3, :, :]
    targets_G_channel = targets[2, :, :]
    targets_B_channel = targets[1, :, :]
    cloudy_R_channel_ori = cloudy_ori[3, :, :]
    cloudy_G_channel_ori = cloudy_ori[2, :, :]
    cloudy_B_channel_ori = cloudy_ori[1, :, :]
    cloudy_R_channel = cloudy[3, :, :]
    cloudy_G_channel = cloudy[2, :, :]
    cloudy_B_channel = cloudy[1, :, :]

    targets_rgb = get_rgb_preview(targets_R_channel, targets_G_channel, targets_B_channel, brighten_limit=2000)
    cloudy_ori_rgb = get_rgb_preview(cloudy_R_channel_ori, cloudy_G_channel_ori, cloudy_B_channel_ori,
                                     brighten_limit=2000)
    cloudy_rgb = get_rgb_preview(cloudy_R_channel, cloudy_G_channel, cloudy_B_channel, brighten_limit=2000)

    sar_rgb = get_preview(sar_path, False, [1, 2, 2], sar_composite=True)

    target_dir = os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + TARGET_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    cloudy_ori_dir = os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + CLOUDY_ORI_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    cloudy_dir = os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + CLOUDY_POSTFIX + OUTPUT_IMAGES_POSTFIX)

    sar_dir = os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + SAR_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    mask_dir = os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + MASK_POSTFIX + OUTPUT_IMAGES_POSTFIX)

    plt.imshow(targets_rgb)
    plt.axis('off')
    plt.savefig(target_dir, dpi=600, bbox_inches='tight', pad_inches=0)

    plt.imshow(cloudy_ori_rgb)
    plt.axis('off')
    plt.savefig(cloudy_ori_dir, dpi=600, bbox_inches='tight', pad_inches=0)

    plt.imshow(cloudy_rgb)
    plt.axis('off')
    plt.savefig(cloudy_dir, dpi=600, bbox_inches='tight', pad_inches=0)

    plt.imshow(sar_rgb)
    plt.axis('off')
    plt.savefig(sar_dir, dpi=600, bbox_inches='tight', pad_inches=0)


    heatmap = jet(mask.squeeze(0).numpy())[:, :, :3]  # 使用jet色彩图，丢弃alpha通道因为原始图像可能不支持透明度
    # 将RGB图像的像素值范围从0-255转换到0-1
    normalized_rgb = targets_rgb / 255.0
    # 将热图应用于原始图像
    highlighted_image = 0.45 * normalized_rgb + 0.55 * heatmap  # 调整这里的系数可以改变热图的透明度
    # 显示结果
    plt.figure(figsize=(10, 5))  # 如果你想更改图像显示大小
    plt.imshow(highlighted_image)
    plt.axis('off')
    plt.savefig(mask_dir, dpi=600, bbox_inches='tight', pad_inches=0)
    return {'target_dir': target_dir, 'cloudy_ori_dir': cloudy_ori_dir,
            'cloudy_dir': cloudy_dir, 'sar_dir': sar_dir, 'mask_dir': mask_dir}


def postprocess_images(image_name):
    device = torch.device('cuda:0')
    input_path = os.path.join(DEFAULT_DSEN_OUTPUT_DIR, image_name + '.tif')
    input_path2 = os.path.join(DEFAULT_GLF_OUTPUT_DIR, image_name + '.tif')
    input_path3 = os.path.join(DEFAULT_UNCRTAIN_OUTPUT_DIR, image_name + '.tif')
    target_path = os.path.join(SOURCE_DIR, 'target', image_name + '.tif')
    images = build_data(input_path=input_path, input_path2=input_path2, target_path=target_path,
                        process_type=POSTPROCESS, input_path3=input_path3)
    inputs = images["input"]
    inputs2 = images["input2"]
    inputs3 = images["input3"]
    targets = images["target"]
    criterion_vgg = lpips.LPIPS(net='alex').to(device)

    ssim1 = ssim(inputs[1:4, :, :].unsqueeze(0), targets[1:4, :, :].unsqueeze(0), window_size=5).item()
    psnr1 = psnr(inputs[1:4, :, :].detach().numpy(), targets[1:4, :, :].detach().numpy())
    loss1_L1 = F.l1_loss(inputs[1:4, :, :].unsqueeze(0).to(device), targets[1:4, :, :].to(device).unsqueeze(0)).item()
    lpips1 = criterion_vgg(inputs[1:4, :, :].unsqueeze(0).to(device), targets[1:4, :, :].to(device).unsqueeze(0)).item()

    ssim2 = ssim(inputs2[1:4, :, :].unsqueeze(0), targets[1:4, :, :].unsqueeze(0), window_size=5).item()
    psnr2 = psnr(inputs2[1:4, :, :].detach().numpy(), targets[1:4, :, :].detach().numpy())
    loss2_L1 = F.l1_loss(inputs2[1:4, :, :].unsqueeze(0).to(device), targets[1:4, :, :].to(device).unsqueeze(0)).item()
    lpips2 = criterion_vgg(inputs2[1:4, :, :].unsqueeze(0).to(device),
                           targets[1:4, :, :].to(device).unsqueeze(0)).item()

    ssim3 = ssim(inputs3[1:4, :, :].unsqueeze(0), targets[1:4, :, :].unsqueeze(0), window_size=5).item()
    psnr3 = psnr(inputs3[1:4, :, :].detach().numpy(), targets[1:4, :, :].detach().numpy())
    loss3_L1 = F.l1_loss(inputs3[1:4, :, :].unsqueeze(0).to(device), targets[1:4, :, :].to(device).unsqueeze(0)).item()
    lpips3 = criterion_vgg(inputs3[1:4, :, :].unsqueeze(0).to(device),
                           targets[1:4, :, :].to(device).unsqueeze(0)).item()

    inputs_R_channel = inputs[3, :, :]
    inputs_G_channel = inputs[2, :, :]
    inputs_B_channel = inputs[1, :, :]
    inputs_R_channel2 = inputs2[3, :, :]
    inputs_G_channel2 = inputs2[2, :, :]
    inputs_B_channel2 = inputs2[1, :, :]
    inputs_R_channel3 = inputs3[3, :, :]
    inputs_G_channel3 = inputs3[2, :, :]
    inputs_B_channel3 = inputs3[1, :, :]
    inputs_rgb = get_rgb_preview(inputs_R_channel, inputs_G_channel, inputs_B_channel, brighten_limit=2000)
    inputs_rgb2 = get_rgb_preview(inputs_R_channel2, inputs_G_channel2, inputs_B_channel2, brighten_limit=2000)
    inputs3_rgb = get_rgb_preview(inputs_R_channel3, inputs_G_channel3, inputs_B_channel3, brighten_limit=1500)
    input1_dir = os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + INPUT1_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    input2_dir = os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + INPUT2_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    input3_dir = os.path.join(DEFAULT_META_OUTPUT_DIR, image_name + INPUT3_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    plt.imshow(inputs3_rgb)
    plt.axis('off')
    plt.savefig(input3_dir, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.imshow(inputs_rgb)
    plt.axis('off')
    plt.savefig(input1_dir, dpi=600, bbox_inches='tight', pad_inches=0)

    plt.imshow(inputs_rgb2)
    plt.axis('off')
    plt.savefig(input2_dir, dpi=600, bbox_inches='tight', pad_inches=0)
    return {
        'input1_dir': input1_dir,
        'input2_dir': input2_dir,
        'input3_dir': input3_dir,
        'ssim1': round(ssim1, 4),
        'psnr1': round(psnr1, 4),
        'loss1_L1': round(loss1_L1, 4),
        'lpips1': round(lpips1, 4),
        'ssim2': round(ssim2, 4),
        'psnr2': round(psnr2, 4),
        'loss2_L1': round(loss2_L1, 4),
        'lpips2': round(lpips2, 4),
        'ssim3': round(ssim3, 4),
        'psnr3': round(psnr3, 4),
        'loss3_L1': round(loss3_L1, 4),
        'lpips3': round(lpips3, 4)
    }


def delete_specific_file(folder_path, name):
    for filename in os.listdir(folder_path):
        if filename == name:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f'File {file_path} has been deleted.')
            break


def copy_specific_file(begin_path, target_path, name):
    beginning_file = os.path.join(begin_path, name)
    destination_file = os.path.join(target_path, name)
    shutil.copy(beginning_file, destination_file)
    print(f'File {name} has been copied to {destination_file}')


def delete_outputs(image_name):
    delete_specific_file(DEFAULT_META_OUTPUT_DIR, image_name + PRED_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    delete_specific_file(DEFAULT_META_OUTPUT_DIR, image_name + INPUT1_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    delete_specific_file(DEFAULT_META_OUTPUT_DIR, image_name + INPUT2_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    delete_specific_file(DEFAULT_META_OUTPUT_DIR, image_name + INPUT3_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    delete_specific_file(DEFAULT_META_OUTPUT_DIR, image_name + TARGET_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    delete_specific_file(DEFAULT_META_OUTPUT_DIR, image_name + CLOUDY_ORI_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    delete_specific_file(DEFAULT_META_OUTPUT_DIR, image_name + CLOUDY_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    delete_specific_file(DEFAULT_META_OUTPUT_DIR, image_name + AVG_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    delete_specific_file(DEFAULT_META_OUTPUT_DIR, image_name + SAR_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    delete_specific_file(DEFAULT_META_OUTPUT_DIR, image_name + MASK_POSTFIX + OUTPUT_IMAGES_POSTFIX)
    delete_specific_file(DEFAULT_GLF_OUTPUT_DIR, image_name + '.tif')
    delete_specific_file(DEFAULT_DSEN_OUTPUT_DIR, image_name + '.tif')
    delete_specific_file(DEFAULT_UNCRTAIN_OUTPUT_DIR, image_name + '.tif')


def delete_source(image_name):
    delete_specific_file(os.path.join(SOURCE_DIR, 's1'), image_name + '.tif')
    delete_specific_file(os.path.join(SOURCE_DIR, 's2'), image_name + '.tif')
    delete_specific_file(os.path.join(SOURCE_DIR, 'target'), image_name + '.tif')


def copy_source(image_name):
    copy_specific_file(os.path.join(ORI_DIR, 's1'), os.path.join(SOURCE_DIR, 's1'), image_name + '.tif')
    copy_specific_file(os.path.join(ORI_DIR, 's2_cloudy'), os.path.join(SOURCE_DIR, 's2'), image_name + '.tif')
    copy_specific_file(os.path.join(ORI_DIR, 's2_cloudFree'), os.path.join(SOURCE_DIR, 'target'), image_name + '.tif')


if __name__ == '__main__':
    file_name = 'ROIs1158_spring_15_p392'
    # delete_outputs(file_name)
    copy_source(file_name)
    out1 = preprocess_images(file_name)
    out2 = predict_dsen(file_name)
    out3 = predict_glf(file_name)
    out4 = predict_uncrtain(file_name)
    out5 = postprocess_images(file_name)

    # out6 = predict_meta(file_name)
    # print(out1)
    # print(out2)
    # print(out3)
    # print(out4)
    print(out5)
    # delete_source(file_name)
