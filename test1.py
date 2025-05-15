import argparse
import os
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from datasets  import VITONDataset, VITONDataLoader
from network import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='./assets/')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='assets/test_pairs.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_dir', type=str, default='./results/')

    parser.add_argument('--display_freq', type=int, default=1)
    parser.add_argument('--dataroot', default='C:\Users\Home\Downloads\Virtual_try_on_skratch\Virtual_try_on_skratch', help='path to dataset root')

    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth')
    parser.add_argument('--alias_checkpoint', type=str, default='alias_final.pth')

    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    parser.add_argument('--grid_size', type=int, default=5)

    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')

    opt = parser.parse_args()
    return opt

# def show_tensor_image(tensor_image, title=None):
#     image = tensor_image.squeeze().permute(1, 2, 0).numpy()
#     plt.imshow(image)
#     if title:
#         plt.title(title)
#     plt.axis('off')
#     plt.show()

def test(opt, seg, gmm, alias):
    opt.dataroot = opt.dataset_dir
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss.cpu()

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_names = inputs['img_name'][0]
            # print(f"[DEBUG] inputs: {inputs}")
            c_names = inputs['c_name']['unpaired'][0]

            img_agnostic = inputs['img_agnostic'].cpu()
            parse_agnostic = inputs['parse_agnostic'].cpu()
            pose = inputs['pose'].cpu()
            c = inputs['cloth']['unpaired'].cpu()
            cm = inputs['cloth_mask']['unpaired'].cpu()

            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).cpu()), dim=1)

            parse_pred_down = seg(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]

            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).cpu()
            parse_old.scatter_(1, parse_pred, 1.0)

            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).cpu()
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]

            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

            output_dir = opt.save_dir  

            output_tensor = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

            output_img = output_tensor[0].cpu().detach()
            output_np = (output_img + 1) / 2.0  
            output_np = (output_np.numpy() * 255).astype(np.uint8)
            output_np = np.transpose(output_np, (1, 2, 0))  

            os.makedirs(output_dir, exist_ok=True)

            img_name = img_names[0] if isinstance(img_names, (list, tuple)) else "output"
            output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_tryon.png")
            Image.fromarray(output_np).save(output_path)
            print(f"Saved try-on image to {output_path}")

            unpaired_names = []
            for img_name, c_name in zip(img_names, c_names):
                unpaired_names.append(f"{img_name.split('_')[0]}_{c_name}")

            # Save full batch using original tensor
            # save_images(output_tensor, unpaired_names, output_dir)  

            # Visualize each image in the batch
            # for idx in range(output_tensor.size(0)):
            #     show_tensor_image(output_tensor[idx].cpu(), title=unpaired_names[idx])

            if (i + 1) % opt.display_freq == 0:
                print(f"step: {i + 1}")

def main():
    opt = get_opt()
    opt.datamode = opt.dataset_mode
    opt.data_list = opt.dataset_list
    opt.stage = 'GMM'
    opt.fine_width = opt.load_width
    opt.fine_height = opt.load_height
    opt.radius = 3
    print(opt)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

    seg.cpu().eval()
    gmm.cpu().eval()
    alias.cpu().eval()
    test(opt, seg, gmm, alias)


if __name__ == '__main__':
    main()