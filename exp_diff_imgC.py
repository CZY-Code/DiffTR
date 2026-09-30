import os
import json
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
import argparse
import yaml
import cv2

from utils.ob_dataV2 import *
from utils.datautil import *
from Diff.Diff4ImgC import GaussianDiffusion1D, Trainer1D
from Diff.networks import ScalarINR2D, ScalarINR3D, ScalarINR4D
dtype = torch.FloatTensor

def main():
    parser = argparse.ArgumentParser(description='Tensor completion')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--SR', type=float, default=0.25, help='Masking ratio')
    parser.add_argument('--type', type=str, default='MSI', help='datatype')
    parser.add_argument('--name', type=str, default='balloons', help='name')
    parser.add_argument('--dev', type=int, default=0, help='CUDA ID')
    parser.add_argument('--tube', action='store_true', help='tube')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.dev))
    # read config
    if args.tube == True:
        conf_path = f'./configs/ImgC_{args.type}_tube_diff_conf.yaml'
    else:
        conf_path = f'./configs/ImgC_{args.type}_diff_conf.yaml'

    with open(conf_path) as f:
        conf = yaml.full_load(f)
    conf = OmegaConf.create(conf)

    # read data
    # name, datatype = 'foreman', 'Video' #['foreman', 'carphone']
    # name, datatype = 'Peppers', 'RGB' #[Plane Peppers House Sailboat]
    # name, datatype = 'toys', 'MSI' #['beers', 'clay', 'flowers', 'balloons']

    if args.type == 'Video':
        showChannel = 0
        file_path = os.path.join('data/Videos', args.name+'.yuv')
        Video_gt = read_yuv_video(file_path, width=176, height=144, frame_count=100)
        H, W, C = Video_gt.shape
        ori_mask = 1.0 - generate_random_mask_3d(H, W, C, visible_ratio=args.SR) #[H,W,3] True if the value is missing
        X = torch.from_numpy(Video_gt).type(dtype)
        data_set = get_ob_data_3D(X, ori_mask)
        model_Unet = ScalarINR3D(out_dim=conf.diffusion.channels, **conf.model)
    
    elif args.type == 'RGB':
        showChannel = [0, 1, 2]
        file_path = os.path.join('data/misc', args.name+'.tiff') 
        image_gt = np.array(Image.open(file_path)).astype(np.float32) / 255.0
        H, W, C = image_gt.shape
        if args.tube == True:
            ori_mask = 1.0 - generate_random_mask(H, W, visible_ratio=args.SR) #[512, 512]
            X = torch.from_numpy(image_gt).type(dtype)
            data_set = get_ob_data_2D(X, ori_mask)
            model_Unet = ScalarINR2D(out_dim=conf.diffusion.channels, **conf.model)
        else:
            ori_mask = 1.0 - generate_random_mask_3d(H, W, C, visible_ratio=args.SR) #[512, 512]
            X = torch.from_numpy(image_gt).type(dtype)
            data_set = get_ob_data_3D(X, ori_mask)
            model_Unet = ScalarINR3D(out_dim=conf.diffusion.channels, **conf.model)

    elif args.type == 'MSI':
        showChannel = [15, 25, 30]
        file_path = os.path.join('data/MSIs', args.name)
        MSI_ori_gt = load_grayscale_images_from_directory(file_path)
        MSI_gt = np.zeros((256, 256, 31), dtype=MSI_ori_gt.dtype)
        for i in range(MSI_ori_gt.shape[-1]):
            MSI_gt[:, :, i] = cv2.resize(MSI_ori_gt[:, :, i], (256, 256), interpolation=cv2.INTER_LINEAR)
        H, W, C = MSI_gt.shape
        if args.tube == True:
            ori_mask = 1.0 - generate_random_mask(H, W, visible_ratio=args.SR)
            X = torch.from_numpy(MSI_gt).type(dtype)
            
            # masked_img = np.clip((X*(1-ori_mask.unsqueeze(-1))).numpy(), 0, 1)
            # ori_img = Image.fromarray((MSI_gt* 255).astype(np.uint8)[..., [30, 16, 5]])
            # ori_img.save('ori.png')
            # masked_img = Image.fromarray((masked_img*255).astype(np.uint8)[..., [30, 16, 5]])
            # masked_img.save('masked.png')
            # exit(0)

            data_set = get_ob_data_2D(X, ori_mask)
            model_Unet = ScalarINR2D(out_dim=conf.diffusion.channels, **conf.model)
        else:
            ori_mask = 1.0 - generate_random_mask_3d(H, W, C, visible_ratio=args.SR)
            X = torch.from_numpy(MSI_gt).type(dtype)
            data_set = get_ob_data_3D(X, ori_mask)
            model_Unet = ScalarINR3D(out_dim=conf.diffusion.channels, **conf.model)

    elif args.type == 'y4m':
        showChannel = 0
        file_path = os.path.join('./data/Videos', args.name+'_cif.y4m')
        Video_gt = read_y4m_video(file_path, frame_count=10)
        H, W, N, C = Video_gt.shape
        if args.tube == True:
            ori_mask = 1.0 - generate_random_mask_3d(H, W, N, visible_ratio=args.SR) #[H,W,10] True if the value is missing
            X = torch.from_numpy(Video_gt).type(dtype)
            data_set = get_ob_data_4D_tube(X, ori_mask)
            model_Unet = ScalarINR3D(out_dim=conf.diffusion.channels, **conf.model)
        else:
            ori_mask = 1.0 - generate_random_mask_4d(H, W, N, C, visible_ratio=args.SR) #[H,W,10] True if the value is missing
            X = torch.from_numpy(Video_gt).type(dtype)
            data_set = get_ob_data_4D(X, ori_mask)
            model_Unet = ScalarINR4D(out_dim=conf.diffusion.channels, **conf.model)
    
    else:
        raise NotImplementedError
    
    print('Model params: ', sum(param.numel() for param in model_Unet.parameters())/1e6, 'M')

    conf.trainer.results_folder = f"outputs/completion/{args.type}/{args.name}/tube{int(args.tube)}_sr{args.SR}/seed{args.seed}"
    diffusion_model = GaussianDiffusion1D(model_Unet, **conf.diffusion)
    trainer = Trainer1D(diffusion_model, data_set['train'], data_set['test'], 
                        gt_tensor=X, ori_mask=ori_mask, visible_ratio=args.SR, is_tube=args.tube,
                        datatype=args.type, showChannel=showChannel, dataname=args.name,
                        **conf.trainer)
    trainer.train()

    from pathlib import Path
    (Path(conf.trainer.results_folder)/"summary.json").write_text(json.dumps(dict(name=args.name, modality=args.type, tube=args.tube, sampling_rate=args.SR, seed=args.seed, metrics=trainer.best_metric), indent=2)+"\n")

    with open(f'imgC_result_log.txt', 'a+') as file:
        file.write(f'tube: {args.tube}, name: {args.name}, SR: {args.SR}, rank: {conf.model.rank} '+ json.dumps(trainer.best_metric) + '\n')

if __name__ == "__main__":
    main()
    