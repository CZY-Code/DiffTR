import os
import json
import numpy as np
import torch
import argparse
import yaml
from omegaconf import OmegaConf

from Diff.Diff4Acc import GaussianDiffusion1D, Trainer1D 
from utils.ob_dataV2 import get_acc_data
from Diff.networks import ScalarUNet, ScalarUNetV2

dtype = torch.cuda.FloatTensor

def main_run_func(args, conf, folds):
    # read data
    data_sets = get_acc_data(data_path='./data')
    data_set = data_sets[folds]

    # model
    model_Unet = ScalarUNetV2(
        tensor_shape = conf.model.tensor_shape,
        rank = args.rank,
        h_dim = conf.model.h_dim,
        act = conf.model.act,
        dropout = conf.model.dropout,
        latent_dim = conf.model.latent_dim,
        x_emb_size = conf.model.x_emb_size,
        sigma_func = conf.model.sigma_func,
        pooling_method = conf.model.pooling_method,
        skip_connection = conf.model.skip_connection
    )
    print('Model params: ', sum(param.numel() for param in model_Unet.parameters())/1e6, 'M')
    conf.trainer.results_folder = f"outputs/completion/acc/rank{args.rank}/seed{args.seed}/fold{folds}"
    # trainer
    diffusion_model = GaussianDiffusion1D(model_Unet, **conf.diffusion)
    trainer = Trainer1D(diffusion_model, data_set['train'], data_set['test'], **conf.trainer)
    trainer.train()

    # with open(f'acc_result.txt', 'a+') as file:
    #     file.write(f'rank: {args.rank}, flod: {folds} '+ json.dumps(trainer.best_metric) + '\n')
    
    return trainer.best_metric

def main():
    parser = argparse.ArgumentParser(description='Tensor completion')
    parser.add_argument('--rank', type=int, default=3, choices=[3, 5, 8, 10])
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--dev', type=int, default=0, help='CUDA ID')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.dev))

    # read config
    conf_path = './configs/Acc_diff_conf.yaml'
    
    with open(conf_path) as f:
        conf = yaml.full_load(f)
    conf = OmegaConf.create(conf)

    # writer
    metrics = []
    for i in range(1):
        best_metric = main_run_func(args, conf, i)
        metrics.append(best_metric)
        print(f'Fold {i} finished!')

    writing_metrics = {}
    for key in metrics[0].keys():
        mean_metric = np.mean([m[key] for m in metrics])
        print(f'rank: {args.rank}, {key}: {mean_metric:.3f}')
        writing_metrics[key] = mean_metric

    with open(f'acc_result.txt', 'a+') as file:
        file.write(f'rank: {args.rank}, ' + json.dumps(writing_metrics) + '\n')


    from pathlib import Path
    summary = Path(f"outputs/completion/acc/rank{args.rank}/seed{args.seed}/summary.json")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(dict(rank=args.rank, seed=args.seed, folds=metrics, metrics=writing_metrics), indent=2)+"\n")


if __name__ == "__main__":
    main()
