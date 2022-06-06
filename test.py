import os
import argparse
from omegaconf import OmegaConf
from importlib import import_module
import pytorch_lightning as pl


def load_conf(args):
    base_cfg = OmegaConf.load("conf/path.yaml")
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    root = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.test.use_exp)
    assert os.path.exists(root), "wrong experiment path"
    root = os.path.join(root, f"{args.task}")
    os.makedirs(root, exist_ok=True)

    # HACK manually setting those properties
    cfg.data.split = args.split
    cfg.data.batch_size = 1
    cfg.general.task = args.task
    cfg.general.root = root
    cfg.cluster.prepare_epochs = -1

    return cfg

def init_data(cfg):
    DATA_MODULE = import_module(cfg.data.module)
    dataloader = getattr(DATA_MODULE, cfg.data.loader)

    if cfg.general.task == "train":
        print("=> loading the train and val datasets...")
    else:
        print("=> loading the {} dataset...".format(cfg.data.split))
        
    dataset, dataloader = dataloader(cfg)
    print("=> loading dataset completed")

    return dataset, dataloader

# def init_model(cfg):
#     MODEL = getattr(import_module(cfg.model.module), cfg.model.classname)
#     model = MODEL(cfg)

#     # checkpoint_path = "/project/3dlg-hcvc/pointgroup-minkowski/pointgroup.tar"
#     checkpoint_path = "/local-scratch/qiruiw/research/pointgroup-minkowski/output/scannet/pointgroup/DETECTOR_F/detector.pth"
#     # checkpoint_path = os.path.join(cfg.general.root, checkpoint_name)
#     # model.load_from_checkpoint(checkpoint_path, cfg)

#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint)

#     model.cuda()
#     model.eval()

#     return model

# TODO: refactor
def init_trainer(cfg):
    trainer = pl.Trainer(
        gpus=-1,  # use all available GPUs
        strategy='ddp',  # use multiple GPUs on the same machine
        num_nodes=args.num_nodes,
        profiler="simple",
    )
    return trainer


def init_model(cfg):
    MODEL = getattr(import_module(cfg.model.module), cfg.model.classname)
    model = MODEL(cfg)
    print("=> loading pretrained checkpoint from {} ...".format(cfg.pretrain))
    model.load_from_checkpoint(cfg.pretrain)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/pointgroup_scannet.yaml', help='path to config file')
    parser.add_argument('-s', '--split', type=str, default='val', help='specify data split')
    parser.add_argument('-t', '--task', type=str, default='test', help='specify task')
    args = parser.parse_args()

    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing data...")
    dataset, dataloader = init_data(cfg)

    print("=> initializing trainer...")
    trainer = init_trainer(cfg)

    print("=> initializing model...")
    pointgroup = init_model(cfg)

    print("=> start inferencing...")
    trainer.predict(model=pointgroup, dataloaders=dataloader["val"], ckpt_path=cfg.pretrain)

    # if args.task == 'test':
    #     model.inference(dataloader[args.split])
    # elif args.task == 'gt_feats':
    #     if args.split == 'train':
    #         epoch = 200
    #     else:
    #         epoch = 1
    #     model.generate_gt_features(dataloader[args.split], cfg.data.split, epoch)