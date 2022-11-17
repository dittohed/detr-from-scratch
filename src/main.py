import torch

from torch.utils.data import DataLoader

from dataset.coco import build_dataset
from utils.misc import collate_fn 


def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: set seeds for reproducibility
    # TODO: build model

    # TODO: for now let's try default samplers
    dataset_train = build_dataset(cfg['coco_dir'], 'train')
    loader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, 
                    collate_fn=collate_fn)  # TODO: set num_workers correctly
    
    for epoch in range(cfg['n_epochs']):
        # TODO: train_one_epoch func
        for nested_tensor, targets in loader_train:
            pass


if __name__ == '__main__':
    # TODO: Read config file to cfg
    cfg = {
        'coco_dir': 'data',
        'n_epochs': 1
    }

    main(cfg)