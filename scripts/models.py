import albumentations as A
import math
import numpy as np
import os
import pandas as pd
import torch
import torchvision

from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from utils import Averager, plot_grad_flow, WheatDataset


class WheatModel:

    def __init__(self, base_path, num_epochs=5, train_val_split=0.8, model_name='faster_rcnn', optimizer=None, lr_scheduler=None, transforms=None, weights_file=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_path = base_path
        self.train_path = os.path.join(base_path, 'train')
        self.num_epochs = num_epochs
        self.train_df = None
        self.val_df = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_val_split = train_val_split
        self.transforms = transforms
        self.load_data()
        self.train_data_loader = None
        self.valid_data_loader = None
        self.set_dataloader()
        self.num_classes = 2
        self.model_name = model_name
        self.model = None
        self.select_model(model_name)  # selects model name and model
        self.model.to(self.device)
        self. params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.params, lr=0.005, momentum=0.9, weight_decay=0.0005) \
            if optimizer is None else optimizer
        self.lr_scheduler = lr_scheduler
        if weights_file:
            self.load_weights(weights_file)

    def load_data(self):
        train_df = pd.read_csv(os.path.join(self.base_path, 'train.csv'))
        train_df['x'] = -1
        train_df['y'] = -1
        train_df['w'] = -1
        train_df['h'] = -1
        bboxes = train_df['bbox'].copy().apply(lambda x: eval(x))
        train_df[['x', 'y', 'w', 'h']] = np.stack(bboxes).astype(int)

        image_ids = train_df['image_id'].unique()
        n_ids = len(image_ids)
        n_train = round(self.train_val_split * n_ids)
        train_ids = image_ids[-n_train:]  # I think everything's already been randomized so we don't need to do it again
        valid_ids = image_ids[:-n_train]
        valid_df = train_df[train_df['image_id'].isin(valid_ids)]
        train_df = train_df[train_df['image_id'].isin(train_ids)]
        self.train_df = train_df
        self.val_df = valid_df
        self.train_dataset = WheatDataset(train_df, self.train_path, self.get_train_transform())
        self.val_dataset = WheatDataset(valid_df, self.train_path, self.get_valid_transform())

    def select_model(self, model_name):
        self.model_name = model_name
        if model_name == 'faster_rcnn':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features

            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            weights_model_name = 'fasterrcnn'
        elif model_name == 'mask_rcnn':
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            dim_reduced = 2  # not sure what this should be yet

            model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, dim_reduced, self.num_classes)
            weights_model_name = 'maskrcnn'
        else:
            raise ValueError('Not a valid model name')
        self.model = model

    def get_train_transform(self):
        transforms = [] if self.transforms is None else self.transforms
        return A.Compose(transforms + [ToTensor()], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    @staticmethod
    def get_valid_transform():
        return A.Compose([
            ToTensor()
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    def set_dataloader(self, train_batch_size=4, val_batch_size=4, shuffle=True, num_workers=0):
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

        if self.val_dataset:
            self.valid_data_loader = DataLoader(
                self.val_dataset,
                batch_size=val_batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn
            )

    def get_full_dataset(self):
        return pd.concat([self.train_df, self.val_df])

    def load_weights(self, weights_file):
        self.model.load_state_dict(torch.load(weights_file))

    def main(self):
        loss_hist = Averager()
        loss = []
        scores = []
        itr = 1

        for epoch in range(self.num_epochs):
            loss_hist.reset()

            for images, targets, image_ids in self.train_data_loader:

                images = list(image.to(self.device) for image in images)
                #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                targets = [{k: v.long().to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                loss_hist.send(loss_value)
                loss.append(loss_value)

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                if math.isnan(loss_value):
                    plot_grad_flow(self.model.named_parameters())
                    raise ValueError('Loss is nan')
                if itr % 50 == 0:
                    print(f"Iteration #{itr} loss: {loss_value}")

                itr += 1

            # update the learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print(f"Epoch #{epoch} loss: {loss_hist.value}")
        return loss

    def save_params(self, save_name=None):
        if not save_name:
            save_name = f'{self.model_name}_resnet50_fpn_{self.num_epochs}epochs.pth'
        torch.save(self.model.state_dict(), save_name)


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    None
