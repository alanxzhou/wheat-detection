import cv2
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.utils import shuffle


from transforms import load_mosaic, load_image, letterbox, augment_hsv, random_affine


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None, mosaic=True, augment=True):
        super().__init__()

        self.df = dataframe
        self.image_ids = dataframe['image_id'].unique()
        self.image_ids = shuffle(self.image_ids)
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * len(self.image_ids)
        self.img_size = 1024
        im_w = 1024
        im_h = 1024
        for i, img_id in enumerate(self.image_ids):
            records = self.df[self.df['image_id'] == img_id]
            boxes = records[['x', 'y', 'w', 'h']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            boxesyolo = []
            for box in boxes:
                x1, y1, x2, y2 = box
                xc, yc, w, h = 0.5 * x1 / im_w + 0.5 * x2 / im_w, 0.5 * y1 / im_h + 0.5 * y2 / im_h, abs(
                    x2 / im_w - x1 / im_w), abs(y2 / im_h - y1 / im_h)
                boxesyolo.append([0, xc, yc, w, h])
            self.labels[i] = np.array(boxesyolo)

        self.image_dir = image_dir
        self.transforms = transforms

        self.mosaic = mosaic
        self.augment = augment

    def __getitem__(self, index: int):

        # img, labels = load_mosaic(self, index)
        self.mosaic = True
        if random.randint(0, 1) == 0:
            self.mosaic = False
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=0,
                                            translate=0,
                                            scale=0,
                                            shear=0)

            # Augment colorspace
            augment_hsv(img, hgain=0.0138, sgain=0.678, vgain=0.36)

        img = torch.from_numpy(torch.from_numpy(img).permute(2, 0, 1).numpy().astype(np.float32) / 255.0)
        target = dict()
        target['boxes'] = torch.from_numpy(labels[:, 1:].astype(np.float32))
        target['labels'] = torch.ones((labels[:, 0].shape[0],), dtype=torch.int64)

        if self.transforms:
            sample = {
                'image': img,
                'bboxes': target['boxes'],
                'labels': target['labels']
            }
            sample = self.transforms(**sample)
            img = sample['image']
            target['boxes'] = torch.as_tensor(sample['bboxes'])

        return img, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]


class WheatDatasetTest(Dataset):
    # taken from  https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = dict()
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            target['boxes'] = torch.as_tensor(sample['bboxes'])

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]