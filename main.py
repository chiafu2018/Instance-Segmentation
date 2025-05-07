import os
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch.optim as optim
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import json
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 15
NUM_CLASSES = 5


class TrainDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.sample_dirs = sorted(os.listdir(root_dir))
        self.transforms = transforms

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = os.path.join(self.root_dir, self.sample_dirs[idx])
        img_path = os.path.join(sample_dir, 'image.tif')
        img = Image.open(img_path).convert("RGB")

        boxes, labels = [], []
        binary_masks, areas = [], []

        for file_name in os.listdir(sample_dir):
            if file_name.endswith('.tif') and file_name != 'image.tif':
                name_without_ext = file_name.split('.')[0]
                class_id = int(''.join(filter(str.isdigit, name_without_ext)))

                mask_path = os.path.join(sample_dir, file_name)
                mask = utils.read_maskfile(mask_path)

                if mask.sum() > 0:
                    number_of_object = np.max(mask)
                    for object_id in range(1, int(number_of_object) + 1):
                        binary_mask = (mask == object_id)

                        pos = np.nonzero(binary_mask)
                        if pos[0].size == 0 or pos[1].size == 0:
                            xmin = ymin = xmax = ymax = 0
                        else:
                            xmin = np.min(pos[1])
                            xmax = np.max(pos[1])
                            ymin = np.min(pos[0])
                            ymax = np.max(pos[0])
                        if xmax <= xmin or ymax <= ymin:
                            continue

                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id)
                        binary_masks.append(binary_mask.astype(np.uint8))
                        areas.append((xmax - xmin) * (ymax - ymin))

        target = {
            "boxes": torch.as_tensor(
                boxes, dtype=torch.float32), "labels": torch.as_tensor(
                labels, dtype=torch.int64), "masks": torch.as_tensor(
                np.stack(
                    binary_masks, axis=0), dtype=torch.uint8), "image_id": torch.tensor(
                        [idx]), "area": torch.as_tensor(
                            areas, dtype=torch.float64), "iscrowd": torch.zeros(
                                0, dtype=torch.int64)}

        if self.transforms:
            img = self.transforms(img)
        else:
            img = np.array(img)

        return img, target


def convert_to_coco_format(outputs, image_id, height, width, threshold=0.5):
    results = []
    output = outputs[0]

    boxes = output['boxes'].detach().cpu().numpy()
    scores = output['scores'].detach().cpu().numpy()
    labels = output['labels'].detach().cpu().numpy()
    masks = output['masks'].detach().cpu().numpy()

    for i in range(len(masks)):
        if scores[i] >= threshold:
            mask = masks[i, 0]
            mask = (mask > 0.5).astype(np.uint8)

            rle = utils.encode_mask(mask)

            box = boxes[i]
            xmin, xmax = min(
                float(
                    box[0]), float(
                    box[2])), max(
                float(
                    box[0]), float(
                        box[2]))
            ymin, ymax = min(
                float(
                    box[1]), float(
                    box[3])), max(
                float(
                    box[1]), float(
                        box[3]))

            bbox = [xmin, ymin, (xmax - xmin), (ymax - ymin)]

            result = {
                "image_id": int(image_id),
                "bbox": bbox,
                "score": float(scores[i]),
                "category_id": int(labels[i]),
                "segmentation": {
                    "size": [height, width],
                    "counts": rle['counts']
                }
            }
            results.append(result)

    return results


def main():
    trainset_path = "./train"

    transform = T.Compose([
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = TrainDataset(trainset_path, transforms=transform)
    trainloader = DataLoader(trainset,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=lambda x: tuple(zip(*x)))

    # model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, NUM_CLASSES)
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, min_lr=1e-8, mode='min')

    model.train()
    for epoch in range(EPOCHS):
        running_loss = []

        for images, targets in trainloader:
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()}
                       for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            running_loss.append(losses.item())

        avg_loss = sum(running_loss) / len(running_loss)
        scheduler.step(avg_loss)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Loss: {sum(running_loss):.4f}")
        torch.cuda.empty_cache()
        torch.save(model.state_dict(), f'model_{epoch + 1}.pth')


if __name__ == '__main__':
    main()
