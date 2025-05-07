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
NUM_CLASSES = 5

class TestDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.tif')])
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.transforms:
            img = self.transforms(img)

        return img, img_name 
    

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
            xmin, xmax = min(float(box[0]), float(box[2])), max(float(box[0]), float(box[2]))
            ymin, ymax = min(float(box[1]), float(box[3])), max(float(box[1]), float(box[3]))

            bbox = [xmin, ymin, (xmax-xmin), (ymax-ymin)]

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



def main(i:int): 
    testset_path = "./test_release"

    transform = T.Compose([
        T.ToTensor(),  
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testset = TestDataset(testset_path, transforms=transform)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
    model.to(DEVICE)

    model.load_state_dict(torch.load(f'model_{i}.pth', map_location=DEVICE))

    all_predictions = []
   

    with open('test_image_name_to_ids.json', 'r') as f:
        test_image_info = json.load(f)

    file_to_id_map = {item['file_name']: item['id'] for item in test_image_info}
    file_to_height_map = {item['file_name']: item['height'] for item in test_image_info}
    file_to_width_map = {item['file_name']: item['width'] for item in test_image_info}

    model.eval()
    for image, img_name in testloader:
        image = list(img.to(DEVICE) for img in image)

        img_name = img_name[0] 
        image_id = file_to_id_map[img_name] 
        height = file_to_height_map[img_name]
        width = file_to_width_map[img_name]

        with torch.no_grad():
            outputs = model(image)

        coco_outputs = convert_to_coco_format(outputs, image_id=image_id, height=height, width=width, threshold=0.5)
        all_predictions.extend(coco_outputs) 


    with open(f"test-results-{i}.json", "w") as f:
        json.dump(all_predictions, f, indent=2)

    print(f"All {len(all_predictions)} predictions saved to predictions.json")


if __name__ == '__main__':
    for i in range(15):
        main(i+1)