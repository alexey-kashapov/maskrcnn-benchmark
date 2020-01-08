from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from PIL import Image
import cv2
import numpy as np
import os
from torch.utils.data import Dataset

import torch

num_classes = 7

label_colours = [[0,0,0],
                 # 0=background
                 [17, 160, 96], [6, 17, 253], [137, 92, 167],
                 [121, 237, 156], [169, 18, 8], [148, 84, 29]]

class2color = {"camera" : [17, 160, 96],
               "screw" : [6, 17, 253],
               "motherboard" : [137, 92, 167],
               "connector" : [121, 237, 156],
               "cable" : [169, 18, 8],
               "battery" : [148, 84, 29]
}

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None, phase_train=True):
        self.phase_train = phase_train
        self.root = data_dir
        self.transforms = transforms

        if self.phase_train:
            list = os.listdir(self.root)  # dir is your directory path
        else:
            list = os.listdir(self.root)  # dir is your directory path
        self.len = round(len(list) / 4)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # load the image as a PIL Image
        if self.phase_train:
            color_path = os.path.join(self.root, str(idx) + "_color.png")
            depth_path = os.path.join(self.root, str(idx) + "_depth.png")
            mask_path = os.path.join(self.root, str(idx) + "_uncolor_mask.png")
        else:
            color_path = os.path.join(self.root, str(idx) + "_color.png")
            depth_path = os.path.join(self.root, str(idx) + "_depth.png")
            mask_path = os.path.join(self.root, str(idx) + "_uncolor_mask.png")

        color_img = Image.open(color_path)
        depth_img = Image.open(depth_path)

        mask = Image.open(mask_path)
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
        # first id is the background, so remove it

        # get bounding box coordinates for each mask
        boxes = []
        labels = []
        masks = []

        for col_num, color in enumerate(label_colours[:1]):
            color_mask = np.where(np.array(color) == mask)[:2]
            if color_mask[0].size != 0:
                new_img = np.zeros(mask.shape[:2], dtype=np.uint8)
                new_img[color_mask] = 255

                _ , contours, _ = cv2.findContours(new_img, 1, 2)
                for cnt in contours:
                    cnt_mask = np.zeros(mask.shape[:2], np.uint8)
                    cv2.drawContours(cnt_mask, [cnt], 0, 255, -1)

                    xmin, ymin, w, h = cv2.boundingRect(cnt)
                    xmax = xmin + w
                    ymax = ymin + h
                    boxes.append([xmin, ymin, xmax, ymax])

                    labels.append(col_num + 1)
                    masks.append(cnt_mask)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # create a BoxList from the boxes
        target = BoxList(boxes, color_img.size, mode="xyxy")
        # add the labels to the boxlist
        target.add_field("labels", labels)

        # convert everything into a torch.Tensor
        masks = torch.as_tensor(masks, dtype=torch.bool)

        masks = SegmentationMask(masks, color_img.size, mode='mask')
        target.add_field("masks", masks)

        if self.transforms:
            color_img, depth_img, target = self.transforms(color_img, depth_img, target)

        # return the image, the boxlist and the idx in your dataset
        return color_img, depth_img, target

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        if self.phase_train:
            color_path = os.path.join(self.root, str(idx) + "_color.png")
        else:
            color_path = os.path.join(self.root, str(idx) + "_color.png")

        color = Image.open(color_path)
        width, height = color.size
        return {"height": height, "width": width}