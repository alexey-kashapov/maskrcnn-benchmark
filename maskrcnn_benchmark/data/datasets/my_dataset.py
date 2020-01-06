from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from PIL import Image
import cv2
import numpy as np
import scipy.io
import imageio
import h5py
import os
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
from skimage import io
import skimage.transform
import random
import torchvision
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
    def __init__(self, transform=None, phase_train=True, data_dir="../dataset/"):
        self.phase_train = phase_train
        self.root = data_dir
        self.transform = transform

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

        color = Image.open(color_path)
        depth = Image.open(depth_path)

        mask = Image.open(mask_path)
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # get bounding box coordinates for each mask
        boxes = []
        labels = []
        masks = []

        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray_mask, 1, 255, 0)
        _, contours, _ = cv2.findContours(thresh, 1, 2)

        for cnt in contours:
            xmin, ymin, w, h = cv2.boundingRect(cnt)
            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])

            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            if mask[cy, cx] in np.array(label_colours):
                labels.append(np.where(np.array(label_colours) == mask[cy, cx])[0][0])
                masks.append(cnt)


        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # create a BoxList from the boxes
        target = BoxList(boxes, color.size, mode="xyxy")
        # add the labels to the boxlist
        target.add_field("labels", labels)

        masks = SegmentationMask(masks, color.size, mode='poly')
        target.add_field("masks", masks)

        if self.transforms:
            color, depth, target = self.transforms(color, depth, target)

        # return the image, the boxlist and the idx in your dataset
        return color, depth, target, idx

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