import cv2
import numpy as np
from tqdm import tqdm
import json

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class VaccumDataset(CustomDataset):

    CLASSES = ("furniture", "door", "cabel", "sock")
    
    def __init__(self, *args, **kwargs):
        np.random.seed(kwargs.pop('seed'))
        path = 'dataset/check/check.json'
        with open(path) as json_file:
            self._dl = json.load(json_file)
        kwargs.pop('dataset_name')
        kwargs.pop('data_split')
        self._split = kwargs.pop('split')
        self._split_ratio = kwargs.pop('split_ratio')
        
        super(VaccumDataset, self).__init__(ann_file=None, *args, **kwargs)

    def load_annotations(self, *args):
        print('Computing image sizes')

        frames = list(self._dl.keys())

        num_train_samples = int(self._split_ratio * len(frames))
        idx = np.arange(0, len(frames))
        np.random.shuffle(idx)
        idx = idx[:num_train_samples] if self._split == 'train' else idx[num_train_samples:]
        
        samples = []
        mapping = {"furniture":0, "door":1, "cabel":2, "sock":3}
        for i in tqdm(idx):
            frame_info = self._dl[frames[i]]
            
            boxes = []
            labels = []
            for box in frame_info:
                boxes += [[box[3], box[4], box[5], box[6]]]
                labels.append(mapping[box[2]])
            
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

            height, width = box[1], box[0]

            samples += [{
                'filename': frames[i],
                'width': width,
                'height': height,
                'ann': {
                    'bboxes': boxes,
                    'labels': labels,
                }
            }]
        return samples


# pipeline=[
#     dict(type='LoadImageFromFile', to_float32=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='MinIoURandomCrop'),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='Normalize',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         to_rgb=True),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
# d1 = VaccumDataset(pipeline = pipeline, split='train', split_ratio=0.5)
# print ('hi')