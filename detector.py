import numpy as np
import torch

from mmdetection.mmdet.apis import init_detector, inference_detector


class Detector:

    def __init__(self, cfg):
        """
        :param cfg: EasyDict with configuration parameters
        :param weights: Path to the trained pytorch weights file.
        """
        self.model = init_detector(
            cfg.model_cfg,
            cfg.trained_weights,
            device='cuda:1' if torch.cuda.is_available() else 'cpu')

    def _get_detections(self, image, score_threshold):
        """
        :param image: An image in the form of numpy array.
        :param score_threshold: A float value representing the minimum confidence of detected boxes.
        """
        all_boxes = []
        all_scores = []
        results = inference_detector(self.model, image)
        [all_boxes.append(results[i][:, :-1]) for i in range(len(results))]
        [all_scores.append(results[i][:, -1]) for i in range(len(results))]

        mask = [i>= score_threshold for i in all_scores]

        all_boxes = np.array([all_boxes[i][mask[i]] for i in range(len(all_boxes))])
        all_scores = np.array([all_scores[i][mask[i]] for i in range(len(all_boxes))])

        # Handles the scenario when there are no detections
        if len(all_boxes.shape) == 1:

            for i in range(all_boxes.shape[0]):
                if all_boxes[i].shape[0] == 0:
                    all_boxes[i] = ( -1 * np.ones([1, 4]))
                    all_scores[i] = ( np.zeros([1, 1]))

                else:
                    all_scores[i] = np.expand_dims(all_scores[i], axis=1)

            detections = {'boxes': all_boxes, 'scores': all_scores}
        else:

            for i in range(all_boxes.shape[0]):
                all_boxes = np.array([-1 * np.ones([1, 4]) for i in range(4)])
                all_scores = np.array([np.zeros([1,1]) for i in range(4)])
            detections = {'boxes': all_boxes, 'scores': all_scores}
        return detections

    def get_detections(self, image_list, score_threshold):
        """
        :param image_list: A list of images in the form of numpy arrays.
        :param score_threshold: A float value representing the minimum confidence of detected boxes.
        """
        if not isinstance(image_list, list):
            image_list = [image_list]

        # get detections for each image
        detections = [
            self._get_detections(image, score_threshold) for image in image_list
        ]
        return detections
