import os
import sys

import mmcv
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_path)  # Manually add base path to python module search path


from detector import Detector
from detector_config import get_config
from utils import imshow_det_bboxes


def main():
    cfg = get_config()
    detector = Detector(cfg)

    test_image = os.path.join(base_path, 'test', 'test.jpg')
    
    img_data = mmcv.imread(test_image)
    img_data = mmcv.imresize(img_data, (1280, 720))

    score_threshold = 0.45
    result = detector.get_detections(img_data, score_threshold)
    print('Detections: {}'.format(result))
    result = [np.concatenate([result[0]['boxes'][j],
                             result[0]['scores'][j]],
                             axis=-1) for j in range(len(result[0]['boxes']))]

    imshow_det_bboxes(img_data,
                      result,
                      labels=list(range(len(result))),
                      class_names=("furniture", "door", "cabels", "socks"),
                      thickness=2,
                      font_scale=1,
                      out_file='./test_output.png')


if __name__ == '__main__':
    main()
