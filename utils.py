import cv2
import numpy as np

import mmcv


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      bbox_color='blue',
                      text_color='green',
                      thickness=1,
                      font_scale=0.35,
                      out_file=None):
    # assert bboxes.ndim == 2
    # assert labels.ndim == 1
    # assert bboxes.shape[0] == labels.shape[0]
    # assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    img = mmcv.imread(img)

    bbox_color = mmcv.color_val(bbox_color)
    text_color = mmcv.color_val(text_color)
    img = np.ascontiguousarray(img)
    for bbox, label in zip(bboxes, labels):
        # convert_box = bbox.astype(np.int32)
        for bbox_int in bbox:
            if all(bbox_int[:4] == np.array([-1.0, -1.0, -1.0, -1.0])): # if unable to predict skip that frame.
                continue
            left_top = (int(bbox_int[0]), int(bbox_int[1]))
            right_bottom = (int(bbox_int[2]), int(bbox_int[3]))
            cv2.rectangle(
                img, left_top, right_bottom, bbox_color, thickness=thickness)
            label_text = class_names[
                label] if class_names is not None else f'cls {label}'
            if bbox_int.shape[0]>4:
                label_text += f'|{bbox_int[-1]:.02f}'
            cv2.putText(img, label_text, (int(bbox_int[0]), int(bbox_int[1]) - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if out_file is not None:
        return mmcv.imwrite(img, out_file)
    return img
