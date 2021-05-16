import argparse
from glob import glob
import os
import time

import cv2
import mmcv
import numpy as np

from object_detection.detector import Detector
from object_detection.detector_config import get_config
from object_detection.utils import imshow_det_bboxes


def make_video(detector, frames, outdir, score_threshold=0.35, fps=15, w=1424, h=800):
    frames = sorted(frames)
    seq_name = os.path.splitext(os.path.basename(frames[0]))[0]
    seq_name = seq_name.split('_fnum')[0]

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    output_video = os.path.join(outdir, seq_name + '.avi')
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"MJPG"), fps,
                          (w, h))

    for i, frame_read in enumerate(frames):
        prev_time = time.time()
        frame_read = mmcv.imread(frame_read)
        if frame_read is None:
            continue
        frame_read = mmcv.imresize(frame_read, (w, h))
        result = detector.get_detections(frame_read, score_threshold)

        result = np.concatenate([result[0]['boxes'], result[0]['scores']],
                                axis=-1)

        image = imshow_det_bboxes(frame_read,
                                  result,
                                  labels=np.zeros(len(result), dtype='int32'),
                                  class_names=('road_sign',),
                                  thickness=1,
                                  font_scale=0.35)
        out.write(image)
        print('\rProcessing frame: {}/{} | Current FPS: {}'.format(
            i + 1, len(frames), np.round(1 / (time.time() - prev_time), 3)),
              end='')
    out.release()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_path', type=str, required=True,
                        help="Path to the sequence.")
    parser.add_argument('--out_dir', type=str, required=False, default='output_videos',
                        help="Output directory.")
    parser.add_argument('--score_threshold', type=float, required=False, default=0.35,
                        help="Score threshold for filtering detections.")
    parser.add_argument('--fps', type=int, required=False, default=15,
                        help="Output video FPS.")
    parser.add_argument('--width', type=int, required=False, default=1280,
                        help="Output video width.")
    parser.add_argument('--height', type=int, required=False, default=720,
                        help="Output video height.")

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    cfg = get_config()
    detector = Detector(cfg)

    frames = glob(os.path.join(args.sequence_path, '*'))

    make_video(
        detector,
        frames,
        args.out_dir,
        args.score_threshold,
        args.fps,
        args.width,
        args.height)


if __name__ == '__main__':
    main()
