import argparse
from glob import glob
import os
import time

import cv2
import mmcv
import numpy as np

from detector import Detector
from detector_config import get_config
from utils import imshow_det_bboxes


def make_video(detector, inp_vid_path, outdir, score_threshold=0.45, fps=15, w=1280, h=720):

    seq_name = inp_vid_path.split('/')[-1][:-4]+"_detector"


    if not os.path.exists(outdir):
        os.mkdir(outdir)

    output_video = os.path.join(outdir, seq_name + '.avi')

    cap = cv2.VideoCapture(inp_vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"MJPG"), fps,
                          (w, h))
    count = 0
    num_warmup = 5
    pure_inf_time = 0

    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()

        if ret:
            if not frame_read.shape == (1280,720):
                frame_read = mmcv.imresize(frame_read, (w, h))
            result = detector.get_detections(frame_read, score_threshold)

            result = [np.concatenate([result[0]['boxes'][j],
                                    result[0]['scores'][j]],
                                    axis=-1) for j in range(len(result[0]['boxes']))]

            image = imshow_det_bboxes(frame_read,
                                    result,
                                    labels=list(range(len(result))),
                                    class_names=("furniture", "door", "cabels", "socks"),
                                    thickness=2,
                                    font_scale=0.8)
            out.write(image)
            count += 1

            if count >= num_warmup:
                pure_inf_time += time.time() - prev_time
                model_fps = (count + 1 - num_warmup) / pure_inf_time
                print(f'\rProcessing frame: {count + 1}/{num_frames} | Current FPS: {model_fps}',end='')
        else:
            break
    out.release()
    cap.release()
    # cv2.destroyAllWindows()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_vid_path', type=str, required=False,
                        default='/home/heh3kor/vaccum_thon/test/test_video.mp4',
                        help="Path to the video.")
    parser.add_argument('--out_dir', type=str, required=False, default='output_videos',
                        help="Output directory.")
    parser.add_argument('--score_threshold', type=float, required=False, default=0.45,
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

    # frames = glob(os.path.join(args.sequence_path, '*'))

    make_video(
        detector,
        args.inp_vid_path,
        args.out_dir,
        args.score_threshold,
        args.fps,
        args.width,
        args.height)


if __name__ == '__main__':
    main()
