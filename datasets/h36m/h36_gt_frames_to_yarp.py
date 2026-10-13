#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reading the GT and videos form h36 and generating Yarp compatible formats

Copyright (C) 2021 Event-driven Perception for Robotics
Author: Franco Di Pietro and Gaurvi Goyal
Refactor by: @mc-cat-tty
"""

import argparse
import sys
from pathlib import Path

import cdflib
import cv2
import numpy as np
from tqdm import tqdm

from datasets.h36m.h36m_v2e import ALL_CAMERAS, DEFAULT_SUBJECTS, find_videos
from datasets.h36m.utils import parsing
from datasets.utils.constants import HPECoreSkeleton
from datasets.utils.export import skeleton_to_yarp_row, format_crop_file
from datasets.utils.export import crop_pose, crop_frame


def write_video_and_pose(input_video_path: Path, gt_path: Path, cam: int, directory_frames: Path,
                         directory_skl: Path, output_width: int, output_height: int, write_frames=True,
                         write_pose=True, overwrite=False, crop=None, write_video=False, output_video_path=None):
    """Convert the video and annotations to yarp formats. Returns 1 if the video cannot be opened, 0 otherwise."""
    vid = cv2.VideoCapture(str(input_video_path))
    if not vid.isOpened():
        return 1

    counter = 0
    frame_lines = []
    pose_lines = []
    fps = vid.get(cv2.CAP_PROP_FPS)
    data = cdflib.CDF(str(gt_path)).varget("Pose").squeeze()

    if not overwrite and (directory_skl / 'data.log').is_file():
        write_pose = False
    if write_frames:
        directory_frames.mkdir(parents=True, exist_ok=True)
    directory_skl.mkdir(parents=True, exist_ok=True)
    if write_video:
        video_out = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                    (output_width, output_height))

    while vid.isOpened():
        frame_exists, frame = vid.read()

        if frame_exists:
            timestamp = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000  # convert timestamp to seconds
        else:
            break

        if counter > 10 and timestamp == 0.0:
            break

        try:
            skeleton = data[counter, :]
        except IndexError:
            break

        filename = f'frame_{str(counter).zfill(10)}.png'

        frame = crop_frame(frame, crop)
        dim_input = frame.shape

        if write_frames or write_video:
            frame_resized = cv2.resize(src=frame, dsize=(output_width, output_height), interpolation=cv2.INTER_AREA)
        if write_frames:
            cv2.imwrite(str(directory_frames / filename), frame_resized)  # create the images
        if write_video:
            video_out.write(frame_resized)

        # convert h3.6m joints order to hpecore one
        skeleton = skeleton.reshape(-1, 2)
        skeleton = parsing.h36m_to_hpecore_skeleton(skeleton)

        # rescale skeleton
        skeleton = crop_pose(skeleton, crop)
        skeleton[:, 0] = skeleton[:, 0] * output_width / dim_input[1]
        skeleton[:, 1] = skeleton[:, 1] * output_height / dim_input[0]
        skeleton = np.rint(skeleton).astype(int)

        torso_size = HPECoreSkeleton.compute_torso_sizes(skeleton)

        # append strings that will be written to yarp's data.log
        frame_lines.append(" %d %.6f %s [rgb]" % (counter, timestamp, filename))
        pose_lines.append(skeleton_to_yarp_row(counter, timestamp, skeleton, torso_size=torso_size))
        counter += 1

    # info.log
    frame_lines_info = ['Type: Image;', f'[0.0] /file/ch{cam}frames:o [connected]']
    pose_lines_info = ['Type: Bottle;', f'[0.0] /file/ch{cam}GT50Hzskeleton:o [connected]']

    vid.release()

    if write_video:
        video_out.release()
    if write_frames:
        parsing.writer(str(directory_frames), frame_lines, frame_lines_info)
    if write_pose:
        parsing.writer(str(directory_skl), pose_lines, pose_lines_info)

    return 0


def parse_args():
    p = argparse.ArgumentParser(description='Convert the H3.6M videos and ground truth to YARP formats, '
                                            'optionally cropping them around the subject.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset_path', type=Path, required=True,
                   help='root of the extracted H3.6M dataset (contains S1, S5, ... folders)')
    p.add_argument('--output_folder', type=Path, required=True, help='folder where the YARP sequences are written')
    p.add_argument('--crop_file', type=Path, default=None,
                   help='file with the crop of each sequence (left,right,top,bottom), as written by h36m_v2e; '
                        'no cropping if omitted')
    p.add_argument('--video_folder', type=Path, default=None,
                   help='folder where the cropped videos are written (default: <output_folder>/../cropped_video)')
    p.add_argument('--error_log', type=Path, default=None,
                   help='file where the videos that cannot be opened are listed '
                        '(default: <output_folder>/../errorlog.txt)')
    p.add_argument('--subjects', nargs='+', default=DEFAULT_SUBJECTS, help='subjects to process')
    p.add_argument('--cams', nargs='+', type=int, default=[2, 4], choices=sorted(ALL_CAMERAS),
                   help='camera indices to process (2 and 4 are the front facing ones)')
    p.add_argument('--output_width', type=int, default=640)
    p.add_argument('--output_height', type=int, default=480)
    p.add_argument('--write_pose', action='store_true', help='write the ground-truth skeletons')
    p.add_argument('--write_frames', action='store_true', help='write the frames as images')
    p.add_argument('--write_video', action='store_true', help='write the cropped and resized videos')
    return p.parse_args()


def main():
    args = parse_args()
    if not (args.write_pose or args.write_frames or args.write_video):
        sys.exit("Nothing to do: select at least one of --write_pose, --write_frames, --write_video")

    video_folder = args.video_folder or args.output_folder.parent / 'cropped_video'
    error_log = args.error_log or args.output_folder.parent / 'errorlog.txt'

    crop_dict = None
    if args.crop_file is not None:
        if not args.crop_file.is_file():
            sys.exit(f"Cropping file {args.crop_file} does not exist")
        crop_dict = format_crop_file(args.crop_file.read_text().splitlines())

    videos = find_videos(str(args.dataset_path), args.subjects, args.cams)
    if not videos:
        sys.exit(f"No videos found in {args.dataset_path}: it should contain <subject>/Videos/*.mp4")
    if args.write_video:
        video_folder.mkdir(parents=True, exist_ok=True)

    for video_file, pose_file, sequence in tqdm(videos):
        video_file, pose_file = Path(video_file), Path(pose_file)
        cam = int(sequence[3])  # sequence names are cam<C>_S<N>_<Action>
        if not pose_file.is_file():
            continue

        crop_values = None
        if crop_dict is not None:
            crop_values = crop_dict.get(sequence)
            if crop_values is None:
                tqdm.write(f"Cropping values not present in file for {sequence}. Proceeding without cropping")

        exitcode = write_video_and_pose(
            video_file, pose_file, cam,
            args.output_folder / sequence / f'ch{cam}frames', args.output_folder / sequence / f'ch{cam}GT50Hzskeleton',
            args.output_width, args.output_height,
            write_frames=args.write_frames, write_pose=args.write_pose, overwrite=False, crop=crop_values,
            write_video=args.write_video, output_video_path=video_folder / f'{sequence}.mp4')

        if exitcode:
            with open(error_log, 'a') as f:
                f.write(f"{video_file}\n")


if __name__ == '__main__':
    main()


# limbs = {0: 'PelvisC', 1: 'PelvisR', 2: 'KneeR', 3: 'AnkleR', 4: 'ToeR', 5: 'ToeROther', 6: 'PelvisL', 7: 'KneeL',
# 8: 'AnkleR', 9: 'ToeR', 10: 'ToeROther', 11: 'Spine', 12: 'SpineM', 13: 'Neck', 14: 'Head', 15: 'HeadOther',
# 16: 'NextAgain', 17: 'ShoulderL', 18: 'ElbowL', 19: 'WristL', 20: 'WristLAgain', 21: 'ThumbL', 22: 'BOHR', 23: 'BOHRAgain',
# 24: 'NeckAgainAgain', 25: 'ShoulderR', 26: 'ElbowR', 27: 'WristR', 28: 'WristAgain', 29: 'ThumbR', 30: 'BOHL', 31: 'BOHLAgain'}

# 0 = 11
# 13 = 16 = 24
# 19 = 20
# 22 = 23
# 26 = 27
# 30 = 31