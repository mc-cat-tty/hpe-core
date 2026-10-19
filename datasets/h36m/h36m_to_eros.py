"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author: Gaurvi Goyal

Refactor by: @mc-cat-tty
"""

import argparse
import json
import sys
from pathlib import Path

import cv2, os, contextlib, threading
import numpy as np
from tqdm import tqdm
from multiprocessing import Manager, Pool, cpu_count
from functools import partial

from datasets.utils.parsing import import_yarp_skeleton_data, batchIterator
from datasets.utils.events_representation import EROS
from datasets.utils.export import ensure_location, str2bool, get_movenet_keypoints, get_center, get_torso_length
from bimvee.importIitYarp import importIitYarp as import_dvs
import bimvee.importIitYarp as iit_yarp

### Some facility objects and functions to collect output from parallel processes and write it to stdout from the main process ###
_log_queue = None

def _drain_logs(queue):
    while (line := queue.get()) is not None:
        tqdm.write(line)

class QueueWriter:
    """File-like stdout/stderr replacement that forwards each complete line to the parent process."""

    def __init__(self, queue):
        self.queue = queue
        self.pending = ''

    def write(self, text):
        self.pending += text
        *lines, self.pending = self.pending.split('\n')
        for line in lines:
            self.queue.put(line)
        return len(text)

    def flush(self):
        if self.pending:
            self.queue.put(self.pending)
            self.pending = ''
### End of output collection facility ###


def export_to_eros(dvs_dir: Path, skeleton_file: Path, output_path: Path, args):
    skip = 1 if args.skip_image is None else args.skip_image + 1
    action_name = dvs_dir.parent.name

    data_vicon = import_yarp_skeleton_data(skeleton_file, multi_channel=True)
    data_dvs = import_dvs(filePathOrName=str(dvs_dir))
    data_dvs['data']['left']['dvs']['ts'] /= args.ts_scaler
    iterator = batchIterator(data_dvs['data']['left']['dvs'], data_vicon)
    eros = EROS(kernel_size=args.eros_kernel, frame_width=args.frame_width, frame_height=args.frame_height)

    poses_movenet = []
    if args.write_video:
        video_out = cv2.VideoWriter(str(output_path / f'{action_name}.mp4'), cv2.VideoWriter_fourcc(*'avc1'),
                                    args.fps, (args.frame_width, args.frame_height))

    for fi, (events, pose, batch_size) in enumerate(iterator):
        for ei in range(batch_size):
            eros.update(vx=int(events['x'][ei]), vy=int(events['y'][ei]))

        if fi < 2:  # Almost empty images, not beneficial for training
            kps_old = get_movenet_keypoints(pose, args.frame_height, args.frame_width)
            continue

        if fi % skip != 0:
            continue

        frame = eros.get_frame()
        sample_anno = {}
        sample_anno['img_name'] = f'{action_name}_{str(fi).zfill(5)}.jpg'
        sample_anno['ts'] = pose['ts']
        sample_anno['keypoints'] = get_movenet_keypoints(pose, args.frame_height, args.frame_width,
                                                         upperbody=args.upperbody)
        sample_anno['center'] = get_center(sample_anno['keypoints'], args.frame_height, args.frame_width)
        sample_anno['torso_size'] = get_torso_length(pose, args.frame_height, args.frame_width)
        sample_anno['keypoints_prev'] = kps_old
        sample_anno['original_sample'] = action_name

        frame = cv2.GaussianBlur(frame, (args.gauss_kernel, args.gauss_kernel), 0)
        if args.write_images:
            cv2.imwrite(str(output_path / sample_anno['img_name']), frame)
        if args.show:
            keypoints = np.reshape(sample_anno['keypoints'], [-1, 3])
            h, w = frame.shape
            for keypoint in keypoints:
                frame = cv2.circle(frame, [int(keypoint[0] * w), int(keypoint[1] * h)], 1, (255, 0, 0), 2)
            frame = cv2.circle(frame, [int(sample_anno['center'][0] * w), int(sample_anno['center'][1] * h)], 1,
                               (255, 0, 0), 4)
            cv2.imshow('', frame)
            cv2.waitKey(1)
        if args.write_video:
            video_out.write(frame)
        poses_movenet.append(sample_anno)

        kps_old = sample_anno['keypoints']

    if args.write_video:
        video_out.release()

    return poses_movenet

def append_gt_poses(args, output_json, poses_movenet):
    # Writing the GT, appending to the previous annotations if any
    if args.write_annotation and output_json is not None:
        poses = json.loads(output_json.read_text()) if output_json.exists() else []
        poses.extend(poses_movenet)
        output_json.write_text(json.dumps(poses, ensure_ascii=False))


def processed_sequences(json_file: Path):
    if not json_file.exists():
        return set()
    return {sample['original_sample'] for sample in json.loads(json_file.read_text())}


def convert_seq_to_eros(args, already_done, output_path_images, sample):
    writer = QueueWriter(_log_queue)
    with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
        poses, msg = _convert_seq_to_eros(args, already_done, output_path_images, sample)
    writer.flush()
    return poses, msg


def _convert_seq_to_eros(args, already_done, output_path_images, sample):
    cam = sample[3]  # sequence folders are named cam<C>_S<N>_<Action>
    dvs_dir = args.input_folder / sample / 'ch0dvs'
    skeleton_file = args.input_folder / sample / f'ch0GT50Hzskeleton' / 'data.log'

    if sample in already_done:
        return f'Skipping: {sample} (already processed)'
    
    if not dvs_dir.exists():
        return f'Skipping: {dvs_dir} does not exist'
    
    if not skeleton_file.exists():
        return f'Skipping: {skeleton_file} does not exist'

    poses = export_to_eros(dvs_dir, skeleton_file, output_path_images, args)
    return poses, f"Done {sample}"

def init_convert_seq_to_ros(queue):
    global _log_queue
    _log_queue = queue
    iit_yarp.tqdm = partial(tqdm, disable=True)


def parse_args():
    p = argparse.ArgumentParser(description='Convert the YARP H3.6M events and ground truth to EROS frames '
                                            'and MoveNet annotations.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input_folder', type=Path, required=True,
                   help='folder with the YARP sequences (one cam<C>_S<N>_<Action> folder each)')
    p.add_argument('--output_folder', type=Path, required=True,
                   help='folder where the frames and annotations are written (training/ or tester/ sub-folders)')
    p.add_argument('--sequences', nargs='+', default=None,
                   help='sequence folders to process (default: all the sub-folders of --input_folder)')
    p.add_argument('--eros_kernel', default=8, type=int)
    p.add_argument('--frame_width', default=640, type=int)
    p.add_argument('--frame_height', default=480, type=int)
    p.add_argument('--gauss_kernel', default=7, type=int)
    p.add_argument('--skip_image', default=None, type=int, help='number of frames to skip between two samples')
    p.add_argument('--from_scratch', type=str2bool, nargs='?', const=True, default=False,
                   help='write the annotation file from scratch')
    p.add_argument('--write_annotation', type=str2bool, nargs='?', const=True, default=False,
                   help='write the annotation file')
    p.add_argument('--write_images', type=str2bool, nargs='?', const=True, default=False, help='save images')
    p.add_argument('--write_video', type=str2bool, nargs='?', const=True, default=False, help='save video')
    p.add_argument('--fps', default=50, type=int)
    p.add_argument('--dev', type=str2bool, nargs='?', const=True, default=False,
                   help='dev mode: write to tester/ and stop after the first sequence')
    p.add_argument('--show', action='store_true',
                   help='display the frames with the keypoints (needs a graphical display)')
    p.add_argument('--ts_scaler', default=1.0, type=float)
    p.add_argument('--upperbody', type=str2bool, nargs='?', const=True, default=False,
                   help='create annotation for upperbody only')
    return p.parse_args()


def main():
    args = parse_args()
    if not args.input_folder.is_dir():
        sys.exit(f"Input folder {args.input_folder} does not exist")

    if args.dev:
        subfolder, suffix = 'tester', ''
    else:
        subfolder, suffix = 'training', '_upper' if args.upperbody else ''
    
    output_path_images = (args.output_folder / subfolder / f'h36m_EROS{suffix}').resolve()
    output_path_anno = (args.output_folder / subfolder / f'h36m_anno{suffix}').resolve()
    output_json = output_path_anno / 'poses.json'
    ensure_location(str(output_path_images))
    ensure_location(str(output_path_anno))

    if args.from_scratch:
        output_json.unlink(missing_ok=True)

    sequences = args.sequences
    if sequences is None:
        sequences = sorted(d.name for d in args.input_folder.iterdir() if d.is_dir())
    if args.dev:
        sequences = sequences[:1]
    already_done = processed_sequences(output_json)

    workers = int(cpu_count()*1.5)
    with Manager() as manager:
        log_queue = manager.Queue()
        log_reader = threading.Thread(target=_drain_logs, args=(log_queue,), daemon=True)
        log_reader.start()
        with Pool(workers, initializer=init_convert_seq_to_ros, initargs=(log_queue,)) as p:
            for poses, msg in tqdm(p.imap_unordered(partial(convert_seq_to_eros, args, already_done, output_path_images), sequences), total=len(sequences), unit='sequence'):
                if poses: append_gt_poses(args, output_json, poses)
                if msg: tqdm.write(msg)
        log_queue.put(None)
        log_reader.join()
    


if __name__ == '__main__':
    main()
