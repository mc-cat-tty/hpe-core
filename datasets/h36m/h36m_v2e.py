import argparse
import glob
import os
import sys
import cdflib
from tqdm import tqdm

ALL_CAMERAS = {1: '54138969', 2: '55011271', 3: '58860488', 4: '60457274'}
DEFAULT_SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
FRAME_SIZE = 1000  # H3.6M videos are 1000x1000


def get_cropping(anno_file, output_width, output_height, buffer):
    cdf_file = cdflib.CDF(anno_file)
    data = (cdf_file.varget("Pose")).squeeze()
    data2 = data.reshape([-1, 2], order='C')
    left = min(data2[:, 0])
    right = max(data2[:, 0])
    top = min(data2[:, 1])
    bottom = max(data2[:, 1])
    width = right - left + buffer
    height = bottom - top + buffer
    center = [int(left + width / 2), int(top + height / 2)]
    if width < output_width and height < output_height:
        crop_left = center[0] - int(output_width / 2)
        crop_right = FRAME_SIZE - int(center[0] + output_width / 2)
        crop_top = center[1] - int(output_height / 2)
        crop_bottom = FRAME_SIZE - int(center[1] + output_height / 2)
        assert crop_left + crop_right == FRAME_SIZE - output_width
        assert crop_top + crop_bottom == FRAME_SIZE - output_height
    else:
        if 0.75 * width > height:
            height = 0.75 * width
        else:
            width = height / 0.75
        if width > FRAME_SIZE:
            width = FRAME_SIZE
            center[0] = FRAME_SIZE // 2
        if height > FRAME_SIZE:
            height = FRAME_SIZE
            center[1] = FRAME_SIZE // 2

        crop_left = center[0] - int(width / 2)
        crop_right = FRAME_SIZE - int(center[0] + width / 2)
        crop_top = center[1] - int(height / 2)
        crop_bottom = FRAME_SIZE - int(center[1] + height / 2)

        if crop_left < 0:
            crop_right = crop_right + crop_left
            crop_left = 0
        if crop_right < 0:
            crop_left = crop_right + crop_left
            crop_right = 0
        if crop_top < 0:
            crop_bottom = crop_bottom + crop_top
            crop_top = 0
        if crop_bottom < 0:
            crop_top = crop_bottom + crop_top
            crop_bottom = 0

    return f'{crop_left},{crop_right},{crop_top},{crop_bottom}'


def find_videos(dataset_path, subjects, cams):
    """Return (video, annotation, output name) for every non-ALL video of the selected subjects and cameras."""
    videos = []
    for sub in subjects:
        for cam in cams:
            pattern = os.path.join(glob.escape(dataset_path), sub, 'Videos', '*' + ALL_CAMERAS[cam] + '.mp4')
            for file in sorted(glob.glob(pattern)):
                if 'ALL' in file:
                    continue
                anno_file = os.path.join(dataset_path, sub, 'Poses_D2_Positions',
                                         os.path.basename(file).replace('mp4', 'cdf'))
                out_name = f"cam{cam}_{sub}_{os.path.basename(file).split('.')[0].replace(' ', '_')}"
                videos.append((file, anno_file, out_name))
    return videos


def build_v2e_args(args, video, output_folder, crop):
    v2e_argv = [
        '-i', video,
        '--overwrite',
        f'--timestamp_resolution={args.ts_res}',
        f'--auto_timestamp_resolution={args.auto_ts_res}',
        '--dvs_exposure', 'duration', str(args.dvs_exposure_duration),
        f'--output_folder={output_folder}',
        f'--pos_thres={args.pos_thres}',
        f'--neg_thres={args.neg_thres}',
        f'--sigma_thres={args.sigma_thres}',
        '--dvs_h5', args.dvs_h5,
        f'--output_width={args.output_width}',
        f'--output_height={args.output_height}',
        f'--cutoff_hz={args.cutoff_hz}',
        '--dvs_aedat2', 'None',
        '--dvs640',
        '--crop', crop,
    ]
    if args.slomo_model is not None:
        v2e_argv += ['--slomo_model', args.slomo_model]
    if not args.dev:
        v2e_argv += ['--batch_size', str(args.batch_size), '--skip_video_output', '--no_preview']
    if args.stop_time is not None:
        v2e_argv += ['--stop_time', str(args.stop_time)]
    return v2e_argv


def run_v2e(v2e_argv):
    # v2e's main() parses sys.argv, so we set it for the duration of the call.
    from v2ecore.v2e import main as v2e_main
    old_argv = sys.argv
    sys.argv = ['v2e'] + v2e_argv
    try:
        v2e_main()
    finally:
        sys.argv = old_argv


def parse_args():
    p = argparse.ArgumentParser(description='Run v2e on the Human3.6M videos, cropping around the subject.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset_path', required=True,
                   help='root of the extracted H3.6M dataset (contains S1, S5, ... folders)')
    p.add_argument('--output_folder', required=True, help='base folder where the events are written')
    p.add_argument('--slomo_model', default=None,
                   help='path to the SuperSloMo checkpoint (not part of this repo, it ships with v2e: '
                        'https://github.com/SensorsINI/v2e); if omitted, v2e uses its own default')
    p.add_argument('--cropping_log_file', default=None,
                   help='file where the crop of each processed video is appended '
                        '(default: <output_folder>/cropping_data.txt)')
    p.add_argument('--subjects', nargs='+', default=DEFAULT_SUBJECTS, help='subjects to process')
    p.add_argument('--cams', nargs='+', type=int, default=[2, 4], choices=sorted(ALL_CAMERAS),
                   help='camera indices to process')
    p.add_argument('--ts_res', type=float, default=.001, help='v2e timestamp resolution')
    p.add_argument('--auto_ts_res', action='store_true', help='enable v2e auto timestamp resolution')
    p.add_argument('--dvs_exposure_duration', type=float, default=0.005, help='DVS exposure duration')
    p.add_argument('--pos_thres', type=float, default=.15, help='positive event threshold')
    p.add_argument('--neg_thres', type=float, default=.15, help='negative event threshold')
    p.add_argument('--sigma_thres', type=float, default=0.03, help='event threshold sigma')
    p.add_argument('--dvs_h5', default='events', help='name of the output hdf5 file')
    p.add_argument('--output_width', type=int, default=640)
    p.add_argument('--output_height', type=int, default=480)
    p.add_argument('--buffer', type=int, default=20, help='margin (pixels) added around the pose when cropping')
    p.add_argument('--cutoff_hz', type=float, default=15)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--stop_time', type=float, default=None, help='stop each video at this time (s)')
    p.add_argument('--dev', action='store_true',
                   help='debug mode: write to <output_folder>/tester, keep video output and stop after one file')
    return p.parse_args()


def main():
    args = parse_args()

    base_output_folder = os.path.join(args.output_folder, 'tester' if args.dev else 'events_fullHD')
    cropping_log_file = args.cropping_log_file or os.path.join(args.output_folder, 'cropping_data.txt')
    os.makedirs(base_output_folder, exist_ok=True)

    videos = find_videos(args.dataset_path, args.subjects, args.cams)
    if not videos:
        sys.exit(f"No videos found: expected {os.path.join(args.dataset_path, '<subject>', 'Videos', '*<camera id>.mp4')} "
                 f"for subjects {args.subjects} and cams {args.cams}. "
                 "Check that --dataset_path points to the extracted H3.6M root (the folder containing S1, S5, ...).")
    for i, (video, anno_file, out_name) in enumerate(tqdm(videos)):
        output_folder = os.path.join(base_output_folder, out_name)
        if os.path.exists(output_folder):
            print(f"Folder already exists. File {video} being skipped.")
            continue

        crop = get_cropping(anno_file, args.output_width, args.output_height, args.buffer)
        print(f"\nFile {i + 1} of {len(videos)} being processed.\n")
        run_v2e(build_v2e_args(args, video, output_folder, crop))
        print(f"File {video} processed.")

        with open(cropping_log_file, 'a') as f:
            f.write(f"{out_name} {crop}\n")
        if args.dev:
            break

    print("Process completed.")


if __name__ == '__main__':
    main()
