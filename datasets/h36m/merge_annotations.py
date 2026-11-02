"""
The purpose of this script is merging the annotation JSON files produced by the xxx_to_eros
conversion script into one big annotation JSON file.

Author: @mc-cat-tty
"""
import re, json, argparse
from pathlib import Path

import humanize

SEQUENCE_GT_FILENAME_PREFIX = r"gt_poses_"
SEQUENCE_GT_FILENAME_FORMAT = SEQUENCE_GT_FILENAME_PREFIX + "{sequence_name}.json"
SEQUENCE_GT_FILENAME_RE = re.compile(SEQUENCE_GT_FILENAME_PREFIX + r"(\w+)\.json")


def get_binary_answer(prompt="Do you like events?") -> bool:
    print(prompt+" ", end="")

    ans = ''
    while ans not in ('y', 'n'):
        ans = input()
        ans = ans.lower()
    
    return ans == "y"


def parse_args():
    p = argparse.ArgumentParser(description='Merge H3.6M GT annotations into a single JSON.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--annotations_folder', type=Path, required=True,
                   help='Fold where the frames and annotations are written.')
    p.add_argument('-o', '--output-filename', default="gt_poses.json",
                   help='Name of the output JSON file, with extension.')
    p.add_argument('--dry-run', default=False, action="store_true",
                   help='Run the tool in dry-run mode. Just list the files matching the required pattern.')
    return p.parse_args()


def main(args):
    annotation_files = [
        f for f in args.annotations_folder.glob(f"{SEQUENCE_GT_FILENAME_PREFIX}*")
        if SEQUENCE_GT_FILENAME_RE.fullmatch(f.name)
    ]



    if args.dry_run:
        total_size = sum(
            map(
                lambda f: f.stat().st_size,
                annotation_files
            )
        )
            
        print(f"Found {len(annotation_files)} annotation files for a total size of {humanize.naturalsize(total_size)}. Eg.: ", end='')
        print(*[f.name for f in annotation_files[:3]], sep=', ')
        return

    output_path = args.annotations_folder / args.output_filename
    
    if output_path.is_file():
        do_overwrite_output_file = get_binary_answer("Output file already exists, overwrite it [Y/n]?")
        if not do_overwrite_output_file: return

        poses = list()
    for file in annotation_files:
        file_poses = json.loads(file.read_text())
        poses.extend(file_poses)

    tmp_output_path = args.annotations_folder / ("tmp"+args.output_filename)
    tmp_output_path.write_text(json.dumps(poses))
    tmp_output_path.replace(output_path)

    print(f"Done! saved merged file in {output_path.name}")

    do_delte_partial_annotations = get_binary_answer("Do you want to delete the annotation files [Y/n]?")
    
    if do_delte_partial_annotations:
        for f in annotation_files: f.unlink()
        print("Annotation files deleted")
    else:    
        print("Ok! Annotation files not deleted")


if __name__ == "__main__":
    args = parse_args()
    main(args)