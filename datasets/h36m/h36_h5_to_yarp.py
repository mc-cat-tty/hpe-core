#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021 Event-driven Perception for Robotics
Author: Franco Di Pietro
Refactor by: @mc-cat-tty

LICENSE GOES HERE
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import h5py
import numpy as np
from bimvee import exportIitYarp
from tqdm import tqdm


def load_events(h5_file):
    with h5py.File(h5_file, 'r') as hf:
        return np.array(hf['events'][:])


def build_container(data, export_path):
    return {
        'info': {'filePathOrName': str(export_path)},
        'data': {'ch0': {'dvs': {
            'ts': (data[:, 0] - data[0, 0]) * 1e-6,
            'x': data[:, 1],
            'y': data[:, 2],
            'pol': data[:, 3].astype(bool),
        }}},
    }


def parse_args():
    p = argparse.ArgumentParser(description='Convert the hdf5 events produced by v2e to the YARP format.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input_folder', type=Path, required=True,
                   help='folder with one sub-folder per sequence, each containing the v2e hdf5 file')
    p.add_argument('--output_folder', type=Path, required=True, help='folder where the YARP datasets are written')
    p.add_argument('--h5_name', default='events.h5', help='name of the hdf5 file inside each sequence folder')
    p.add_argument('--sequences', nargs='+', default=None,
                   help='sequence folders to process (default: all the sub-folders of --input_folder)')
    p.add_argument('--skip_existing', action='store_true', help='skip sequences already exported')
    return p.parse_args()

def process_sequence(args, seq: str):
    """Export one sequence. Returns a message to report, or None on success (workers must not print)."""
    h5_files = list((args.input_folder / seq).rglob(args.h5_name))
    if len(h5_files) > 1:
        return f"Skipping {seq}: too many .h5 files: {h5_files}"
    if len(h5_files) < 1:
        return f"Skipping {seq}: .h5 file not found"

    h5_file = h5_files[0]
    export_path = args.output_folder / seq
    if args.skip_existing and (export_path / 'ch0dvs').is_dir():
        return f"Skipping {seq}: already processed"

    data = load_events(h5_file)
    if data.ndim != 2 or len(data) == 0:
        return f"Skipping {seq}: no events"
    # exportIitYarp draws its own progress bar, which would clash with the overall one
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        exportIitYarp.exportIitYarp(build_container(data, export_path), exportFilePath=str(export_path),
                                    protectedWrite=False)
    return f"Done {seq}"


def main():
    args = parse_args()
    if not args.input_folder.is_dir():
        sys.exit(f"Input folder {args.input_folder} does not exist")

    sequences = args.sequences
    if sequences is None:
        out_abs = args.output_folder.resolve()
        sequences = sorted(d.name for d in args.input_folder.iterdir()
                           if d.is_dir() and d.resolve() != out_abs)

    workers = int(cpu_count()*1.5)
    with Pool(workers) as p:
        for msg in tqdm(p.imap(partial(process_sequence, args), sequences), total=len(sequences)):
            if msg:
                tqdm.write(msg)


if __name__ == '__main__':
    main()
