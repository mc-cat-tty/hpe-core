"""
Copyright (C) 2021 Event-driven Perception for Robotics

Author: @mc-cat-tty

E.g.: python -m datasets.h36m.h36m_to_scarf --input_folder <EH36M-FOLDER> --output_folder <EH36M-FOLDER> --write_images --write_annotation --from_scratch --ts_scaler 1 --inspect_every 200
"""

import argparse
from .h36m_to_surf import run_conversion
from datasets.utils.events_representation import ScarfSurface

def get_argparser():
    p = argparse.ArgumentParser(description='Convert the YARP H3.6M events and ground truth to SCARF frames '
                                            'and MoveNet annotations.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--frame_width', default=640)
    p.add_argument('--frame_height', default=480)
    p.add_argument('--block_size', default=14, help="Receptive field size; square shape")
    p.add_argument('--alpha', default=1.0, help="Events accumulation factor")
    p.add_argument('--C', default=0.3, help="Intensity")
    return p

def main():
    p = get_argparser()
    run_conversion(ScarfSurface, p)


if __name__ == '__main__':
    main()
