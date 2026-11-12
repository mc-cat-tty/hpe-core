"""
Copyright (C) 2021 Event-driven Perception for Robotics

Author: @mc-cat-tty

E.g.: python -m datasets.h36m.h36m_to_eros --input_folder <EH36M-FOLDER> --output_folder <EH36M-FOLDER> --write_images --write_annotation --from_scratch --ts_scaler 1 --inspect_every 200
"""

import argparse
from .h36m_to_surf import run_conversion
from datasets.utils.events_representation import ErosSurface

def get_argparser():
    p = argparse.ArgumentParser(description='Convert the YARP H3.6M events and ground truth to EROS frames '
                                            'and MoveNet annotations.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--eros_kernel', default=8, type=int)
    p.add_argument('--frame_width', default=640, type=int)
    p.add_argument('--frame_height', default=480, type=int)
    p.add_argument('--gauss_kernel', default=7, type=int)
    return p

def main():
    p = get_argparser()
    run_conversion(ErosSurface, p)


if __name__ == '__main__':
    main()
