import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def main(instance_path: str, start_index: int, end_index: int):
    """
    evaluate multiple instances of the trained models
    :param instance_path: path to the hydra instances of the trained model
    :param start_index: start index of the instances
    :param end_index: end index of the instances
    """
    os.chdir(instance_path)
    for i in range(start_index, end_index):
        os.chdir(f'{i}')
        _, dirnames, _ = next(os.walk(os.getcwd(), ))
        for seed_dir in dirnames:
            if seed_dir.isdigit():
                for fname in os.listdir(Path(os.getcwd(), seed_dir)):
                    if fname != "checkpoint.pt":
                        print(fname)
                        # os.remove(Path(Path(os.getcwd(), seed_dir), fname))

        os.chdir('..')
    log.info("Process finished successfully")


if __name__ == "__main__":
    os.chdir('..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=False, default='multirun/2023-07-30/16-56-51',
                        help='Path to the trained instance, can be either relative to the project dir or absolute')

    parser.add_argument('--start', required=False, default=0, type=int, help='start index')
    parser.add_argument('--end', required=False, default=1, type=int, help='end index')
    args = parser.parse_args()

    log.setLevel(logging.INFO)
    console_handle = logging.StreamHandler(sys.stdout)
    log.addHandler(console_handle)
    main(args.path, args.start, args.end)
