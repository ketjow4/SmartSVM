import os
import subprocess
import argparse

def parse() -> str:
    """
    Parse arguments
    :return: str
    """
    parser = argparse.ArgumentParser(description='Saving repo state')

    parser.add_argument('-output_path',
                        action="store",
                        dest="output_path",
                        help='Output Path')

    return parser.parse_args().output_path

if __name__ == "__main__":
    hash_filename = "commit.hash"
    diff_filename = "diff_patch.diff"

    output_path = parse()

    with open(os.path.join(output_path, hash_filename), 'w') as f:
        commit_hash = subprocess.Popen('git rev-parse HEAD', stdout=f)
    with open(os.path.join(output_path,diff_filename), 'w') as f:
        diff = subprocess.Popen('git diff', stdout=f)


