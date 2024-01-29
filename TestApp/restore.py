import os
import subprocess
import argparse

def parse() -> str:
    """
    Parse arguments
    :return: str
    """
    parser = argparse.ArgumentParser(description='Restore repo state')

    parser.add_argument('-input_path',
                        action="store",
                        dest="input_path",
                        help='Input Path')

    return parser.parse_args().input_path

if __name__ == "__main__":
    hash_filename = "commit.hash"
    diff_filename = "diff_patch.diff"
    REPO_PATH = "../../../"

    os.chdir(REPO_PATH)
    print("Repository path: {}".format(os.getcwd()))

    input_path = parse()

    with open(os.path.join(input_path, hash_filename), 'r') as f:
        commit_hash = f.read().rstrip('\n')
        checkout = subprocess.Popen('git checkout ' + commit_hash)
        diff = subprocess.Popen('git apply ' + os.path.join(input_path, diff_filename))


