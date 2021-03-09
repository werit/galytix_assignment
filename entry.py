from manager import AssignmentManager
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Galytix assignment.')
    parser.add_argument('-s', type=str, help='Sentence to get distance to closest phrase', required=False)

    args = parser.parse_args()

    manager = AssignmentManager()
    manager.manage(args.s)
