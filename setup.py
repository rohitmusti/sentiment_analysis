import torch
import numpy as np
from args import get_setup_args

def main(args):
    print(f"pos file: {args.raw_pos}")
    print(f"neg file: {args.raw_neg}")

if __name__ == "__main__":
    args = get_setup_args()
    main(args)
    