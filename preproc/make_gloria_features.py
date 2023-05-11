from datasets.chexpert import CheXpert
import argparse

"""
    Script to encode a dataset with GLoRIA's image encoder.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, required=True, help="Path to data.")
    parser.add_argument("--db", type=str, required=True, default="chexpert",
                        choices=["chexpert"], help="Dataset to encode.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=10)
    args = parser.parse_args()

    if args.db == "chexpert":
        dataset = CheXpert(args.root)
        dataset.build_prior(args.batch_size, args.num_workers)
    else:
        raise ValueError("Unknown dataset: %s"%args.db)

