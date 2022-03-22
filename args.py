from argparse import ArgumentParser

def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Training of MLP Project Model")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        nargs="?",
        choices=["AgreemFlat", "AgreemDeep", "AgreemNet"],
        help="Name of project",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        nargs="?",
        default=5,
        help="Number of epochs to perform",
    )

    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        nargs="?",
        default=64,
        help="Batch size for training and validation loops",
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        nargs="?",
        default=0.01,
        help="Learning rate for training loop",
    )

    return parser

def main(): # For debugging
    args = create_parser().parse_args()
    print(args)

if __name__ == "__main__":
    main()