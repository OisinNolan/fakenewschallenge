from argparse import ArgumentParser

def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Training of MLP Project Model")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        nargs=1,
        choices=["AgreemFlat", "AgreemDeep", "AgreemNet"],
        help="Name of project",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        nargs='+',
        default=[5],
        help="Number of epochs to perform",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        nargs=1,
        default=[64],
        help="Batch size for training and validation loops",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        nargs=1,
        default=[0.01],
        help="Learning rate for training loop",
    )

    parser.add_argument(
        "-hd",
        "--hidden_dims",
        type=int,
        nargs="*",
        default=[1024, 512],
        help="Dimension of hidden layers in AgreemDeep and AgreemNet",
    )

    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        nargs=1,
        default=[5],
        help="Top-K similar embeddings used in AgreemDeep",
    )

    return parser

def main(): # For debugging
    args = create_parser().parse_args()
    print(args)

if __name__ == "__main__":
    main()