from argparse import ArgumentParser

def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Training of MLP Project Model")

    parser.add_argument(
        "-p",
        "--project_name",
        type=str,
        help="Name of WandB project",
        required=True,
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=[
            "RelatedNet",
            "TopKNet",
            "AgreemNet",
        ],
        required=True,
        help="Name of model",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to perform",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and validation loops",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for training loop",
    )

    parser.add_argument(
        "-hdA",
        "--hidden_dims_A",
        type=int,
        default=1024,
        help="Dimension of INNER hidden layers in AgreemDeep and AgreemNet",
    )

    parser.add_argument(
        "-hdB",
        "--hidden_dims_B",
        type=int,
        default=512,
        help="Dimension of OUTER hidden layer in AgreemDeep and AgreemNet",
    )

    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=5,
        help="Top-K similar embeddings used in AgreemDeep",
    )

    parser.add_argument(
        "-ah",
        "--attention_heads",
        type=int,
        default=5,
        help="Number of attention heads to use in AgreemNet",
    )

    parser.add_argument(
        "-d",
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability",
    )

    parser.add_argument(
        "-cw",
        "--use_class_weights",
        type=bool,
        default=False,
        help="Weight cross entropy loss by the class counts (for 'Related' classifications)."
    )

    parser.add_argument(
        "-arc",
        "--train_with_arc",
        type=bool,
        default=False,
        help="Train using the ARC dataset."
    )

    parser.add_argument(
        "-synth",
        "--use_synth_data",
        type=bool,
        default=False,
        help="Use synthetic negative samples."
    )

    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="Name of saved trained model."
    )

    return parser

def main(): # For debugging
    args = create_parser().parse_args()
    print(args)

if __name__ == "__main__":
    main()