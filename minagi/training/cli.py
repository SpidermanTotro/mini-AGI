"""Command-line ownership for Greenlight training."""


def dispatch(args):
    """Run the handler selected by the parser and normalize its exit status."""
    handler = getattr(args, "fn", None)
    if handler is None:
        raise ValueError("training command has no handler")
    return handler(args) or 0


def add_ponder_probe_command(subparsers, handler):
    """Register the ponder-probe command on a parser collection."""
    parser = subparsers.add_parser("ponder-probe")
    parser.add_argument(
        "--ckpt", default="weights",
        help="the weights directory, or a .pt checkpoint",
    )
    parser.add_argument("--data", default="data_math_char")
    parser.add_argument("--task", default="add")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-digits", type=int, default=8)
    parser.set_defaults(fn=handler)
    return parser
