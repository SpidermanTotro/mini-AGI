"""Small dispatch primitives for the Greenlight training CLI."""


def dispatch(args):
    """Run the handler selected by the parser and normalize its exit status."""
    handler = getattr(args, "fn", None)
    if handler is None:
        raise ValueError("training command has no handler")
    return handler(args) or 0
