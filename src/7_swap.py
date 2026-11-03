"""Refine completed A/B configurations inside the same 13-site pool."""
from shared.session import run_step
from shared.swaps import run


def main(ctx):run(ctx)


if __name__=='__main__':raise SystemExit(run_step('7_swap',main))
