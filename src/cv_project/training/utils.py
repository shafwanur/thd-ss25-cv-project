import sys
from typing import Any


def fullname(o: object):
    module = o.__module__
    return module + "." + o.__qualname__


def eprint(*args: Any, **kwargs: Any):
    print(*args, file=sys.stderr, **kwargs)
