"""`make`-shaped wheel"""

import inspect
import re
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import file_digest, sha256
from inspect import signature
from pathlib import Path
from typing import Callable, Literal, ParamSpec, TypeVar, cast

from .lazy import lazy
from .utils import eprint, fullname

T = TypeVar("T")
P = ParamSpec("P")


DigestType = Literal["hash", "args"]

FILE = Path()
DIR = Path()


class Store:
    def __init__(self, data_path: Path):
        self.data: Path = data_path
        self.no_compute: bool = False
        self.trace_hashing: bool = False

    def hash_path(self, path: Path):
        if path is FILE:
            return b"file"
        if path is DIR:
            return b"dir"

        if path.is_relative_to(self.data):
            return self.hash_object({"internal": str(path.relative_to(self.data))})

        eprint(f"{path}: hashing external file")

        with path.open("rb", buffering=0) as f:
            digest = file_digest(f, "sha256").digest()
            return self.hash_object(
                {"external": str(path.absolute()), "digest": digest}
            )

    def hash_object(self, obj: object) -> bytes:
        h = sha256()

        def update(b: bytes | Iterable[bytes]):
            if self.trace_hashing:
                eprint(f"h.update({b})")
            if isinstance(b, bytes):
                h.update(b)
            else:
                for x in b:
                    h.update(x)

        match obj:
            case None:
                update(b"none;")
            case str():
                update(b"str;")
                update(repr(obj).encode())
            case int():
                update(b"int;")
                update(repr(obj).encode())
            case float():
                update(b"float;")
                update(repr(obj).encode())
            case bool():
                update(b"bool;")
                update(repr(obj).encode())
            case re.Pattern():
                update(b"pattern;")
                update(repr(obj).encode())
            case list():
                update(b"list;")
                obj = cast(Iterable[object], obj)
                update(map(self.hash_object, obj))
            case tuple():
                update(b"tuple;")
                obj = cast(Iterable[object], obj)
                update(map(self.hash_object, obj))
            case set():
                update(b"set;")
                obj = cast(Iterable[object], obj)
                hashed = [self.hash_object(x) for x in obj]
                update(sorted(hashed))
            case dict():
                update(b"dict;")
                obj = cast(dict[object, object], obj)
                hashed = [
                    (self.hash_object(k), self.hash_object(v)) for k, v in obj.items()
                ]
                hashed = sorted(hashed, key=lambda x: x[0])
                for k, v in hashed:
                    update(k)
                    update(v)
            case Path():
                update(b"path;")
                update(self.hash_path(obj))
            case datetime():
                update(b"datetime;")
                formatted = obj.astimezone(timezone.utc).isoformat(
                    timespec="microseconds"
                )
                update(formatted.encode())
            case _:
                raise NotImplementedError(type(obj), obj)
        return h.digest()

    def mk_output(
        self,
        is_dir: bool,
        fn: Callable[P, None],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        sig = inspect.signature(fn)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        args_digest = self.hash_object(
            {"fn": fullname(fn), "args": dict(bound_args.arguments)}
        ).hex()

        suffix = fn.__name__
        obj_name = args_digest + "-" + suffix

        out_path = self.data / obj_name

        if out_path.exists():
            eprint(f"{out_path}: exists")
            return out_path
        elif self.no_compute:
            eprint(f"{out_path}: skip: dry run")
            return out_path

        eprint(f"{out_path}: creating...")

        tmp_path = self.data / (obj_name + ".tmp")

        if tmp_path.is_dir():
            shutil.rmtree(tmp_path)
        elif tmp_path.exists():
            tmp_path.unlink()
        if is_dir:
            tmp_path.mkdir(parents=True, exist_ok=True)
        else:
            self.data.mkdir(parents=True, exist_ok=True)

        kwargs["output"] = tmp_path
        try:
            fn(*args, **kwargs)
        except BaseException as exc:
            shutil.rmtree(tmp_path, ignore_errors=True)
            raise exc

        _ = tmp_path.rename(out_path)
        return out_path

    def keep(self, fn: Callable[P, None]):
        def f(*args: P.args, **kwargs: P.kwargs) -> Path:
            assert "output" not in kwargs
            sig = signature(fn)

            param = sig.parameters["output"]
            assert param.annotation is Path
            if param.default is FILE:
                is_dir = False
            else:
                assert param.default is DIR
                is_dir = True

            return lazy(self.mk_output, is_dir, fn, *args, **kwargs)

        return f
