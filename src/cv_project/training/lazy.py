from itertools import chain
from operator import (
    add,
    call,
    contains,
    delitem,
    eq,
    floordiv,
    ge,
    getitem,
    gt,
    le,
    lt,
    mod,
    mul,
    ne,
    setitem,
    sub,
    truediv,
)
from typing import (
    Any,
    Callable,
    ParamSpec,
    TypeVar,
    cast,
    overload,
    override,
)

T = TypeVar("T")
# R = TypeVar("R")
P = ParamSpec("P")

_evaluating = False


def lazy(
    fn: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    lzv = Lazy(LazyState(fn, *args, **kwargs))
    return _mask(lzv)


def _mask(lzv: "Lazy[T]") -> T:
    return cast(T, lzv)


def is_lazy(lzv: object):
    return hasattr(lzv, "_lazy_state")


def _state(lzv: object) -> "LazyState[Any]":
    assert is_lazy(lzv)
    return getattr(lzv, "_lazy_state")


def is_computed(lzv: object):
    return _state(lzv).is_computed


@overload
def force(lzv: "Lazy[T]") -> T: ...
@overload
def force(lzv: T) -> T: ...


def force(lzv: object) -> Any:
    if is_lazy(lzv):
        return getattr(lzv, "_lazy_force")()
    else:
        return lzv


def _force_no_deps(lzv: object) -> Any:
    assert is_lazy(lzv)
    return getattr(lzv, "_lazy_force_no_deps")()


def _unwrap(lzv: "Any") -> Any:
    if is_lazy(lzv):
        return _state(lzv).instance
    return lzv


class LazyState[T]:
    def __init__(
        self,
        # typ: type[T],
        fn: Callable[..., T],
        *args: object,
        **kwargs: object,
    ):
        self.fn: Callable[..., T] = fn
        self.args: tuple[object, ...] = args
        self.kwargs: dict[str, Any] = kwargs
        self.is_computed: bool = False
        self.computed_instance: T

        self.deps: list[object] = list(
            chain(
                filter(is_lazy, args),
                filter(is_lazy, kwargs.values()),
            )
        )

    @property
    def instance(self) -> T:
        assert self.is_computed
        return self.computed_instance


class Lazy[T]:
    def __init__(
        self,
        state: LazyState[T],
    ):
        self._lazy_state: LazyState[T] = state

    # @__class__.setter
    # def __set_class(self, type: type, /) -> None:
    #     raise NotImplemented

    def _lazy_force(self):
        s = self._lazy_state
        if s.is_computed:
            return s.instance

        new_deps: list[object] = s.deps
        deps: list[object] = []

        while new_deps:
            dep = new_deps.pop()
            deps.append(dep)
            new_deps.extend(_state(dep).deps)

        for dep in reversed(deps):
            _ = _force_no_deps(dep)

        return self._lazy_force_no_deps()

    def _lazy_force_no_deps(self):
        # print(f"_lazy_force_no_deps: {self!r}")
        s = self._lazy_state
        if s.is_computed:
            return s.instance

        args: list[Any] = []
        for v in s.args:
            args.append(_unwrap(v))

        kwargs: dict[str, Any] = {}
        for k, v in s.kwargs.items():
            kwargs[k] = _unwrap(v)

        global _evaluating
        assert not _evaluating
        try:
            _evaluating = True
            instance = s.fn(*args, **kwargs)
        finally:
            _evaluating = False
        assert not is_lazy(instance)

        s.is_computed = True
        s.computed_instance = instance
        # self.__class__ = type(instance)
        return instance

    # @override
    # def __getattribute__(self, name: str, /) -> Any:
    #     if name == "_lazy_instance":

    #     return super().__getattribute__(name)

    @overload
    def _lazy_update_any[R](self, fn: Callable[[Any], R]) -> R: ...
    @overload
    def _lazy_update_any[R, A1](self, fn: Callable[[Any, A1], R], a1: A1, /) -> R: ...
    @overload
    def _lazy_update_any[R, A1, A2](
        self, fn: Callable[[Any, A1, A2], R], a1: A1, a2: A2, /
    ) -> R: ...

    def _lazy_update_any(self, fn: Any, *args: object, **kwargs: object) -> Any:
        assert not _evaluating

        old_state = self._lazy_state
        old_self = _mask(Lazy(old_state))

        fn_r = lazy(fn, old_self, *args, **kwargs)

        def mk_new(instance: T, _r: object):
            return instance

        self._lazy_state = LazyState(mk_new, old_self, fn_r)
        return fn_r

    @overload
    def _lazy_update[R](self, fn: Callable[[T], R]) -> R: ...
    @overload
    def _lazy_update[R, A1](self, fn: Callable[[T, A1], R], a1: A1, /) -> R: ...
    @overload
    def _lazy_update[R, A1, A2](
        self, fn: Callable[[T, A1, A2], R], a1: A1, a2: A2, /
    ) -> R: ...

    def _lazy_update(self, fn: Any, *args: object) -> Any:
        return self._lazy_update_any(fn, *args)

    def __getattr__(self, name: str):
        return self._lazy_update(getattr, name)

    @override
    def __setattr__(self, name: str, value: object):
        if name == "_lazy_state":
            object.__setattr__(self, name, value)
            return

        return self._lazy_update(setattr, name, value)

    @override
    def __delattr__(self, name: str):
        return self._lazy_update(delattr, name)

    @override
    def __repr__(self):
        if not is_computed(self):
            return f"lazy {self._lazy_state.fn}"
        return f"computed {_unwrap(self)!r}"

    @override
    def __str__(self):
        return self._lazy_update(str)

    def __len__(self):
        return self._lazy_update_any(len)

    def __getitem__(self, key: str):
        return self._lazy_update_any(getitem, key)

    def __setitem__(self, key: str, value: object):
        return self._lazy_update_any(setitem, key, value)

    def __delitem__(self, key: str):
        return self._lazy_update_any(delitem, key)

    def __iter__(self):
        return self._lazy_update_any(iter)

    def __next__(self):
        return self._lazy_update_any(next)

    def __contains__(self, item: object):
        return self._lazy_update_any(contains, item)

    def __call__(self, *args: object, **kwargs: object):
        return self._lazy_update_any(call, *args, **kwargs)

    @override
    def __eq__(self, other: object):
        return self._lazy_update(eq, other)

    @override
    def __ne__(self, other: object):
        return self._lazy_update(ne, other)

    def __lt__(self, other: Any):
        return self._lazy_update_any(lt, other)

    def __le__(self, other: Any):
        return self._lazy_update_any(le, other)

    def __gt__(self, other: Any):
        return self._lazy_update_any(gt, other)

    def __ge__(self, other: Any):
        return self._lazy_update_any(ge, other)

    def __add__(self, other: object):
        return self._lazy_update(add, other)

    def __sub__(self, other: object):
        return self._lazy_update(sub, other)

    def __mul__(self, other: object):
        return self._lazy_update(mul, other)

    def __truediv__(self, other: object):
        return self._lazy_update(truediv, other)

    def __floordiv__(self, other: object):
        return self._lazy_update(floordiv, other)

    def __mod__(self, other: object):
        return self._lazy_update(mod, other)

    def __pow__(self, other: object):
        return self._lazy_update_any(pow, other)

    def __enter__(self):
        raise NotImplemented

    def __exit__(self):
        raise NotImplemented

    @override
    def __hash__(self):
        raise NotImplemented

    def __bool__(self):
        raise NotImplemented
