"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    # TODO: Implement for Task 0.1.
    return x * y


def id(x: float) -> float:
    # TODO: Implement for Task 0.1.
    return x
    raise NotImplementedError("Need to implement for Task 0.1")


def add(x: float, y: float) -> float:
    return x + y
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def neg(x: float) -> float:
    return -x
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    if x < y:
        return 1.0
    else:
        return 0.0
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    if x == y:
        return 1.0
    else:
        return 0.0
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    if x > y:
        return x
    else:
        return y
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    difference = abs(x - y)
    if difference < 0.01:
        return 1.0
    else:
        return 0.0
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    if x >= 0:
        # exp = (math.e) ** (-x)
        # res = 1 / (1 + exp)
        # return res
        return (1 / (1 + exp(-x)))
    else:
        # exp = (math.e) ** (x)
        # res = exp / (1 + exp)
        # return res
        return (exp(x) / (1 + exp(x)))


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    if x > 0:
        return x
    else:
        return 0

    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return (math.exp(x))


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"

    derv = 1 / x
    return d * derv

    # TODO: Implement for Task 0.1.


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    return 1 / x
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    derv = -(1 / (x * x))
    return d * derv

    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    if x > 0:
        return d * 1
    else:
        return d * 0

    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list

    """
    ret = []

    def new_fn(element: Iterable[float]) -> Iterable[float]:
        for x in element:
            ret.append(fn(x))
        return ret

    return new_fn


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    negatives = map(neg)
    return negatives(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    def new_fn(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return new_fn


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    zip_object = zipWith(add)

    return zip_object(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def new_fn(iterable: Iterable[float]) -> float:
        answer = start
        for element in iterable:
            answer = fn(answer, element)
        return answer

    return new_fn

    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    # TODO: Implement for Task 0.3.
    total_sum = reduce(add, 0.0)
    return total_sum(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    # TODO: Implement for Task 0.3.
    total_prod = reduce(mul, 1.0)
    return total_prod(ls)
