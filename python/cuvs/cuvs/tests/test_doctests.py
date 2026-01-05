#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import doctest
import inspect
import io
import sys
from pathlib import Path

import pytest

import cuvs.cluster
import cuvs.distance
import cuvs.neighbors
import cuvs.preprocessing.quantize

# Code adapted from https://github.com/rapidsai/cudf/blob/branch-23.02/python/cudf/cudf/tests/test_doctests.py  # noqa


def _name_in_all(parent, name):
    return name in getattr(parent, "__all__", [])


def _is_public_name(parent, name):
    return not name.startswith("_")


def _find_doctests_in_obj(obj, finder=None, criteria=None):
    """Find all doctests in an object.

    Parameters
    ----------
    obj : module or class
        The object to search for docstring examples.
    finder : doctest.DocTestFinder, optional
        The DocTestFinder object to use. If not provided, a DocTestFinder is
        constructed.
    criteria : callable, optional
        Callable indicating whether to recurse over members of the provided
        object. If not provided, names not defined in the object's ``__all__``
        property are ignored.

    Yields
    ------
    doctest.DocTest
        The next doctest found in the object.
    """
    if finder is None:
        finder = doctest.DocTestFinder()
    if criteria is None:
        criteria = _name_in_all
    for docstring in finder.find(obj):
        if docstring.examples:
            yield docstring
    for name, member in inspect.getmembers(obj):
        # Only recurse over members matching the criteria
        if not criteria(obj, name):
            continue
        # Recurse over the public API of modules (objects defined in the
        # module's __all__)
        if inspect.ismodule(member):
            yield from _find_doctests_in_obj(
                member, finder, criteria=_name_in_all
            )
        # Recurse over the public API of classes (attributes not prefixed with
        # an underscore)
        if inspect.isclass(member):
            yield from _find_doctests_in_obj(
                member, finder, criteria=_is_public_name
            )

        # doctest finder seems to dislike cython functions, since
        # `inspect.isfunction` doesn't return true for them. hack around this
        if callable(member) and not inspect.isfunction(member):
            for docstring in finder.find(member):
                if docstring.examples:
                    yield docstring


# since the root pylibraft module doesn't import submodules (or define an
# __all__) we are explicitly adding all the submodules we want to run
# doctests for here
DOC_STRINGS = list(_find_doctests_in_obj(cuvs.neighbors))
DOC_STRINGS.extend(_find_doctests_in_obj(cuvs.neighbors.cagra))
DOC_STRINGS.extend(_find_doctests_in_obj(cuvs.neighbors.brute_force))
DOC_STRINGS.extend(_find_doctests_in_obj(cuvs.neighbors.ivf_flat))
DOC_STRINGS.extend(_find_doctests_in_obj(cuvs.common))
DOC_STRINGS.extend(_find_doctests_in_obj(cuvs.cluster))
DOC_STRINGS.extend(_find_doctests_in_obj(cuvs.distance))
DOC_STRINGS.extend(_find_doctests_in_obj(cuvs.preprocessing.quantize))


def _test_name_from_docstring(docstring):
    filename = Path(docstring.filename).name.split(".")[0]
    return f"{filename}:{docstring.name}"


class VerboseDocTestRunner(doctest.DocTestRunner):
    """A DocTestRunner that prints each example before executing it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_test_name = None

    def run(self, test, compileflags=None, out=None, clear_globs=True):
        self.current_test_name = test.name
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"DOCTEST: {test.name}", file=sys.stderr)
        print(f"FILE: {test.filename}", file=sys.stderr)
        print(f"EXAMPLES: {len(test.examples)}", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
        sys.stderr.flush()
        return super().run(test, compileflags, out, clear_globs)

    def report_start(self, out, test, example):
        """Called before each example is run."""
        idx = test.examples.index(example)
        source = example.source.strip()
        # Truncate long lines for readability
        if len(source) > 100:
            source = source[:97] + "..."
        print(f"  [{idx:2d}] Executing: {source}", file=sys.stderr)
        sys.stderr.flush()
        return super().report_start(out, test, example)

    def report_success(self, out, test, example, got):
        """Called when an example succeeds."""
        idx = test.examples.index(example)
        print(f"  [{idx:2d}] OK", file=sys.stderr)
        sys.stderr.flush()
        return super().report_success(out, test, example, got)

    def report_failure(self, out, test, example, got):
        """Called when an example fails."""
        idx = test.examples.index(example)
        print(f"  [{idx:2d}] FAILED!", file=sys.stderr)
        print(f"       Expected: {example.want!r}", file=sys.stderr)
        print(f"       Got: {got!r}", file=sys.stderr)
        sys.stderr.flush()
        return super().report_failure(out, test, example, got)

    def report_unexpected_exception(self, out, test, example, exc_info):
        """Called when an example raises an unexpected exception."""
        idx = test.examples.index(example)
        print(f"  [{idx:2d}] EXCEPTION!", file=sys.stderr)
        print(f"       {exc_info[0].__name__}: {exc_info[1]}", file=sys.stderr)
        sys.stderr.flush()
        return super().report_unexpected_exception(
            out, test, example, exc_info
        )


@pytest.mark.parametrize(
    "docstring", DOC_STRINGS, ids=_test_name_from_docstring
)
def test_docstring(docstring):
    # We ignore differences in whitespace in the doctest output, and enable
    # the use of an ellipsis "..." to match any string in the doctest
    # output. An ellipsis is useful for, e.g., memory addresses or
    # imprecise floating point values.
    optionflags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

    # Use verbose runner to see exactly which example crashes
    runner = VerboseDocTestRunner(optionflags=optionflags, verbose=True)

    # Capture stdout and include failing outputs in the traceback.
    doctest_stdout = io.StringIO()
    with contextlib.redirect_stdout(doctest_stdout):
        runner.run(docstring)
        results = runner.summarize()

    print(
        f"\nRESULT: {results.attempted} attempted, {results.failed} failed",
        file=sys.stderr,
    )
    sys.stderr.flush()

    assert not results.failed, (
        f"{results.failed} of {results.attempted} doctests failed for "
        f"{docstring.name}:\n{doctest_stdout.getvalue()}"
    )
