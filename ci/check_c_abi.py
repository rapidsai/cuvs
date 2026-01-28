#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Checks for breaking changes to the C-ABI

This scripts uses the libclang python bindings to check for breaking C-ABI changes.
It works by comparing two sets of header files: the `c/include` header files for
the published ABI, as well as the `c/include` header files inside a new pull request.
Each set of header files is parsed using libclang, and the differences between the
old and new header files are examined for changes that would cause a breaking
ABI change.

Currently the following checks are made and flagged by this tool:

* Functions that have been removed from the C-ABI
* Functions that have extra parameters added
* Functions that have parameters removed
* Functions that have the type of any parameter changed
* Structs that have been removed
* Structs that have have members removed
* Structs that have the types of members changed
* Enum values that have been removed
* Enum values that have their definition changed
"""

import argparse
from dataclasses import dataclass, InitVar
import itertools
import pathlib
from typing import Optional
import sys


import clang.cindex

try:
    from termcolor import colored
except ImportError:

    def colored(text, *args, **kwargs):
        return text


@dataclass
class Abi:
    functions: dict
    structs: dict
    enums: dict

    @classmethod
    def from_include_path(cls, root, header, extra_clang_args=None):
        """Loads the Abi from a root path ('/source/cuvs/c/include') and a header file
        ("cuvs/include/all.h")
        """
        path = pathlib.Path(root).resolve()
        all_header = path / header

        index = clang.cindex.Index.create()

        args = [f"-I{str(path)}"]
        if extra_clang_args:
            args.extend(extra_clang_args)

        tu = index.parse(all_header, args=args)

        functions, structs, enums = {}, {}, {}

        # note: we could use tu.cursor.walk_preorder() here instead to recurse through the AST
        # but it is slightly slower to do so (extra 100ms or so) and for the cuvs C-ABI everything
        # is at the top level
        for child in tu.cursor.get_children():
            # ignore things like cuda headers and other files not installed in
            # in the cuvs C path
            if not child.location.file or not pathlib.Path(
                child.location.file.name
            ).is_relative_to(path):
                continue

            # break up the AST into symbol -> node for the things we care about
            if child.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                functions[child.spelling] = child
            elif child.kind == clang.cindex.CursorKind.STRUCT_DECL:
                # ignore unnamed structs (will get picked up via the typedef)
                if _is_unnamed_struct(child):
                    continue
                structs[child.spelling] = child
            elif child.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
                # check if this is a typedef to an unnamed struct, if so use the
                # typedef as the symbolname for the struct
                grandchildren = list(child.get_children())
                if len(grandchildren) == 1 and _is_unnamed_struct(
                    grandchildren[0]
                ):
                    structs[child.spelling] = grandchildren[0]
            elif child.kind == clang.cindex.CursorKind.ENUM_DECL:
                # store the enum values for checking, since we don't actually care if the
                # enum itself gets renamed for the C-ABI
                for value in child.get_children():
                    enums[value.spelling] = value

        return cls(functions, structs, enums)


@dataclass
class AbiError:
    """Holds information about an ABI breaking error"""

    error: str
    symbol: Optional[str] = None
    filename: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    cursor: InitVar[clang.cindex.Cursor | None] = None

    def __post_init__(self, cursor):
        if cursor:
            # populate fields from the
            if self.symbol is None:
                self.symbol = cursor.spelling
            if self.filename is None:
                self.filename = cursor.location.file.name
            if self.line is None:
                self.line = cursor.location.line
            if self.column is None:
                self.column = cursor.location.column


def _is_unnamed_struct(cursor):
    return (
        cursor.kind == clang.cindex.CursorKind.STRUCT_DECL
        and cursor.spelling.startswith("struct ")
        and "unnamed " in cursor.spelling
    )


def analyze_c_abi(old_path, new_path, include_file, extra_clang_args=None):
    old_abi = Abi.from_include_path(old_path, include_file, extra_clang_args)
    new_abi = Abi.from_include_path(new_path, include_file, extra_clang_args)

    # iterate over every function in the existing abi, and make sure that no functions
    # have been removed or had function arguments removed, arguments added or the type of any
    # argument changed. Note: adding new functions to the new abi is allowed
    errors = []
    for name, old_function in old_abi.functions.items():
        new_function = new_abi.functions.get(name)
        if new_function is None:
            errors.append(
                AbiError("Function has been removed", cursor=old_function)
            )
            continue

        old_result_type = old_function.result_type.spelling
        new_result_type = new_function.result_type.spelling
        if old_result_type != new_result_type:
            errors.append(
                AbiError(
                    f"Function has return type changed from '{old_result_type}' to '{new_result_type}'",
                    cursor=new_function,
                )
            )

        for old_arg, new_arg in itertools.zip_longest(
            old_function.get_children(),
            new_function.get_children(),
            fillvalue=None,
        ):
            if old_arg is None:
                errors.append(
                    AbiError(
                        f"Function has a new argument '{new_arg.type.spelling} {new_arg.spelling}'",
                        cursor=new_function,
                    )
                )

            elif new_arg is None:
                errors.append(
                    AbiError(
                        f"Function has a deleted argument '{old_arg.type.spelling} {old_arg.spelling}'",
                        cursor=old_function,
                    )
                )

            elif new_arg.type.spelling != old_arg.type.spelling:
                errors.append(
                    AbiError(
                        f"Function has a changed argument type '{old_arg.type.spelling}' to '{new_arg.type.spelling} for argument '{old_arg.spelling}'",
                        cursor=new_function,
                    )
                )

    # check to see if any existing structures have had items removed, reordered, renamed, or types
    # changed (adding new members is considered to be ok, as long as functions are initialized via
    # a create factory function)
    for name, old_struct in old_abi.structs.items():
        new_struct = new_abi.structs.get(name)
        if new_struct is None:
            errors.append(
                AbiError(
                    "Struct has been removed", symbol=name, cursor=old_struct
                )
            )
            continue

        for old_member, new_member in itertools.zip_longest(
            old_struct.get_children(),
            new_struct.get_children(),
            fillvalue=None,
        ):
            if old_member is None:
                errors.append(
                    AbiError(
                        f"Struct has a new member '{new_member.type.spelling} {new_member.spelling}'",
                        symbol=name,
                        cursor=new_member,
                    )
                )

            elif new_member is None:
                errors.append(
                    AbiError(
                        f"Struct has a deleted member '{old_member.type.spelling} {old_member.spelling}'",
                        symbol=name,
                        cursor=old_member,
                    )
                )

            elif new_member.type.spelling != old_member.type.spelling:
                errors.append(
                    AbiError(
                        f"Struct member has changed type '{old_member.type.spelling}' to '{new_member.type.spelling} for member '{old_member.spelling}'",
                        symbol=name,
                        cursor=new_member,
                    )
                )

    # check to see if enum values have been removed, or had their numeric values changed
    for name, old_enum in old_abi.enums.items():
        new_enum = new_abi.enums.get(name)
        if new_enum is None:
            errors.append(
                AbiError(
                    f"Enum value {old_enum.spelling} has been removed",
                    symbol=old_enum.lexical_parent.spelling,
                    cursor=old_enum,
                )
            )
        elif new_enum.enum_value != old_enum.enum_value:
            errors.append(
                AbiError(
                    f"Enum value {old_enum.spelling} has been changed from {old_enum.enum_value} to {new_enum.enum_value}",
                    symbol=old_enum.lexical_parent.spelling,
                    cursor=new_enum,
                )
            )

    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze C headers for breaking ABI changes"
    )
    parser.add_argument(
        "old", help="Path of existing stable C headers to use a baseline"
    )
    parser.add_argument(
        "new", help="Path of new C headers to examine for breaking ABI changes"
    )
    parser.add_argument(
        "--include_file",
        help="root header file to examine (default: %(default)s)",
        default="cuvs/core/all.h",
    )
    args = parser.parse_args()

    # TODO: better way of specifying the dlpack header source, since missing the dlpack.h
    # header means that we all dlpack types get treated as 'int' which could be misleading
    # when looking for differences in the ABI (like if we change a field from `DLDataType` to
    # `int` without specifying the dlpack include directory, we won't know that the type has
    # changed)
    dlpack_header_path = (
        pathlib.Path(args.new).parent.parent
        / "cpp"
        / "build"
        / "_deps"
        / "dlpack-src"
        / "include"
    )
    extra_clang_args = [f"-I{str(dlpack_header_path)}"]

    errors = analyze_c_abi(
        args.old, args.new, args.include_file, extra_clang_args
    )
    for error in errors:
        print(
            f"Error: {colored(error.error, attrs=['bold'])}. Symbol {colored(error.symbol, 'red')} from {error.filename}:{error.line}"
        )

    if errors:
        sys.exit(1)
