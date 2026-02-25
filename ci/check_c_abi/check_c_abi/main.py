#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import gzip
import msgspec
import pathlib
import sys
import sysconfig


from check_c_abi.abi import analyze_c_abi, Abi

try:
    from termcolor import colored
except ImportError:

    def colored(text, *args, **kwargs):
        return text


def main_cli():
    # figure out the default header c path - by scanning recursively for the
    # `cuvs/core/all.h` header
    current_path = pathlib.Path(".").resolve()
    while current_path:
        if (
            current_path / "c" / "include" / "cuvs" / "core" / "all.h"
        ).is_file():
            default_c_header_path = current_path / "c" / "include"
            break
        if current_path.parent == current_path:
            default_c_header_path = None
            break
        current_path = current_path.parent

    parser = argparse.ArgumentParser(
        description="Analyze C headers for breaking ABI changes"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available subcommands"
    )

    parser_extract = subparsers.add_parser(
        "extract", help="Extract the ABI from a set of header files"
    )
    parser_extract.add_argument(
        "--output-file",
        type=str,
        help="The file to output the ABI into (default: %(default)s)",
        default="c_abi.json.gz",
    )
    parser_extract.add_argument(
        "--header-path",
        help="Path of C headers to extract the ABI from (default: %(default)s)",
        default=str(default_c_header_path),
    )
    parser_extract.add_argument(
        "--include-file",
        help="root header file to examine (default: %(default)s)",
        default="cuvs/core/all.h",
    )

    parser_analyze = subparsers.add_parser(
        "analyze", help="Analyze a set of header files for breaking changes"
    )
    parser_analyze.add_argument(
        "--abi-file",
        type=str,
        help="The extracted ABI file to compare against (default: %(default)s)",
        default="c_abi.json.gz",
    )
    parser_analyze.add_argument(
        "--header-path",
        help="Path of C headers to analyze (default: %(default)s)",
        default=str(default_c_header_path),
    )
    parser_analyze.add_argument(
        "--include-file",
        help="root header file to examine (default: %(default)s)",
        default="cuvs/core/all.h",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    header_path = pathlib.Path(args.header_path)

    # TODO: better way of specifying the dlpack header source, since missing the dlpack.h
    # header means that we all dlpack types get treated as 'int' which could be misleading
    # when looking for differences in the ABI (like if we change a field from `DLDataType` to
    # `int` without specifying the dlpack include directory, we won't know that the type has
    # changed)
    dlpack_header_path = (
        header_path.parent.parent
        / "cpp"
        / "build"
        / "_deps"
        / "dlpack-src"
        / "include"
    )
    if not dlpack_header_path.is_dir():
        # check if dlpack is installed w/ 'pip install dlpack' before giving up
        python_include_path = pathlib.Path(sysconfig.get_paths()["include"])
        dlpack_header_path = python_include_path.parent
        if not (dlpack_header_path / "dlpack" / "dlpack.h").is_file():
            raise ValueError(f"dlpack header {dlpack_header_path} not found")

    extra_clang_args = [f"-I{str(dlpack_header_path)}"]

    if args.command == "extract":
        abi = Abi.from_include_path(
            header_path, args.include_file, extra_clang_args
        )
        with open(args.output_file, "wb") as o:
            o.write(gzip.compress(msgspec.json.encode(abi)))
        print(f"wrote abi to {args.output_file}")
    elif args.command == "analyze":
        old_abi = msgspec.json.decode(
            gzip.decompress(open(args.abi_file, "rb").read()), type=Abi
        )
        new_abi = Abi.from_include_path(
            header_path, args.include_file, extra_clang_args
        )
        errors = analyze_c_abi(old_abi, new_abi)
        for error in errors:
            print(
                f"Error: {colored(error.error, attrs=['bold'])}. Symbol {colored(error.symbol, 'red')} from {error.location.filename}:{error.location.line}"
            )

        if errors:
            sys.exit(1)
        else:
            print("no breaking abi changes detected")


if __name__ == "__main__":
    main_cli()
