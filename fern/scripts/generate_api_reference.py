#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Generate Fern API reference pages from cuVS source code.

The generator intentionally does not import ``cuvs`` or compile native code.
It reads the original C/C++ headers, Cython/Python sources, and Java sources
directly so the Fern reference can be refreshed in lightweight docs jobs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import ast
import re
import shutil
import textwrap
from pathlib import Path
from typing import Iterable


REPO_DIR = Path(__file__).resolve().parents[2]
FERN_PAGES = REPO_DIR / "fern" / "pages"
PYTHON_DIR = REPO_DIR / "python" / "cuvs" / "cuvs"
RUST_SOURCE_DIR = REPO_DIR / "rust" / "cuvs" / "src"
GO_SOURCE_DIR = REPO_DIR / "go"
NATIVE_HEADER_DIRS = [REPO_DIR / "c" / "include", REPO_DIR / "cpp" / "include"]
JAVA_SOURCE_DIRS = [
    REPO_DIR / "java" / "cuvs-java" / "src" / "main" / "java",
    REPO_DIR / "java" / "cuvs-java" / "src" / "main" / "java22",
]
API_NAV_SECTIONS = [
    ("C API Documentation", "c_api", "c-api-documentation", "C API", "c-api"),
    (
        "Cpp API Documentation",
        "cpp_api",
        "cpp-api-documentation",
        "C++ API",
        "cpp-api",
    ),
    (
        "Python API Documentation",
        "python_api",
        "python-api-documentation",
        "Python API",
        "python-api",
    ),
    (
        "Java API Documentation",
        "java_api",
        "java-api-documentation",
        "Java API",
        "java-api",
    ),
    (
        "Rust API Documentation",
        "rust_api",
        "rust-api-documentation",
        "Rust API",
        "rust-api",
    ),
    (
        "Go API Documentation",
        "go_api",
        "go-api-documentation",
        "Go API",
        "go-api",
    ),
]

COMMENT_RE = re.compile(r"/\*\*.*?\*/|(?:///[^\n]*(?:\n|$))+", re.DOTALL)
DOXYGEN_COMMAND_RE = re.compile(r"[@\\](\w+)\b")
DOXYGEN_LIST_ITEM_RE = re.compile(r"^(?:-\s+|\d+\.\s+)")
DOXYGEN_FIELD_LIST_ITEM_RE = re.compile(
    r"^(?:-\s+)?`?(?P<name>[A-Za-z_]\w*)`?\s*:\s*(?P<description>.*)"
)
PUBLIC_JAVA_TYPE_RE = re.compile(
    r"\bpublic\s+(?:abstract\s+|final\s+|sealed\s+|non-sealed\s+)?"
    r"(?P<kind>class|interface|enum|record)\s+(?P<name>[A-Za-z_]\w*)"
)
SPHINX_ROLE_RE = re.compile(
    r"(?<!\w):?(?:(?:[A-Za-z][\w-]*):)?[A-Za-z][\w-]*:`(?P<target>[^`]+)`"
)
MATH_PLACEHOLDER_RE = re.compile(r"@@FERN_MATH_([0-9a-f]+)@@")
CPP_COMPOUND_RE = re.compile(
    r"^\s*(?:typedef\s+)?(?:struct|class|enum(?:\s+class)?)\b"
)


@dataclass
class DoxygenParam:
    name: str
    description: str = ""
    direction: str = ""


@dataclass
class FunctionParam:
    name: str
    c_type: str
    default: str = ""


@dataclass
class DoxygenEntry:
    kind: str
    name: str
    signature: str
    summary: str = ""
    details: list[str] = field(default_factory=list)
    params: list[DoxygenParam] = field(default_factory=list)
    tparams: list[DoxygenParam] = field(default_factory=list)
    returns: str = ""
    source: str = ""
    line: int = 0
    code_blocks: list[tuple[str, list[str]]] = field(default_factory=list)


@dataclass
class DoxygenGroup:
    name: str
    title: str = ""
    entries: list[DoxygenEntry] = field(default_factory=list)


@dataclass
class NativePage:
    slug: str
    title: str
    source: str
    groups: list[DoxygenGroup] = field(default_factory=list)


@dataclass
class PythonSymbol:
    name: str
    kind: str
    signature: str
    doc: str
    source: str
    line: int
    decorators: list[str] = field(default_factory=list)
    members: list["PythonSymbol"] = field(default_factory=list)
    value: str = ""


@dataclass
class PythonPage:
    module: str
    title: str
    slug: str
    symbols: list[PythonSymbol]


@dataclass
class JavaDoc:
    summary: str = ""
    params: list[DoxygenParam] = field(default_factory=list)
    returns: str = ""
    throws: list[DoxygenParam] = field(default_factory=list)


@dataclass
class JavaMember:
    name: str
    signature: str
    doc: JavaDoc
    line: int


@dataclass
class JavaClass:
    package: str
    name: str
    kind: str
    signature: str
    doc: JavaDoc
    source: str
    line: int
    members: list[JavaMember] = field(default_factory=list)


@dataclass
class RustItem:
    name: str
    kind: str
    signature: str
    doc: str
    source: str
    line: int
    attributes: list[str] = field(default_factory=list)
    members: list["RustItem"] = field(default_factory=list)


@dataclass
class RustPage:
    module: str
    title: str
    slug: str
    source: str
    module_doc: str
    items: list[RustItem]


@dataclass
class GoItem:
    name: str
    kind: str
    signature: str
    doc: str
    source: str
    line: int
    receiver: str = ""


@dataclass
class GoPage:
    package: str
    title: str
    slug: str
    sources: list[str]
    items: list[GoItem]


class DoxygenHeaderIndex:
    def __init__(self) -> None:
        self.groups: dict[str, DoxygenGroup] = {}
        self.structs: dict[str, DoxygenEntry] = {}
        self.enums: dict[str, DoxygenEntry] = {}
        self.short_structs: dict[str, DoxygenEntry] = {}
        self.short_enums: dict[str, DoxygenEntry] = {}

    @classmethod
    def build(cls, header_dirs: Iterable[Path]) -> "DoxygenHeaderIndex":
        index = cls()
        for root in header_dirs:
            for path in sorted(root.rglob("*")):
                if path.suffix in {".h", ".hpp", ".cuh"}:
                    index._parse_header(path)
        return index

    def _parse_header(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        rel_path = path.relative_to(REPO_DIR).as_posix()
        current_groups: list[str] = []

        for match in COMMENT_RE.finditer(text):
            raw_comment = match.group(0)
            if raw_comment.lstrip().startswith("///<"):
                continue

            comment = clean_doxygen_comment(raw_comment)
            if not comment.strip():
                continue

            group_command = find_doxygen_command(comment, "defgroup")
            group_kind = "defgroup"
            if group_command is None:
                group_command = find_doxygen_command(comment, "addtogroup")
                group_kind = "addtogroup"

            group_name = ""
            if group_command:
                group_name, group_title = split_command_payload(group_command)
                group = self.groups.setdefault(
                    group_name, DoxygenGroup(group_name)
                )
                if group_kind == "defgroup" and group_title:
                    group.title = group_title

            closes_group = bool(re.search(r"(^|\s)[@\\]}", comment))
            opens_group = bool(re.search(r"(^|\s)[@\\]{", comment))
            explicit_groups = re.findall(r"[@\\]ingroup\s+([\w:.-]+)", comment)
            candidate_groups = list(
                dict.fromkeys([*explicit_groups, *current_groups])
            )

            if not group_command and not closes_group:
                declaration, decl_line = read_declaration_after(
                    text, match.end()
                )
                if declaration:
                    entry = parse_doxygen_entry(
                        comment, declaration, rel_path, decl_line
                    )
                    if is_namespace_entry(entry):
                        continue
                    if entry.kind == "member" and not is_type_alias_signature(
                        entry.signature
                    ):
                        continue
                    self._qualify_function(entry, text[: match.start()])
                    target_groups = candidate_groups
                    if not target_groups and is_native_type_entry(entry):
                        target_groups = [synthetic_native_group_name(rel_path)]
                    for candidate in target_groups:
                        self.groups.setdefault(
                            candidate,
                            DoxygenGroup(
                                candidate,
                                "Types"
                                if is_synthetic_native_group(candidate)
                                else "",
                            ),
                        ).entries.append(entry)
                    self._index_compound(entry, text[: match.start()])

            if group_command and opens_group and group_name:
                current_groups.append(group_name)
                if not next_non_whitespace_is_comment(text, match.end()):
                    declaration, decl_line = read_declaration_after(
                        text, match.end()
                    )
                    if declaration:
                        entry = parse_doxygen_entry(
                            comment, declaration, rel_path, decl_line
                        )
                        if is_namespace_entry(entry):
                            continue
                        self._qualify_function(entry, text[: match.start()])
                        if not entry.summary and self.groups[group_name].title:
                            entry.summary = self.groups[group_name].title
                        self.groups[group_name].entries.append(entry)
                        self._index_compound(entry, text[: match.start()])
            if closes_group and current_groups:
                current_groups.pop()

    def _index_compound(self, entry: DoxygenEntry, prefix: str) -> None:
        if (
            is_type_alias_signature(entry.signature)
            and "{" not in entry.signature
        ):
            return
        namespace, class_scope = infer_cpp_scope(prefix)
        qualifiers = [namespace] if namespace else []
        qualifiers.extend(class_scope)
        struct_name = parse_struct_name(entry.signature)
        if struct_name:
            fq_name = "::".join([*qualifiers, struct_name])
            entry.kind = "struct"
            entry.name = fq_name
            self.structs[fq_name] = entry
            self.short_structs.setdefault(struct_name, entry)

        enum_name = parse_enum_name(entry.signature)
        if enum_name:
            fq_name = enum_name
            if qualifiers and "::" not in enum_name:
                fq_name = "::".join([*qualifiers, enum_name])
            entry.kind = "enum"
            entry.name = fq_name
            self.enums[fq_name] = entry
            self.short_enums.setdefault(enum_name.split("::")[-1], entry)

    def _qualify_function(self, entry: DoxygenEntry, prefix: str) -> None:
        if entry.kind != "function" or "::" in entry.name:
            return
        namespace, class_scope = infer_cpp_scope(prefix)
        qualifiers = [namespace] if namespace else []
        qualifiers.extend(class_scope)
        if qualifiers:
            entry.name = "::".join([*qualifiers, entry.name])


def main() -> int:
    remove_old_api_pages()
    native_index = DoxygenHeaderIndex.build(NATIVE_HEADER_DIRS)
    native_pages_by_api = {
        "c": collect_native_pages(native_index, "c"),
        "cpp": collect_native_pages(native_index, "cpp"),
    }
    global_links, page_links = build_native_symbol_links(native_pages_by_api)
    generate_native_api_pages(
        native_pages_by_api["c"], "c", global_links, page_links
    )
    generate_native_api_pages(
        native_pages_by_api["cpp"], "cpp", global_links, page_links
    )
    generate_python_api_pages()
    generate_java_api_pages()
    generate_rust_api_pages()
    generate_go_api_pages()
    update_api_navigation()
    return 0


def remove_old_api_pages() -> None:
    for path in [
        FERN_PAGES / "c_api",
        FERN_PAGES / "cpp_api",
        FERN_PAGES / "python_api",
        FERN_PAGES / "java_api",
        FERN_PAGES / "rust_api",
        FERN_PAGES / "go_api",
    ]:
        if path.exists():
            shutil.rmtree(path)
    for path in [
        FERN_PAGES / "c_api.md",
        FERN_PAGES / "cpp_api.md",
        FERN_PAGES / "python_api.md",
        FERN_PAGES / "rust_api.md",
        FERN_PAGES / "go_api.md",
    ]:
        if path.exists():
            path.unlink()


def is_type_alias_signature(signature: str) -> bool:
    stripped = signature.strip()
    return stripped.startswith("typedef ") or stripped.startswith("using ")


def is_native_type_entry(entry: DoxygenEntry) -> bool:
    return entry.kind in {"enum", "struct"} or is_type_alias_signature(
        entry.signature
    )


def synthetic_native_group_name(source: str) -> str:
    return f"__types__:{source}"


def is_synthetic_native_group(group_name: str) -> bool:
    return group_name.startswith("__types__:")


def generate_native_api_pages(
    pages: list[NativePage],
    api: str,
    global_links: dict[str, dict[str, str]],
    page_links: dict[tuple[str, str], dict[str, str]],
) -> None:
    out_dir = FERN_PAGES / f"{api}_api"
    out_dir.mkdir(parents=True, exist_ok=True)
    title = "C API Documentation" if api == "c" else "C++ API Documentation"
    directory = f"{api}_api"

    index_lines = [
        f"# {title}",
        "",
        "These pages are generated from the documented public headers in the cuVS source tree.",
        "",
    ]
    for page in pages:
        index_lines.append(
            f"- [{page.title}]({api_doc_url(directory, page.slug)})"
        )
    write_page(out_dir / "index.md", index_lines)

    language = "c" if api == "c" else "cpp"
    for page in pages:
        lines = [
            *api_frontmatter(api_page_route(directory, page.slug)),
            f"# {page.title}",
            "",
            f"_Source header: `{page.source}`_",
            "",
        ]
        page_headings: set[str] = set()
        symbol_links = {
            **global_links.get(api, {}),
            **page_links.get((directory, page.slug), {}),
        }
        for group in page.groups:
            lines.extend(
                render_native_group(
                    group, language, page_headings, symbol_links
                )
            )
            lines.append("")
        write_page(
            out_dir / f"{api_page_route(directory, page.slug)}.md", lines
        )


def collect_native_pages(
    index: DoxygenHeaderIndex, api: str
) -> list[NativePage]:
    prefix = "c/include/" if api == "c" else "cpp/include/"
    pages: dict[str, NativePage] = {}

    for group in index.groups.values():
        entries = [
            entry
            for entry in group.entries
            if entry.source.startswith(prefix)
            and not is_detail_namespace_entry(entry)
            and not is_namespace_entry(entry)
        ]
        if not entries:
            continue
        source = sorted({entry.source for entry in entries})[0]
        slug = native_page_slug(source)
        page = pages.setdefault(
            slug, NativePage(slug, native_page_title(source), source)
        )
        copied = DoxygenGroup(
            group.name,
            group.title,
            sorted(entries, key=lambda entry: entry.line),
        )
        page.groups.append(copied)

    ordered = sorted(
        pages.values(), key=lambda page: (page.source, page.title)
    )
    for page in ordered:
        page.groups.sort(
            key=lambda group: min(
                (entry.line for entry in group.entries), default=0
            )
        )
    return ordered


def build_native_symbol_links(
    pages_by_api: dict[str, list[NativePage]],
) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    global_links: dict[str, dict[str, str]] = defaultdict(dict)
    page_links: dict[tuple[str, str], dict[str, str]] = {}

    for api, pages in pages_by_api.items():
        directory = f"{api}_api"
        for page in pages:
            entries = [
                entry
                for group in page.groups
                for entry in group.entries
                if is_linkable_native_type(entry)
            ]
            short_counts = Counter(
                short_symbol_name(entry.name) for entry in entries
            )
            local_links: dict[str, str] = {}
            for entry in entries:
                url = f"{api_doc_url(directory, page.slug)}#{symbol_anchor(entry.name)}"
                for symbol in native_link_symbols(entry, api):
                    global_links[api].setdefault(symbol, url)
                    local_links[symbol] = url
                short_name = short_symbol_name(entry.name)
                if short_counts[short_name] == 1:
                    local_links[short_name] = url
            page_links[(directory, page.slug)] = local_links

    return dict(global_links), page_links


def is_linkable_native_type(entry: DoxygenEntry) -> bool:
    return is_native_type_entry(entry)


def native_link_symbols(entry: DoxygenEntry, api: str) -> list[str]:
    symbols = [entry.name]
    if api == "c" and entry.kind == "struct" and not entry.name.endswith("_t"):
        symbols.append(f"{entry.name}_t")
    return symbols


def short_symbol_name(name: str) -> str:
    return name.split("::")[-1]


def is_detail_namespace_entry(entry: DoxygenEntry) -> bool:
    return "detail" in entry.name.split("::")


def is_namespace_entry(entry: DoxygenEntry) -> bool:
    return bool(re.match(r"^\s*(?:inline\s+)?namespace\b", entry.signature))


def render_native_group(
    group: DoxygenGroup,
    language: str,
    page_headings: set[str] | None = None,
    symbol_links: dict[str, str] | None = None,
) -> list[str]:
    lines = [f"## {heading_text(group.title or group.name)}", ""]

    if page_headings is None:
        page_headings = set()
    if symbol_links is None:
        symbol_links = {}
    for entry in group.entries:
        if entry.kind == "function":
            include_heading = entry.name not in page_headings
            page_headings.add(entry.name)
            lines.extend(
                render_native_function(
                    entry,
                    language,
                    include_heading=include_heading,
                    symbol_links=symbol_links,
                )
            )
        elif entry.kind in {"struct", "enum"}:
            page_headings.add(entry.name)
            lines.extend(render_native_compound(entry, language, symbol_links))
        else:
            page_headings.add(entry.name)
            lines.extend(render_native_member(entry, language))
        lines.append("")
    return trim_blank_lines(lines)


def render_native_function(
    entry: DoxygenEntry,
    language: str,
    include_heading: bool = True,
    symbol_links: dict[str, str] | None = None,
) -> list[str]:
    if symbol_links is None:
        symbol_links = {}
    signature = normalize_signature(entry.signature)
    if include_heading:
        lines = [
            symbol_anchor_line(entry.name),
            f"### {heading_text(entry.name)}",
            "",
        ]
    else:
        lines = [f"**Additional overload:** `{escape_code(entry.name)}`", ""]
    if entry.summary:
        lines.extend([escape_text(entry.summary), ""])
    lines.extend([f"```{language}", signature, "```", ""])

    if entry.details:
        lines.extend(render_doxygen_details(entry.details))
        lines.append("")

    if entry.tparams:
        lines.extend(["**Template Parameters**", ""])
        lines.extend(
            render_param_table(
                [
                    {
                        "name": param.name,
                        "type": "",
                        "description": param.description,
                    }
                    for param in entry.tparams
                ],
                include_direction=False,
            )
        )
        lines.append("")

    params = parse_function_params(signature)
    documented_params = {param.name: param for param in entry.params}
    rows = []
    for param in params:
        documented = documented_params.get(param.name)
        rows.append(
            {
                "name": param.name,
                "direction": documented.direction if documented else "",
                "type": param.c_type,
                "description": documented.description if documented else "",
                "default": param.default,
            }
        )
    if rows:
        lines.extend(["**Parameters**", ""])
        lines.extend(
            render_param_table(
                rows, include_direction=True, symbol_links=symbol_links
            )
        )
        lines.append("")

    return_type = parse_return_type(signature)
    if return_type:
        lines.extend(
            [
                "**Returns**",
                "",
                render_type_reference(return_type, symbol_links),
            ]
        )
        if entry.returns:
            lines.extend(["", escape_text(entry.returns)])
        lines.append("")

    lines.extend([source_line(entry), ""])
    return trim_blank_lines(lines)


def render_native_compound(
    entry: DoxygenEntry,
    language: str,
    symbol_links: dict[str, str] | None = None,
) -> list[str]:
    if symbol_links is None:
        symbol_links = {}
    lines = [
        symbol_anchor_line(entry.name),
        f"### {heading_text(entry.name)}",
        "",
    ]
    if entry.summary:
        lines.extend([escape_text(entry.summary), ""])

    members: list[DoxygenEntry] = []
    values: list[dict[str, str]] = []
    field_descriptions: dict[str, str] = {}
    details = entry.details
    if entry.kind == "enum":
        values = parse_enum_values(entry.signature)
        field_descriptions, details = extract_field_descriptions(
            entry.details, {value["name"] for value in values}
        )
    elif not is_class_signature(entry.signature):
        members = parse_struct_members(entry)
        field_descriptions, details = extract_field_descriptions(
            entry.details, {member.name for member in members}
        )

    if details:
        lines.extend(render_doxygen_details(details))
        lines.append("")
    lines.extend(
        [
            f"```{language}",
            compact_compound_signature(entry.signature),
            "```",
            "",
        ]
    )

    if entry.kind == "enum":
        if values:
            if field_descriptions:
                lines.extend(
                    [
                        "**Values**",
                        "",
                        "| Name | Value | Description |",
                        "| --- | --- | --- |",
                    ]
                )
            else:
                lines.extend(
                    ["**Values**", "", "| Name | Value |", "| --- | --- |"]
                )
            for value in values:
                name = escape_code(value["name"])
                enum_value = escape_code(value.get("value", ""))
                if field_descriptions:
                    description = render_table_description(
                        field_descriptions.get(value["name"], "")
                    )
                    lines.append(
                        f"| `{name}` | `{enum_value}` | {description} |"
                    )
                else:
                    lines.append(f"| `{name}` | `{enum_value}` |")
            lines.append("")
    else:
        if members:
            lines.extend(["**Fields**", ""])
            rows = []
            for member in members:
                rows.append(
                    {
                        "name": member.name,
                        "type": member_c_type(member),
                        "description": field_descriptions.get(
                            member.name, member_description(member)
                        ),
                    }
                )
            lines.extend(
                render_param_table(
                    rows, include_direction=False, symbol_links=symbol_links
                )
            )
            lines.append("")

    lines.extend([source_line(entry), ""])
    return trim_blank_lines(lines)


def render_native_member(entry: DoxygenEntry, language: str) -> list[str]:
    lines = [
        symbol_anchor_line(entry.name),
        f"### {heading_text(entry.name)}",
        "",
    ]
    if entry.summary:
        lines.extend([escape_text(entry.summary), ""])
    lines.extend(
        [f"```{language}", normalize_signature(entry.signature), "```", ""]
    )
    lines.extend([source_line(entry), ""])
    return trim_blank_lines(lines)


def generate_python_api_pages() -> None:
    out_dir = FERN_PAGES / "python_api"
    out_dir.mkdir(parents=True, exist_ok=True)
    symbol_index = build_python_symbol_index()
    pages = collect_python_pages(symbol_index)

    index_lines = [
        "# Python API Documentation",
        "",
        "These pages are generated from the Python and Cython sources under `python/cuvs/cuvs`.",
        "",
    ]
    for group, group_pages in group_python_pages(pages).items():
        index_lines.extend([f"## {group}", ""])
        for page in group_pages:
            index_lines.append(
                f"- [{page.title}]({api_doc_url('python_api', page.slug)})"
            )
        index_lines.append("")
    write_page(out_dir / "index.md", index_lines)

    for page in pages:
        lines = [
            *api_frontmatter(api_page_route("python_api", page.slug)),
            f"# {page.title}",
            "",
            f"_Python module: `{page.module}`_",
            "",
        ]
        for symbol in page.symbols:
            lines.extend(render_python_symbol(symbol))
            lines.append("")
        write_page(
            out_dir / f"{api_page_route('python_api', page.slug)}.md", lines
        )


def build_python_symbol_index() -> dict[str, dict[str, PythonSymbol]]:
    index: dict[str, dict[str, PythonSymbol]] = defaultdict(dict)
    for path in sorted(PYTHON_DIR.rglob("*")):
        if path.suffix not in {".py", ".pyx"}:
            continue
        if "/tests/" in path.as_posix():
            continue
        module = python_module_name(path)
        for symbol in parse_python_source(path):
            index[module][symbol.name] = symbol
    return index


def collect_python_pages(
    symbol_index: dict[str, dict[str, PythonSymbol]],
) -> list[PythonPage]:
    pages: list[PythonPage] = []
    for init_path in sorted(PYTHON_DIR.rglob("__init__.py")):
        module = python_module_name(init_path)
        exports = read_python_exports(init_path)
        if not exports:
            continue
        symbols = []
        for name in exports:
            symbol = find_python_symbol(module, name, symbol_index)
            if symbol is not None:
                symbols.append(symbol)
        if not symbols:
            continue
        pages.append(
            PythonPage(
                module, python_title(module), python_slug(module), symbols
            )
        )

    pages.sort(key=lambda page: (python_group(page.module), page.title))
    return pages


def find_python_symbol(
    module: str,
    name: str,
    symbol_index: dict[str, dict[str, PythonSymbol]],
) -> PythonSymbol | None:
    candidates = [
        candidate
        for candidate in symbol_index
        if candidate == module or candidate.startswith(f"{module}.")
    ]
    candidates.sort(key=lambda candidate: (candidate.count("."), candidate))
    for candidate in candidates:
        if name in symbol_index[candidate]:
            return symbol_index[candidate][name]
    return None


def parse_python_source(path: Path) -> list[PythonSymbol]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    symbols: list[PythonSymbol] = []
    pending_decorators: list[str] = []
    idx = 0
    while idx < len(lines):
        raw_line = lines[idx]
        stripped = raw_line.strip()
        indent = indentation(raw_line)
        if indent == 0 and stripped.startswith("@"):
            pending_decorators.append(stripped)
            idx += 1
            continue

        class_match = re.match(r"(?:cdef\s+)?class\s+([A-Za-z_]\w*)", stripped)
        if indent == 0 and class_match:
            signature, end_idx = collect_python_signature(lines, idx)
            doc, _ = collect_python_docstring(lines, end_idx + 1)
            members = collect_python_class_members(
                lines, end_idx + 1, indent, path
            )
            symbols.append(
                PythonSymbol(
                    name=class_match.group(1),
                    kind="class",
                    signature=signature.rstrip(":"),
                    doc=doc,
                    source=path.relative_to(REPO_DIR).as_posix(),
                    line=idx + 1,
                    decorators=pending_decorators,
                    members=members,
                )
            )
            pending_decorators = []
            idx = end_idx + 1
            continue

        function_match = re.match(r"def\s+([A-Za-z_]\w*)", stripped)
        if indent == 0 and function_match:
            signature, end_idx = collect_python_signature(lines, idx)
            doc, _ = collect_python_docstring(lines, end_idx + 1)
            symbols.append(
                PythonSymbol(
                    name=function_match.group(1),
                    kind="function",
                    signature=signature.rstrip(":"),
                    doc=doc,
                    source=path.relative_to(REPO_DIR).as_posix(),
                    line=idx + 1,
                    decorators=pending_decorators,
                )
            )
            pending_decorators = []
            idx = end_idx + 1
            continue

        assignment_match = re.match(r"([A-Z][A-Z0-9_]+)\s*=", stripped)
        if indent == 0 and assignment_match:
            name = assignment_match.group(1)
            value, end_idx = collect_python_assignment(lines, idx)
            symbols.append(
                PythonSymbol(
                    name=name,
                    kind="constant",
                    signature=name,
                    doc="",
                    source=path.relative_to(REPO_DIR).as_posix(),
                    line=idx + 1,
                    value=value,
                )
            )
            idx = end_idx + 1
            continue

        if indent == 0 and stripped and not stripped.startswith("#"):
            pending_decorators = []
        idx += 1
    return symbols


def collect_python_class_members(
    lines: list[str],
    start: int,
    class_indent: int,
    path: Path,
) -> list[PythonSymbol]:
    members: list[PythonSymbol] = []
    decorators: list[str] = []
    idx = start
    while idx < len(lines):
        raw_line = lines[idx]
        stripped = raw_line.strip()
        indent = indentation(raw_line)
        if (
            stripped
            and indent <= class_indent
            and not stripped.startswith("@")
        ):
            break
        if indent > class_indent and stripped.startswith("@"):
            decorators.append(stripped)
            idx += 1
            continue
        method_match = re.match(r"def\s+([A-Za-z_]\w*)", stripped)
        if indent > class_indent and method_match:
            name = method_match.group(1)
            signature, end_idx = collect_python_signature(lines, idx)
            doc, _ = collect_python_docstring(lines, end_idx + 1)
            if not name.startswith("_") or name == "__init__":
                kind = "property" if "@property" in decorators else "method"
                members.append(
                    PythonSymbol(
                        name=name,
                        kind=kind,
                        signature=signature.rstrip(":"),
                        doc=doc,
                        source=path.relative_to(REPO_DIR).as_posix(),
                        line=idx + 1,
                        decorators=decorators,
                    )
                )
            decorators = []
            idx = end_idx + 1
            continue
        if stripped and indent <= class_indent:
            decorators = []
        idx += 1
    return members


def render_python_symbol(symbol: PythonSymbol) -> list[str]:
    lines = [f"## {heading_text(symbol.name)}", ""]
    if symbol.kind == "constant":
        lines.extend(["```python", symbol.value or symbol.name, "```", ""])
        lines.extend([f"_Source: `{symbol.source}:{symbol.line}`_", ""])
        return trim_blank_lines(lines)

    for decorator in symbol.decorators:
        lines.append(f"`{decorator}`")
    if symbol.decorators:
        lines.append("")
    lines.extend(["```python", symbol.signature, "```", ""])
    if symbol.doc:
        lines.extend(render_doc_text(symbol.doc))
        lines.append("")

    init_member = next(
        (member for member in symbol.members if member.name == "__init__"),
        None,
    )
    if init_member is not None:
        lines.extend(
            [
                "**Constructor**",
                "",
                "```python",
                init_member.signature,
                "```",
                "",
            ]
        )

    visible_members = [
        member for member in symbol.members if member.name != "__init__"
    ]
    if visible_members:
        lines.extend(
            [
                "**Members**",
                "",
                "| Name | Kind |",
                "| --- | --- |",
            ]
        )
        for member in visible_members:
            lines.append(f"| `{escape_code(member.name)}` | {member.kind} |")
        lines.append("")
        for member in visible_members:
            lines.extend(
                [
                    f"### {heading_text(member.name)}",
                    "",
                    "```python",
                    member.signature,
                    "```",
                    "",
                ]
            )
            if member.doc:
                lines.extend(render_doc_text(member.doc))
                lines.append("")
            lines.extend([f"_Source: `{member.source}:{member.line}`_", ""])

    lines.extend([f"_Source: `{symbol.source}:{symbol.line}`_", ""])
    return trim_blank_lines(lines)


def generate_java_api_pages() -> None:
    out_dir = FERN_PAGES / "java_api"
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = collect_java_classes()

    index_lines = [
        "# Java API Documentation",
        "",
        "These pages are generated from the Java source files in `java/cuvs-java/src/main`.",
        "",
    ]
    by_package: dict[str, list[JavaClass]] = defaultdict(list)
    for klass in classes:
        by_package[klass.package].append(klass)
    for package in sorted(by_package):
        index_lines.extend([f"## `{package}`", ""])
        for klass in sorted(by_package[package], key=lambda item: item.name):
            index_lines.append(
                f"- [{klass.name}]({api_doc_url('java_api', java_slug(klass))})"
            )
        index_lines.append("")
    write_page(out_dir / "index.md", index_lines)

    for klass in classes:
        lines = [
            *api_frontmatter(api_page_route("java_api", java_slug(klass))),
            f"# {klass.name}",
            "",
            f"_Java package: `{klass.package}`_",
            "",
            "```java",
            klass.signature,
            "```",
            "",
        ]
        lines.extend(render_javadoc(klass.doc))
        if klass.doc.summary:
            lines.append("")
        if klass.members:
            lines.extend(["## Public Members", ""])
            for member in klass.members:
                lines.extend(
                    [
                        f"### {heading_text(member.name)}",
                        "",
                        "```java",
                        member.signature,
                        "```",
                        "",
                    ]
                )
                lines.extend(render_javadoc(member.doc))
                if (
                    member.doc.summary
                    or member.doc.params
                    or member.doc.returns
                ):
                    lines.append("")
                lines.extend([f"_Source: `{klass.source}:{member.line}`_", ""])
        lines.extend([f"_Source: `{klass.source}:{klass.line}`_", ""])
        write_page(
            out_dir / f"{api_page_route('java_api', java_slug(klass))}.md",
            lines,
        )


def collect_java_classes() -> list[JavaClass]:
    classes: list[JavaClass] = []
    for root in JAVA_SOURCE_DIRS:
        for path in sorted(root.rglob("*.java")):
            if (
                "internal" in path.relative_to(root).parts
                or path.name == "module-info.java"
            ):
                continue
            klass = parse_java_class(path)
            if klass is not None:
                classes.append(klass)
    classes.sort(key=lambda item: (item.package, item.name))
    return classes


def parse_java_class(path: Path) -> JavaClass | None:
    text = path.read_text(encoding="utf-8")
    package_match = re.search(r"^\s*package\s+([\w.]+);", text, re.MULTILINE)
    package = package_match.group(1) if package_match else ""
    type_match = PUBLIC_JAVA_TYPE_RE.search(text)
    if not type_match:
        return None
    signature, class_line = java_signature_at(text, type_match.start())
    doc = parse_javadoc(comment_before(text, type_match.start()))
    klass = JavaClass(
        package=package,
        name=type_match.group("name"),
        kind=type_match.group("kind"),
        signature=signature,
        doc=doc,
        source=path.relative_to(REPO_DIR).as_posix(),
        line=class_line,
    )
    klass.members = parse_java_members(text, klass.name)
    return klass


def parse_java_members(text: str, class_name: str) -> list[JavaMember]:
    members: list[JavaMember] = []
    for match in COMMENT_RE.finditer(text):
        signature, line = java_signature_at(text, match.end())
        if "(" not in signature:
            continue
        if re.search(r"\b(class|interface|enum|record)\b", signature):
            continue
        if re.search(r"\b(if|for|while|switch|catch)\s*\(", signature):
            continue
        name_match = re.search(r"([A-Za-z_]\w*)\s*\(", signature)
        if not name_match:
            continue
        name = name_match.group(1)
        doc = parse_javadoc(match.group(0))
        members.append(
            JavaMember(name=name, signature=signature, doc=doc, line=line)
        )
    return members


def clean_doxygen_comment(raw: str) -> str:
    if raw.lstrip().startswith("///"):
        lines = [re.sub(r"^\s*/// ?", "", line) for line in raw.splitlines()]
    else:
        body = raw
        body = re.sub(r"^\s*/\*\* ?", "", body)
        body = re.sub(r"\*/\s*$", "", body)
        lines = [re.sub(r"^\s*\* ?", "", line) for line in body.splitlines()]
    return "\n".join(line.rstrip() for line in lines).strip()


def find_doxygen_command(comment: str, command: str) -> str | None:
    match = re.search(rf"[@\\]{command}\s+(.+)", comment)
    return match.group(1).strip() if match else None


def split_command_payload(payload: str) -> tuple[str, str]:
    parts = payload.split(None, 1)
    if not parts:
        return "", ""
    return parts[0], parts[1].strip() if len(parts) > 1 else ""


def read_declaration_after(text: str, offset: int) -> tuple[str, int]:
    idx = offset
    line_no = text.count("\n", 0, offset) + 1
    while idx < len(text):
        line_end = text.find("\n", idx)
        if line_end == -1:
            line_end = len(text)
        line = text[idx:line_end]
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped
            in {'extern "C" {', "{", "}", "public:", "private:", "protected:"}
        ):
            idx = line_end + 1
            line_no += 1
            continue
        if stripped.startswith(("//", "/*")):
            return "", line_no
        break

    declaration_start = idx
    depth = {"(": 0, "[": 0, "{": 0, "<": 0}
    saw_brace = False
    declaration_probe = text[declaration_start : declaration_start + 1000]
    is_compound = is_compound_declaration(declaration_probe)
    while idx < len(text):
        if text.startswith("//", idx):
            next_line = text.find("\n", idx)
            if next_line == -1:
                idx = len(text)
            else:
                idx = next_line
            continue
        if text.startswith("/*", idx):
            end_comment = text.find("*/", idx + 2)
            idx = len(text) if end_comment == -1 else end_comment + 2
            continue
        char = text[idx]
        update_depth(depth, char)
        if char == "{":
            saw_brace = True
        if char == "}" and saw_brace and not depth["{"] and not is_compound:
            idx += 1
            break
        if char == ";" and structural_depth_is_zero(depth):
            idx += 1
            break
        if (
            char == ","
            and not any(depth.values())
            and not saw_brace
            and not is_compound
        ):
            idx += 1
            break
        idx += 1

    declaration = text[declaration_start:idx].strip()
    declaration = re.sub(r"///<.*", "", declaration)
    declaration = "\n".join(
        line.rstrip() for line in declaration.splitlines()
    ).strip()
    if is_simple_member_declaration(declaration):
        declaration = declaration.splitlines()[0].rstrip(",")
    return declaration, line_no


def structural_depth_is_zero(depth: dict[str, int]) -> bool:
    return not depth["("] and not depth["["] and not depth["{"]


def next_non_whitespace_is_comment(text: str, offset: int) -> bool:
    idx = offset
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return text.startswith("/**", idx) or text.startswith("///", idx)


def is_simple_member_declaration(declaration: str) -> bool:
    if is_compound_declaration(declaration):
        return False
    first_line = declaration.splitlines()[0].strip() if declaration else ""
    if not first_line or re.search(r"\b(?:struct|class|enum)\b", first_line):
        return False
    if "(" in first_line:
        return False
    return "\n" in declaration and "\n}" in declaration


def parse_doxygen_entry(
    comment: str, declaration: str, source: str, line: int
) -> DoxygenEntry:
    summary = ""
    details: list[str] = []
    params: list[DoxygenParam] = []
    tparams: list[DoxygenParam] = []
    returns = ""
    code_blocks: list[tuple[str, list[str]]] = []
    active_code: tuple[str, list[str]] | None = None
    active_param: DoxygenParam | None = None

    for raw_line in comment.splitlines():
        line_text = raw_line.strip()
        if active_code is not None:
            if re.match(r"[@\\]endcode", line_text):
                code_blocks.append(active_code)
                active_code = None
            else:
                active_code[1].append(raw_line.rstrip())
            continue

        code_match = re.match(r"[@\\]code(?:\{\.?([\w+-]+)\})?", line_text)
        if code_match:
            active_code = (code_match.group(1) or "", [])
            active_param = None
            continue

        if re.match(
            r"[@\\](?:defgroup|ingroup|addtogroup)\b", line_text
        ) or line_text in {
            "@{",
            "\\{",
            "@}",
            "\\}",
        }:
            active_param = None
            continue

        brief = consume_command(line_text, "brief")
        if brief is not None:
            summary = append_sentence(summary, clean_doxygen_text(brief))
            active_param = None
            continue

        param_match = re.match(
            r"[@\\]param(?:\[(?P<direction>[^\]]+)\])?\s+(?P<name>\w+)\s*(?P<desc>.*)",
            line_text,
        )
        if param_match:
            active_param = DoxygenParam(
                param_match.group("name"),
                clean_doxygen_text(param_match.group("desc")),
                param_match.group("direction") or "",
            )
            params.append(active_param)
            continue

        tparam_match = re.match(r"[@\\]tparam\s+(\w+)\s*(.*)", line_text)
        if tparam_match:
            active_param = DoxygenParam(
                tparam_match.group(1),
                clean_doxygen_text(tparam_match.group(2)),
            )
            tparams.append(active_param)
            continue

        return_text = consume_command(line_text, "return") or consume_command(
            line_text, "returns"
        )
        if return_text is not None:
            returns = append_sentence(returns, clean_doxygen_text(return_text))
            active_param = None
            continue

        if line_text.startswith(("@", "\\")):
            active_param = None
            continue

        if active_param is not None and (
            raw_line.startswith((" ", "\t")) or not line_text
        ):
            active_param.description = append_doxygen_line(
                active_param.description, clean_doxygen_text(line_text)
            )
            continue

        details.append(clean_doxygen_text(raw_line.rstrip()))
        active_param = None

    if active_code is not None:
        code_blocks.append(active_code)

    details = trim_blank_lines(details)
    if not summary and details:
        summary = details.pop(0).strip()

    kind = parse_doxygen_kind(declaration)
    if is_type_alias_signature(declaration):
        name = parse_type_alias_name(declaration) or parse_entry_name(
            declaration
        )
    elif kind == "member":
        name = parse_member_name(declaration)
    else:
        name = parse_entry_name(declaration)
    return DoxygenEntry(
        kind=kind,
        name=name,
        signature=normalize_entry_signature(declaration, kind),
        summary=summary,
        details=details,
        params=params,
        tparams=tparams,
        returns=returns,
        source=source,
        line=line,
        code_blocks=code_blocks,
    )


def consume_command(line: str, command: str) -> str | None:
    match = re.match(rf"[@\\]{command}\b\s*(.*)", line)
    return match.group(1).strip() if match else None


def clean_doxygen_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[@\\](?:p|c|a)\s+([\w:]+)", r"`\1`", text)
    text = re.sub(
        r"[@\\]ref\s+([\w:]+)(?:\s+\"([^\"]+)\")?",
        lambda match: match.group(2) or f"`{match.group(1)}`",
        text,
    )
    text = normalize_doxygen_math(text)
    text = text.replace("@copydoc", "")
    return text.strip()


def normalize_doxygen_math(text: str) -> str:
    text = re.sub(
        r"\\f\[(.*?)\\f\]",
        lambda match: math_placeholder(
            f"${clean_latex_math(match.group(1))}$"
        ),
        text,
    )
    text = re.sub(
        r"\\f\$(.*?)\\f\$",
        lambda match: math_placeholder(
            f"${clean_latex_math(match.group(1))}$"
        ),
        text,
    )
    return text


def clean_latex_math(math: str) -> str:
    math = math.strip()
    math = re.sub(
        r"<\s*([^<>]*?)\s*>",
        lambda match: rf"\langle {match.group(1).strip()} \rangle",
        math,
    )
    math = re.sub(
        r"\|\s*([^|]*?)\s*\|",
        lambda match: rf"\lVert {match.group(1).strip()} \rVert",
        math,
    )
    math = re.sub(
        r"(\\(?:mathrm|operatorname|text)\{)([^{}]*)(\})",
        lambda match: (
            f"{match.group(1)}"
            f"{re.sub(r'(?<!\\\\)_', r'\\\\_', match.group(2))}"
            f"{match.group(3)}"
        ),
        math,
    )
    return math


def math_placeholder(math: str) -> str:
    return f"@@FERN_MATH_{math.encode('utf-8').hex()}@@"


def append_sentence(existing: str, addition: str) -> str:
    addition = addition.strip()
    if not addition:
        return existing
    if not existing:
        return addition
    return f"{existing} {addition}"


def append_doxygen_line(existing: str, addition: str) -> str:
    addition = addition.strip()
    if not addition:
        return existing
    if not existing:
        return addition
    lines = existing.splitlines()
    if DOXYGEN_LIST_ITEM_RE.match(addition):
        lines.append(addition)
    else:
        lines[-1] = append_sentence(lines[-1], addition)
    return "\n".join(lines)


def parse_doxygen_kind(declaration: str) -> str:
    untemplated = strip_leading_cpp_templates(declaration)
    if re.search(r"^\s*(?:typedef\s+)?(?:struct|class)\b", untemplated) and (
        "{" in declaration or not untemplated.lstrip().startswith("typedef")
    ):
        return "struct"
    if (
        re.search(r"^\s*(?:typedef\s+)?enum(?:\s+class)?\b", untemplated)
        and "{" in declaration
    ):
        return "enum"
    if "(" in declaration and ")" in declaration:
        return "function"
    return "member"


def parse_type_alias_name(declaration: str) -> str | None:
    signature = normalize_signature(declaration).rstrip(";").strip()
    using_match = re.match(r"using\s+([A-Za-z_]\w*)\s*=", signature)
    if using_match:
        return using_match.group(1)
    function_pointer_match = re.search(
        r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)", signature
    )
    if function_pointer_match:
        return function_pointer_match.group(1)
    compound_alias_match = re.search(
        r"}\s*([A-Za-z_]\w*)\s*$", signature, re.DOTALL
    )
    if compound_alias_match:
        return compound_alias_match.group(1)
    typedef_match = re.search(r"\b([A-Za-z_]\w*)\s*$", signature)
    if typedef_match and signature.startswith("typedef "):
        return typedef_match.group(1)
    return None


def parse_entry_name(declaration: str) -> str:
    struct_name = parse_struct_name(declaration)
    if struct_name:
        return struct_name
    enum_name = parse_enum_name(declaration)
    if enum_name:
        return enum_name
    before_paren = declaration.split("(", 1)[0].strip()
    if "(" in declaration and before_paren:
        return before_paren.split()[-1].split("::")[-1].strip("*&")
    declaration_lines = [
        line.strip() for line in declaration.splitlines() if line.strip()
    ]
    first_line = declaration_lines[0] if declaration_lines else ""
    name_line = (
        declaration_lines[-1] if len(declaration_lines) > 1 else first_line
    )
    token_match = re.match(
        r"\s*(?:[\w:<>,]+\s+)*([A-Za-z_]\w*)\s*(?:=|;|,|$)", name_line
    )
    return token_match.group(1) if token_match else first_line.strip()


def parse_member_name(declaration: str) -> str:
    declaration = strip_member_initializer(declaration)
    _, name = split_param_name(declaration)
    return name or declaration


def parse_struct_name(declaration: str) -> str | None:
    declaration = strip_leading_cpp_templates(declaration)
    match = re.search(
        r"^\s*(?:typedef\s+)?(?:struct|class)\s+([A-Za-z_]\w*)",
        declaration,
    )
    return match.group(1) if match else None


def is_class_signature(signature: str) -> bool:
    signature = strip_leading_cpp_templates(signature)
    return bool(
        re.match(
            r"^\s*(?:typedef\s+)?class\b",
            signature,
        )
    )


def parse_enum_name(declaration: str) -> str | None:
    declaration = strip_leading_cpp_templates(declaration)
    if not re.search(r"^\s*(?:typedef\s+)?enum(?:\s+class)?\b", declaration):
        return None
    match = re.search(
        r"^\s*(?:typedef\s+)?enum(?:\s+class)?\s+([A-Za-z_]\w*)", declaration
    )
    if match:
        return match.group(1)
    typedef_match = re.search(
        r"}\s*([A-Za-z_]\w*)\s*;", declaration, re.DOTALL
    )
    return typedef_match.group(1) if typedef_match else None


def infer_namespace(prefix: str) -> str:
    return infer_cpp_scope(prefix)[0]


def strip_leading_cpp_templates(declaration: str) -> str:
    text = declaration.lstrip()
    while True:
        match = re.match(r"template\s*<", text)
        if not match:
            return text
        idx = match.end()
        depth = 1
        while idx < len(text) and depth:
            char = text[idx]
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            idx += 1
        if depth:
            return text
        text = text[idx:].lstrip()


def is_compound_declaration(declaration: str) -> bool:
    return bool(
        CPP_COMPOUND_RE.match(strip_leading_cpp_templates(declaration))
    )


def infer_cpp_scope(prefix: str) -> tuple[str, list[str]]:
    text = COMMENT_RE.sub("", prefix)
    text = re.sub(r"//.*", "", text)
    scope_stack: list[tuple[str, str, int]] = []
    depth = 0
    token_re = re.compile(
        r"\bnamespace\s+([A-Za-z_][\w:]*)(?:\s*=\s*[^;{}]+)?\s*{"
        r"|\b(?:class|struct)\s+([A-Za-z_]\w*)[^;{}]*{"
        r"|[{}]"
    )
    for match in token_re.finditer(text):
        namespace_name = match.group(1)
        if namespace_name:
            depth += 1
            scope_stack.append(("namespace", namespace_name, depth))
            continue
        class_name = match.group(2)
        if class_name:
            depth += 1
            scope_stack.append(("class", class_name, depth))
            continue
        token = match.group(0)
        if token == "{":
            depth += 1
        elif token == "}":
            depth = max(depth - 1, 0)
            while scope_stack and scope_stack[-1][2] > depth:
                scope_stack.pop()
    namespaces = [name for kind, name, _ in scope_stack if kind == "namespace"]
    classes = [name for kind, name, _ in scope_stack if kind == "class"]
    return "::".join(namespaces), classes


def normalize_entry_signature(declaration: str, kind: str) -> str:
    signature = normalize_signature(declaration)
    if kind == "function" and "{" in signature:
        signature = signature.split("{", 1)[0].rstrip()
        if not signature.endswith(";"):
            signature = f"{signature};"
    return signature


def normalize_signature(declaration: str) -> str:
    declaration = re.sub(r"\n\s+", "\n", declaration.strip())
    return "\n".join(
        line.rstrip() for line in declaration.splitlines()
    ).strip()


def compact_compound_signature(signature: str) -> str:
    signature = signature.strip()
    if "{" not in signature:
        return signature.splitlines()[0] if signature else ""
    prefix = "\n".join(
        line.rstrip() for line in signature.split("{", 1)[0].splitlines()
    ).strip()
    suffix = signature.rsplit("}", 1)[1].strip() if "}" in signature else ""
    if suffix:
        separator = "" if suffix.startswith(";") else " "
        return f"{prefix} {{ ... }}{separator}{suffix}".strip()
    return f"{prefix} {{ ... }};"


def parse_function_params(signature: str) -> list[FunctionParam]:
    params_text = function_params_text(signature)
    if not params_text or params_text.strip() == "void":
        return []
    params: list[FunctionParam] = []
    for idx, raw_param in enumerate(
        split_top_level(params_text, ","), start=1
    ):
        param = raw_param.strip()
        if not param:
            continue
        default = ""
        if "=" in param:
            left, default = split_default(param)
            param = left.strip()
            default = default.strip()
        c_type, name = split_param_name(param)
        params.append(
            FunctionParam(
                name=name or f"arg{idx}",
                c_type=c_type or param,
                default=default,
            )
        )
    return params


def function_params_text(signature: str) -> str:
    start = signature.find("(")
    if start == -1:
        return ""
    depth = 0
    for idx in range(start, len(signature)):
        char = signature[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return signature[start + 1 : idx]
    return ""


def split_default(param: str) -> tuple[str, str]:
    depth = {"(": 0, "[": 0, "{": 0, "<": 0}
    for idx, char in enumerate(param):
        update_depth(depth, char)
        if char == "=" and not any(depth.values()):
            return param[:idx], param[idx + 1 :]
    return param, ""


def split_param_name(param: str) -> tuple[str, str]:
    param = re.sub(r"\s+", " ", param.strip())
    param = re.sub(r"\[[^\]]*\]\s*$", "", param)
    match = re.search(r"([A-Za-z_]\w*)\s*$", param)
    if not match:
        return param, ""
    name = match.group(1)
    c_type = param[: match.start()].rstrip()
    c_type = c_type.rstrip("*& ").strip()
    suffix = param[: match.start()][len(c_type) :]
    if suffix.strip():
        c_type = f"{c_type}{suffix.strip()}".strip()
    if not c_type:
        return param, ""
    return c_type, name


def parse_return_type(signature: str) -> str:
    if ")" in signature and "->" in signature.split(")", 1)[1]:
        tail = signature.split(")", 1)[1]
        return tail.split("->", 1)[1].strip().rstrip(";")
    before_paren = signature.split("(", 1)[0]
    before_paren = re.sub(
        r"template\s*<[^>]+>", "", before_paren, flags=re.DOTALL
    )
    before_paren = " ".join(before_paren.split())
    if not before_paren:
        return "void"
    tokens = before_paren.split()
    if len(tokens) <= 1:
        return "void"
    return " ".join(tokens[:-1]).strip()


def parse_struct_members(entry: DoxygenEntry) -> list[DoxygenEntry]:
    members: list[DoxygenEntry] = []
    for match in COMMENT_RE.finditer(entry.signature):
        if not is_top_level_compound_comment(entry.signature, match.start()):
            continue
        comment = clean_doxygen_comment(match.group(0))
        declaration, line = read_declaration_after(
            entry.signature, match.end()
        )
        declaration = declaration.strip()
        if not declaration or declaration.startswith(
            ("public:", "private:", "protected:")
        ):
            continue
        if declaration in {"}", "};"}:
            continue
        if parse_doxygen_kind(declaration) != "member":
            continue
        member = parse_doxygen_entry(
            comment, declaration, entry.source, entry.line + line - 1
        )
        member.kind = "member"
        members.append(member)
    members.extend(
        parse_plain_struct_members(entry, {member.name for member in members})
    )
    return members


def parse_plain_struct_members(
    entry: DoxygenEntry, existing_names: set[str]
) -> list[DoxygenEntry]:
    body = compound_body(entry.signature)
    if not body:
        return []
    members: list[DoxygenEntry] = []
    for declaration in top_level_member_declarations(body):
        if "(" in declaration:
            continue
        declaration = strip_member_initializer(declaration)
        if not declaration:
            continue
        if not declaration or declaration.startswith(
            (
                "enum ",
                "friend ",
                "static_assert",
                "struct ",
                "template ",
                "typedef ",
                "using ",
            )
        ):
            continue
        c_type, name = split_param_name(declaration)
        if not name or name in existing_names:
            continue
        member = DoxygenEntry(
            kind="member",
            name=name,
            signature=c_type or declaration,
            summary="",
            source=entry.source,
            line=entry.line,
        )
        members.append(member)
        existing_names.add(name)
    return members


def is_top_level_compound_comment(signature: str, offset: int) -> bool:
    bounds = compound_body_bounds(signature)
    if bounds is None:
        return False
    body_start, body_end = bounds
    if offset < body_start or offset >= body_end:
        return False
    depth = 0
    for char in signature[body_start:offset]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(depth - 1, 0)
    return depth == 0


def top_level_member_declarations(body: str) -> list[str]:
    text = COMMENT_RE.sub("", body)
    text = re.sub(r"//.*", "", text)
    declarations: list[str] = []
    current: list[str] = []
    depth = {"(": 0, "[": 0, "{": 0, "<": 0}
    for char in text:
        current.append(char)
        update_depth(depth, char)
        if char == ";" and not any(depth.values()):
            declaration = "".join(current).strip().rstrip(";").strip()
            if declaration:
                declarations.append(" ".join(declaration.split()))
            current = []
    return declarations


def extract_field_descriptions(
    details: list[str], field_names: set[str]
) -> tuple[dict[str, str], list[str]]:
    descriptions: dict[str, str] = {}
    remaining: list[str] = []
    idx = 0
    while idx < len(details):
        line = details[idx]
        match = DOXYGEN_FIELD_LIST_ITEM_RE.match(line.strip())
        if match and match.group("name") in field_names:
            name = match.group("name")
            description = [match.group("description").strip()]
            idx += 1
            while idx < len(details):
                next_line = details[idx]
                next_stripped = next_line.strip()
                if (
                    not next_stripped
                    or DOXYGEN_LIST_ITEM_RE.match(next_stripped)
                    or DOXYGEN_FIELD_LIST_ITEM_RE.match(next_stripped)
                ):
                    break
                description.append(next_stripped)
                idx += 1
            descriptions[name] = " ".join(part for part in description if part)
            continue
        remaining.append(line)
        idx += 1
    return descriptions, trim_blank_lines(remaining)


def compound_body(signature: str) -> str:
    bounds = compound_body_bounds(signature)
    if bounds is None:
        return ""
    start, end = bounds
    return signature[start:end]


def compound_body_bounds(signature: str) -> tuple[int, int] | None:
    start = signature.find("{")
    end = signature.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return start + 1, end


def strip_member_initializer(declaration: str) -> str:
    declaration = declaration.strip().rstrip(";")
    declaration, _ = split_default(declaration)
    declaration = declaration.strip()
    if declaration.endswith("}") and "{" in declaration:
        before_brace = declaration.rsplit("{", 1)[0].strip()
        if re.search(r"\b[A-Za-z_]\w*$", before_brace):
            return before_brace
    return declaration


def split_inline_comment(line: str) -> tuple[str, str]:
    if "//" not in line:
        return line, ""
    declaration, comment = line.split("//", 1)
    return declaration, comment.lstrip("/<").strip()


def member_c_type(member: DoxygenEntry) -> str:
    declaration = strip_member_initializer(member.signature)
    c_type, name = split_param_name(declaration)
    if name and name != member.name:
        return declaration
    return c_type or declaration


def member_description(member: DoxygenEntry) -> str:
    lines = []
    if member.summary:
        lines.append(member.summary)
    lines.extend(member.details)
    return "\n".join(line for line in lines if line.strip())


def parse_enum_values(signature: str) -> list[dict[str, str]]:
    body_match = re.search(r"{(?P<body>.*)}", signature, re.DOTALL)
    if not body_match:
        return []
    values: list[dict[str, str]] = []
    for raw_value in split_top_level(body_match.group("body"), ","):
        cleaned = re.sub(
            r"/\*.*?\*/|//.*", "", raw_value, flags=re.DOTALL
        ).strip()
        if not cleaned:
            continue
        name, _, value = cleaned.partition("=")
        enum_name = name.strip()
        if re.match(r"^[A-Za-z_]\w*$", enum_name):
            item = {"name": enum_name}
            if value.strip():
                item["value"] = value.strip()
            values.append(item)
    return values


def collect_python_signature(lines: list[str], start: int) -> tuple[str, int]:
    chunks: list[str] = []
    depth = 0
    idx = start
    while idx < len(lines):
        stripped = lines[idx].strip()
        chunks.append(stripped)
        depth += (
            stripped.count("(") + stripped.count("[") + stripped.count("{")
        )
        depth -= (
            stripped.count(")") + stripped.count("]") + stripped.count("}")
        )
        if stripped.endswith(":") and depth <= 0:
            break
        idx += 1
    return " ".join(chunks), idx


def collect_python_assignment(lines: list[str], start: int) -> tuple[str, int]:
    chunks = [lines[start].strip()]
    depth = chunks[0].count("{") + chunks[0].count("[") + chunks[0].count("(")
    depth -= chunks[0].count("}") + chunks[0].count("]") + chunks[0].count(")")
    idx = start
    while depth > 0 and idx + 1 < len(lines):
        idx += 1
        stripped = lines[idx].strip()
        chunks.append(stripped)
        depth += (
            stripped.count("{") + stripped.count("[") + stripped.count("(")
        )
        depth -= (
            stripped.count("}") + stripped.count("]") + stripped.count(")")
        )
    return "\n".join(chunks), idx


def collect_python_docstring(lines: list[str], start: int) -> tuple[str, int]:
    idx = start
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return "", idx
    stripped = lines[idx].strip()
    quote = None
    for candidate in ('"""', "'''"):
        if stripped.startswith(candidate):
            quote = candidate
            break
    if quote is None:
        return "", idx

    content: list[str] = []
    remainder = stripped[len(quote) :]
    if quote in remainder:
        content.append(remainder.split(quote, 1)[0])
        return clean_python_docstring("\n".join(content)), idx
    if remainder:
        content.append(remainder)
    idx += 1
    while idx < len(lines):
        line = lines[idx]
        if quote in line:
            content.append(line.split(quote, 1)[0])
            return clean_python_docstring("\n".join(content)), idx
        content.append(line)
        idx += 1
    return clean_python_docstring("\n".join(content)), idx


def clean_python_docstring(doc: str) -> str:
    doc = textwrap.dedent(doc).strip()
    doc = doc.replace("\\\n", "")
    doc = doc.replace(
        "{resources_docstring}", "resources : cuvs.common.Resources, optional"
    )
    return strip_sphinx_roles(doc)


def strip_sphinx_roles(value: str) -> str:
    return SPHINX_ROLE_RE.sub(
        lambda match: sphinx_role_label(match.group("target")), value
    )


def sphinx_role_label(target: str) -> str:
    target = target.strip()
    if target.startswith("~"):
        target = target[1:]
    if "<" in target and target.endswith(">"):
        label, _, destination = target.rpartition("<")
        return (label.strip() or destination[:-1].strip()).strip()
    return target


NUMPY_FIELD_SECTIONS = {
    "attributes",
    "keyword arguments",
    "other parameters",
    "parameters",
    "raises",
    "returns",
    "warns",
    "warnings",
    "yields",
}
NUMPY_SECTION_UNDERLINE_RE = re.compile(r"^-{3,}\s*$")
NUMPY_FIELD_RE = re.compile(
    r"^(?P<name>[A-Za-z_*][A-Za-z0-9_*,\s]*?)\s*:\s*(?P<type>.*)$"
)


def render_doc_text(doc: str) -> list[str]:
    preamble, sections = split_numpy_doc_sections(doc.splitlines())
    if not sections:
        return render_doc_lines(doc.splitlines())

    lines: list[str] = []
    if preamble:
        lines.extend(render_doc_lines(preamble))
        lines.append("")
    for title, section_lines in sections:
        if lines and lines[-1] != "":
            lines.append("")
        if title.lower() in NUMPY_FIELD_SECTIONS:
            lines.extend(render_numpy_field_section(title, section_lines))
        else:
            lines.extend([f"**{heading_text(title)}**", ""])
            lines.extend(render_doc_lines(section_lines))
        lines.append("")
    return trim_blank_lines(lines)


def split_numpy_doc_sections(
    raw_lines: list[str],
) -> tuple[list[str], list[tuple[str, list[str]]]]:
    preamble: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    idx = 0
    while idx < len(raw_lines):
        line = raw_lines[idx]
        if (
            idx + 1 < len(raw_lines)
            and line.strip()
            and NUMPY_SECTION_UNDERLINE_RE.match(raw_lines[idx + 1].strip())
        ):
            if current_title is None:
                preamble = current_lines
            else:
                sections.append((current_title, current_lines))
            current_title = line.strip()
            current_lines = []
            idx += 2
            continue
        current_lines.append(line)
        idx += 1

    if current_title is None:
        return current_lines, []
    sections.append((current_title, current_lines))
    return preamble, sections


def render_numpy_field_section(title: str, raw_lines: list[str]) -> list[str]:
    fields = parse_numpy_fields(raw_lines)
    if not fields:
        lines = [f"**{heading_text(title)}**", ""]
        lines.extend(render_doc_lines(raw_lines))
        return trim_blank_lines(lines)

    lines = [
        f"**{heading_text(title)}**",
        "",
        "| Name | Type | Description |",
        "| --- | --- | --- |",
    ]
    for doc_field in fields:
        name = render_numpy_field_names(doc_field["name"])
        field_type = (
            f"`{escape_code(doc_field['type'])}`" if doc_field["type"] else ""
        )
        description = render_table_description(doc_field["description"])
        lines.append(f"| {name} | {field_type} | {description} |")
    return lines


def parse_numpy_fields(raw_lines: list[str]) -> list[dict[str, str]]:
    fields: list[dict[str, str]] = []
    current: dict[str, object] | None = None
    for raw_line in dedent_doc_lines(raw_lines):
        stripped = raw_line.strip()
        field_match = (
            NUMPY_FIELD_RE.match(stripped)
            if stripped and indentation(raw_line) == 0
            else None
        )
        if field_match:
            if current is not None:
                fields.append(finish_numpy_field(current))
            current = {
                "name": field_match.group("name").strip(),
                "type": field_match.group("type").strip(),
                "description": [],
            }
            continue
        if current is None:
            if stripped:
                return []
            continue
        description = current["description"]
        assert isinstance(description, list)
        if (
            stripped
            and not description
            and numpy_type_needs_continuation(str(current["type"]))
        ):
            current["type"] = f"{current['type']} {stripped}"
        else:
            description.append(stripped)
    if current is not None:
        fields.append(finish_numpy_field(current))
    return fields


def finish_numpy_field(field: dict[str, object]) -> dict[str, str]:
    description = field["description"]
    assert isinstance(description, list)
    return {
        "name": str(field["name"]),
        "type": str(field["type"]),
        "description": "\n".join(description).strip(),
    }


def numpy_type_needs_continuation(value: str) -> bool:
    value = value.strip()
    if value.endswith(","):
        return True
    depth = {"(": 0, "[": 0, "{": 0, "<": 0}
    for char in value:
        update_depth(depth, char)
    return any(depth.values())


def render_numpy_field_names(value: str) -> str:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        return ""
    return ", ".join(f"`{escape_code(name)}`" for name in names)


def render_table_description(value: str) -> str:
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for raw_line in value.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if current:
                paragraphs.append(current)
                current = []
            continue
        current.append(stripped)
    if current:
        paragraphs.append(current)

    rendered: list[str] = []
    for paragraph in paragraphs:
        normalized = normalize_description_lines(paragraph)
        if any(DOXYGEN_LIST_ITEM_RE.match(line) for line in normalized):
            rendered.append(
                "<br />".join(escape_text(line) for line in normalized)
            )
        else:
            rendered.append(escape_text(" ".join(normalized)))
    return "<br /><br />".join(rendered)


def normalize_description_lines(raw_lines: list[str]) -> list[str]:
    lines: list[str] = []
    paragraph: list[str] = []
    in_list = False

    for raw_line in raw_lines:
        line = raw_line.strip()
        if DOXYGEN_LIST_ITEM_RE.match(line):
            if paragraph:
                lines.append(" ".join(paragraph))
                paragraph = []
            lines.append(line)
            in_list = True
            continue

        if in_list and lines:
            lines[-1] = append_sentence(lines[-1], line)
            continue

        paragraph.append(line)

    if paragraph:
        lines.append(" ".join(paragraph))
    return lines


def render_doxygen_details(raw_lines: list[str]) -> list[str]:
    lines: list[str] = []
    paragraph: list[str] = []
    in_list = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            lines.append(escape_text(" ".join(paragraph)))
            paragraph = []

    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            flush_paragraph()
            if lines and lines[-1] != "":
                lines.append("")
            in_list = False
            continue

        if DOXYGEN_LIST_ITEM_RE.match(stripped):
            flush_paragraph()
            if lines and lines[-1] != "" and not in_list:
                lines.append("")
            lines.append(escape_text(stripped))
            in_list = True
            continue

        if in_list and lines and lines[-1] != "":
            lines[-1] = f"{lines[-1]} {escape_text(stripped)}"
            continue

        paragraph.append(stripped)

    flush_paragraph()
    return trim_blank_lines(lines)


def render_doc_lines(raw_lines: list[str]) -> list[str]:
    lines = []
    in_code = False
    for raw_line in dedent_doc_lines(raw_lines):
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith(">>>") or stripped.startswith("..."):
            if not in_code:
                lines.extend(["```python"])
                in_code = True
            lines.append(stripped)
            continue
        if in_code and stripped:
            lines.append(line)
            continue
        if in_code:
            lines.append("```")
            in_code = False
        lines.append(escape_text(stripped))
    if in_code:
        lines.append("```")
    return trim_blank_lines(lines)


def dedent_doc_lines(raw_lines: list[str]) -> list[str]:
    return textwrap.dedent("\n".join(raw_lines)).splitlines()


def read_python_exports(init_path: Path) -> list[str]:
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        return []
                    return [item for item in value if isinstance(item, str)]
    return []


def python_module_name(path: Path) -> str:
    rel = path.relative_to(PYTHON_DIR)
    if rel.name == "__init__.py":
        parts = rel.parent.parts
    else:
        parts = rel.with_suffix("").parts
    return ".".join(("cuvs", *parts))


def python_group(module: str) -> str:
    if module.startswith("cuvs.cluster"):
        return "Cluster"
    if module.startswith("cuvs.distance"):
        return "Distance"
    if module.startswith("cuvs.neighbors.mg"):
        return "Multi-GPU Neighbors"
    if module.startswith("cuvs.neighbors"):
        return "Nearest Neighbors"
    if module.startswith("cuvs.preprocessing"):
        return "Preprocessing"
    if module.startswith("cuvs.common"):
        return "Common"
    return "Other"


def group_python_pages(pages: list[PythonPage]) -> dict[str, list[PythonPage]]:
    grouped: dict[str, list[PythonPage]] = defaultdict(list)
    for page in pages:
        grouped[python_group(page.module)].append(page)
    return dict(sorted(grouped.items()))


def python_title(module: str) -> str:
    leaf = module.split(".")[-1]
    title = humanize_slug(leaf.replace("_", "-"))
    replacements = {
        "Mg": "Multi-GPU",
        "Ivf": "IVF",
        "Pq": "PQ",
        "Pca": "PCA",
        "Hnsw": "HNSW",
        "Nn": "NN",
    }
    for old, new in replacements.items():
        title = re.sub(rf"\b{old}\b", new, title)
    if module.startswith("cuvs.neighbors.mg."):
        title = f"Multi-GPU {title}"
    return title


def python_slug(module: str) -> str:
    return slugify(module.removeprefix("cuvs."))


def java_signature_at(text: str, start: int) -> tuple[str, int]:
    line = text.count("\n", 0, start) + 1
    idx = start
    depth = {"(": 0, "[": 0, "{": 0, "<": 0}
    while idx < len(text):
        char = text[idx]
        update_depth(depth, char)
        if char in "{;" and not depth["("] and not depth["["]:
            signature = text[start:idx].strip()
            signature = re.sub(r"\s+", " ", signature)
            return signature, line
        idx += 1
    signature = text[start:].splitlines()[0].strip()
    return re.sub(r"\s+", " ", signature), line


def comment_before(text: str, start: int) -> str:
    prefix = text[:start].rstrip()
    match = re.search(
        r"/\*\*.*?\*/\s*(?:@\w+(?:\([^)]*\))?\s*)*$", prefix, re.DOTALL
    )
    return match.group(0) if match else ""


def parse_javadoc(raw: str) -> JavaDoc:
    raw = raw.strip()
    if not raw:
        return JavaDoc()
    comment_match = re.search(r"/\*\*(.*?)\*/", raw, re.DOTALL)
    if not comment_match:
        return JavaDoc()
    body = comment_match.group(1)
    lines = [
        re.sub(r"^\s*\* ?", "", line).rstrip() for line in body.splitlines()
    ]
    doc = JavaDoc()
    summary_lines: list[str] = []
    active: DoxygenParam | None = None
    active_kind = ""
    for line in lines:
        stripped = clean_javadoc_text(line.strip())
        if not stripped:
            if active is None:
                summary_lines.append("")
            continue
        param_match = re.match(r"@param\s+(\w+)\s*(.*)", stripped)
        if param_match:
            active = DoxygenParam(
                param_match.group(1), param_match.group(2).strip()
            )
            doc.params.append(active)
            active_kind = "param"
            continue
        throws_match = re.match(r"@throws\s+([\w.]+)\s*(.*)", stripped)
        if throws_match:
            active = DoxygenParam(
                throws_match.group(1), throws_match.group(2).strip()
            )
            doc.throws.append(active)
            active_kind = "throws"
            continue
        returns_match = re.match(r"@return\s*(.*)", stripped)
        if returns_match:
            doc.returns = returns_match.group(1).strip()
            active = None
            active_kind = "return"
            continue
        if stripped.startswith("@"):
            active = None
            active_kind = ""
            continue
        if active is not None and active_kind in {"param", "throws"}:
            active.description = append_sentence(active.description, stripped)
        elif active_kind == "return":
            doc.returns = append_sentence(doc.returns, stripped)
        else:
            summary_lines.append(stripped)
    doc.summary = "\n".join(trim_blank_lines(summary_lines)).strip()
    return doc


def clean_javadoc_text(text: str) -> str:
    text = re.sub(r"\{@code\s+([^}]+)\}", r"`\1`", text)
    text = re.sub(
        r"\{@link\s+([^}\s]+)(?:\s+([^}]+))?\}",
        lambda m: m.group(2) or f"`{m.group(1)}`",
        text,
    )
    text = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", text)
    text = re.sub(r"</?p>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def render_javadoc(doc: JavaDoc) -> list[str]:
    lines: list[str] = []
    if doc.summary:
        lines.extend(escape_text(line) for line in doc.summary.splitlines())
        lines.append("")
    if doc.params:
        lines.extend(
            ["**Parameters**", "", "| Name | Description |", "| --- | --- |"]
        )
        for param in doc.params:
            lines.append(
                f"| `{escape_code(param.name)}` | {escape_text(param.description)} |"
            )
        lines.append("")
    if doc.returns:
        lines.extend(["**Returns**", "", escape_text(doc.returns), ""])
    if doc.throws:
        lines.extend(
            ["**Throws**", "", "| Type | Description |", "| --- | --- |"]
        )
        for param in doc.throws:
            lines.append(
                f"| `{escape_code(param.name)}` | {escape_text(param.description)} |"
            )
        lines.append("")
    return trim_blank_lines(lines)


def generate_rust_api_pages() -> None:
    out_dir = FERN_PAGES / "rust_api"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = collect_rust_pages()

    index_lines = [
        "# Rust API Documentation",
        "",
        "These pages are generated from the Rust crate sources under `rust/cuvs/src`.",
        "",
    ]
    for group, group_pages in group_rust_pages(pages).items():
        index_lines.extend([f"## {group}", ""])
        for page in group_pages:
            index_lines.append(
                f"- [`{page.module}`]({api_doc_url('rust_api', page.slug)})"
            )
        index_lines.append("")
    write_page(out_dir / "index.md", index_lines)

    for page in pages:
        lines = [
            *api_frontmatter(api_page_route("rust_api", page.slug)),
            f"# {page.title}",
            "",
            f"_Rust module: `{page.module}`_",
            "",
            f"_Source: `{page.source}`_",
            "",
        ]
        if page.module_doc:
            lines.extend(render_markdown_doc(page.module_doc))
            lines.append("")
        for item in page.items:
            lines.extend(render_rust_item(item))
            lines.append("")
        write_page(
            out_dir / f"{api_page_route('rust_api', page.slug)}.md", lines
        )


def collect_rust_pages() -> list[RustPage]:
    pages: list[RustPage] = []
    for path in sorted(RUST_SOURCE_DIR.rglob("*.rs")):
        if path.name.endswith("_test.rs"):
            continue
        page = parse_rust_page(path)
        if page.module_doc or page.items:
            pages.append(page)
    pages.sort(key=lambda page: page.module)
    return pages


def parse_rust_page(path: Path) -> RustPage:
    lines = path.read_text(encoding="utf-8").splitlines()
    module = rust_module_name(path)
    items = parse_rust_top_level_items(lines, path)
    items_by_name = {
        item.name: item
        for item in items
        if item.kind in {"struct", "enum", "trait"}
    }

    for impl_type, methods, impl_line in parse_rust_impl_methods(lines, path):
        if not methods:
            continue
        target = items_by_name.get(impl_type)
        if target is not None:
            target.members.extend(methods)
        else:
            items.append(
                RustItem(
                    name=f"impl {impl_type}",
                    kind="impl",
                    signature=f"impl {impl_type}",
                    doc="",
                    source=path.relative_to(REPO_DIR).as_posix(),
                    line=impl_line,
                    members=methods,
                )
            )

    return RustPage(
        module=module,
        title=rust_title(module),
        slug=rust_slug(module),
        source=path.relative_to(REPO_DIR).as_posix(),
        module_doc=collect_rust_module_doc(lines),
        items=items,
    )


def parse_rust_top_level_items(lines: list[str], path: Path) -> list[RustItem]:
    items: list[RustItem] = []
    pending_doc: list[str] = []
    pending_attributes: list[str] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()

        if stripped.startswith("///"):
            pending_doc.append(stripped.removeprefix("///").lstrip())
            idx += 1
            continue
        if stripped.startswith("#["):
            pending_attributes.append(stripped)
            idx += 1
            continue
        if stripped.startswith("impl"):
            _, signature_end, has_block = collect_rust_signature(lines, idx)
            pending_doc = []
            pending_attributes = []
            idx = (
                skip_rust_block(lines, signature_end) + 1
                if has_block
                else signature_end + 1
            )
            continue
        if is_public_rust_declaration(stripped):
            signature, signature_end, has_block = collect_rust_signature(
                lines, idx
            )
            kind, name = parse_rust_item_kind_name(signature)
            if kind and name:
                items.append(
                    RustItem(
                        name=name,
                        kind=kind,
                        signature=signature,
                        doc=clean_rust_doc(pending_doc),
                        source=path.relative_to(REPO_DIR).as_posix(),
                        line=idx + 1,
                        attributes=pending_attributes,
                    )
                )
            pending_doc = []
            pending_attributes = []
            idx = (
                skip_rust_block(lines, signature_end) + 1
                if has_block
                else signature_end + 1
            )
            continue
        if stripped and not stripped.startswith("//"):
            pending_doc = []
            pending_attributes = []
        idx += 1
    return items


def parse_rust_impl_methods(
    lines: list[str], path: Path
) -> list[tuple[str, list[RustItem], int]]:
    impls: list[tuple[str, list[RustItem], int]] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped.startswith("impl"):
            idx += 1
            continue
        signature, signature_end, has_block = collect_rust_signature(
            lines, idx
        )
        if not has_block:
            idx = signature_end + 1
            continue
        block_end = skip_rust_block(lines, signature_end)
        impl_type = parse_rust_impl_type(signature)
        if impl_type:
            methods = parse_rust_methods_in_impl(
                lines, signature_end + 1, block_end, path
            )
            impls.append((impl_type, methods, idx + 1))
        idx = block_end + 1
    return impls


def parse_rust_methods_in_impl(
    lines: list[str],
    start: int,
    end: int,
    path: Path,
) -> list[RustItem]:
    methods: list[RustItem] = []
    pending_doc: list[str] = []
    pending_attributes: list[str] = []
    idx = start
    depth = 1
    while idx < end:
        stripped = lines[idx].strip()
        if depth == 1 and stripped.startswith("///"):
            pending_doc.append(stripped.removeprefix("///").lstrip())
            idx += 1
            continue
        if depth == 1 and stripped.startswith("#["):
            pending_attributes.append(stripped)
            idx += 1
            continue
        if depth == 1 and is_public_rust_function(stripped):
            signature, signature_end, has_block = collect_rust_signature(
                lines, idx
            )
            name = parse_rust_function_name(signature)
            if name:
                methods.append(
                    RustItem(
                        name=name,
                        kind="method",
                        signature=signature,
                        doc=clean_rust_doc(pending_doc),
                        source=path.relative_to(REPO_DIR).as_posix(),
                        line=idx + 1,
                        attributes=pending_attributes,
                    )
                )
            pending_doc = []
            pending_attributes = []
            idx = (
                skip_rust_block(lines, signature_end) + 1
                if has_block
                else signature_end + 1
            )
            continue
        if depth == 1 and stripped and not stripped.startswith("//"):
            pending_doc = []
            pending_attributes = []
        depth += rust_brace_delta(lines[idx])
        idx += 1
    return methods


def collect_rust_module_doc(lines: list[str]) -> str:
    doc: list[str] = []
    saw_module_doc = False
    in_header_comment = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("/*"):
            in_header_comment = True
        if in_header_comment:
            if stripped.endswith("*/"):
                in_header_comment = False
            continue
        if stripped.startswith("//!"):
            doc.append(stripped.removeprefix("//!").lstrip())
            saw_module_doc = True
            continue
        if not stripped and not saw_module_doc:
            continue
        if saw_module_doc and not stripped:
            doc.append("")
            continue
        if saw_module_doc:
            break
        if stripped.startswith("//"):
            continue
        break
    return clean_rust_doc(doc)


def is_public_rust_declaration(stripped: str) -> bool:
    if not stripped.startswith("pub ") or stripped.startswith("pub(crate)"):
        return False
    return bool(
        re.match(
            r"pub\s+(?:unsafe\s+|async\s+)?(?:fn|struct|enum|trait|type|use|mod)\b",
            stripped,
        )
    )


def is_public_rust_function(stripped: str) -> bool:
    if not stripped.startswith("pub ") or stripped.startswith("pub(crate)"):
        return False
    return bool(re.match(r"pub\s+(?:unsafe\s+|async\s+)?fn\b", stripped))


def collect_rust_signature(
    lines: list[str], start: int
) -> tuple[str, int, bool]:
    chunks: list[str] = []
    paren_depth = 0
    bracket_depth = 0
    angle_depth = 0
    idx = start
    while idx < len(lines):
        stripped = lines[idx].strip()
        chunks.append(stripped)
        for char_idx, char in enumerate(stripped):
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(paren_depth - 1, 0)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(bracket_depth - 1, 0)
            elif char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(angle_depth - 1, 0)
            elif (
                char in "{;"
                and paren_depth == 0
                and bracket_depth == 0
                and angle_depth == 0
            ):
                prefix = stripped[: char_idx + 1].rstrip()
                chunks[-1] = prefix
                signature = normalize_rust_signature("\n".join(chunks))
                return signature, idx, char == "{"
        idx += 1
    return normalize_rust_signature("\n".join(chunks)), idx - 1, False


def normalize_rust_signature(signature: str) -> str:
    signature = "\n".join(line.rstrip() for line in signature.splitlines())
    signature = re.sub(r"\s+\{$", " {", signature)
    if signature.endswith("{"):
        signature = f"{signature[:-1].rstrip()} {{ ... }}"
    return signature.strip()


def skip_rust_block(lines: list[str], start: int) -> int:
    depth = 0
    entered = False
    idx = start
    while idx < len(lines):
        line = strip_rust_line_for_braces(lines[idx])
        for char in line:
            if char == "{":
                depth += 1
                entered = True
            elif char == "}":
                depth = max(depth - 1, 0)
                if entered and depth == 0:
                    return idx
        idx += 1
    return min(start, len(lines) - 1)


def rust_brace_delta(line: str) -> int:
    clean = strip_rust_line_for_braces(line)
    return clean.count("{") - clean.count("}")


def strip_rust_line_for_braces(line: str) -> str:
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(?:\\.|[^'\\])'", "''", line)
    return line.split("//", 1)[0]


def parse_rust_item_kind_name(signature: str) -> tuple[str, str]:
    first_line = signature.splitlines()[0].strip()
    match = re.match(
        r"pub\s+(?P<kind>struct|enum|trait|type|mod)\s+(?P<name>[A-Za-z_]\w*)",
        first_line,
    )
    if match:
        return match.group("kind"), match.group("name")
    if re.match(r"pub\s+(?:unsafe\s+|async\s+)?fn\b", first_line):
        return "function", parse_rust_function_name(signature)
    use_match = re.match(r"pub\s+use\s+(.+?);?$", signature.replace("\n", " "))
    if use_match:
        return "reexport", use_match.group(1).strip()
    return "", ""


def parse_rust_function_name(signature: str) -> str:
    match = re.search(r"\bfn\s+([A-Za-z_]\w*)", signature)
    return match.group(1) if match else ""


def parse_rust_impl_type(signature: str) -> str:
    signature = signature.split("{", 1)[0].replace("\n", " ").strip()
    signature = re.sub(r"^impl\s*<[^>]+>\s*", "impl ", signature)
    if " for " in signature:
        candidate = signature.rsplit(" for ", 1)[1].strip()
    else:
        candidate = re.sub(r"^impl\s+", "", signature).strip()
    candidate = re.sub(r"<.*", "", candidate).strip()
    candidate = candidate.split("::")[-1].strip()
    match = re.match(r"([A-Za-z_]\w*)", candidate)
    return match.group(1) if match else ""


def clean_rust_doc(lines: list[str]) -> str:
    return "\n".join(line.rstrip() for line in trim_blank_lines(lines)).strip()


def render_rust_item(item: RustItem) -> list[str]:
    lines = [f"## {heading_text(item.name)}", ""]
    signature_lines = (
        [*item.attributes, item.signature]
        if item.attributes
        else [item.signature]
    )
    lines.extend(["```rust", "\n".join(signature_lines), "```", ""])
    if item.doc:
        lines.extend(render_markdown_doc(item.doc))
        lines.append("")
    if item.members:
        lines.extend(["**Methods**", "", "| Name | Source |", "| --- | --- |"])
        for member in item.members:
            lines.append(
                f"| `{escape_code(member.name)}` | `{escape_code(f'{member.source}:{member.line}')}` |"
            )
        lines.append("")
        for member in item.members:
            lines.extend([f"### {heading_text(member.name)}", ""])
            member_signature = (
                [*member.attributes, member.signature]
                if member.attributes
                else [member.signature]
            )
            lines.extend(["```rust", "\n".join(member_signature), "```", ""])
            if member.doc:
                lines.extend(render_markdown_doc(member.doc))
                lines.append("")
            lines.extend([f"_Source: `{member.source}:{member.line}`_", ""])
    lines.extend([f"_Source: `{item.source}:{item.line}`_", ""])
    return trim_blank_lines(lines)


def group_rust_pages(pages: list[RustPage]) -> dict[str, list[RustPage]]:
    grouped: dict[str, list[RustPage]] = defaultdict(list)
    for page in pages:
        grouped[rust_group(page.module)].append(page)
    return dict(sorted(grouped.items()))


def rust_group(module: str) -> str:
    if "::cluster" in module:
        return "Cluster"
    if "::distance" in module:
        return "Distance"
    if any(
        token in module
        for token in (
            "::brute_force",
            "::cagra",
            "::ivf_flat",
            "::ivf_pq",
            "::vamana",
        )
    ):
        return "Nearest Neighbors"
    return "Core"


def rust_module_name(path: Path) -> str:
    rel = path.relative_to(RUST_SOURCE_DIR)
    if rel.name == "lib.rs":
        parts: tuple[str, ...] = ()
    elif rel.name == "mod.rs":
        parts = rel.parent.parts
    else:
        parts = rel.with_suffix("").parts
    return "::".join(("cuvs", *parts))


def rust_title(module: str) -> str:
    if module == "cuvs":
        return "cuVS Rust Crate"
    return f"{humanize_slug(module.removeprefix('cuvs::').replace('::', '-').replace('_', '-'))} Module"


def rust_slug(module: str) -> str:
    return slugify(module.replace("::", "-"))


def generate_go_api_pages() -> None:
    out_dir = FERN_PAGES / "go_api"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = collect_go_pages()

    index_lines = [
        "# Go API Documentation",
        "",
        "These pages are generated from the Go source files under `go`.",
        "",
    ]
    for page in pages:
        index_lines.append(
            f"- [`{page.package}`]({api_doc_url('go_api', page.slug)})"
        )
    write_page(out_dir / "index.md", index_lines)

    for page in pages:
        lines = [
            *api_frontmatter(api_page_route("go_api", page.slug)),
            f"# {page.title}",
            "",
            f"_Go package: `{page.package}`_",
            "",
            f"_Sources: `{escape_code(', '.join(page.sources))}`_",
            "",
        ]
        for section, items in group_go_items(page.items).items():
            lines.extend([f"## {section}", ""])
            for item in items:
                lines.extend(render_go_item(item))
                lines.append("")
        write_page(
            out_dir / f"{api_page_route('go_api', page.slug)}.md", lines
        )


def collect_go_pages() -> list[GoPage]:
    by_package: dict[str, list[GoItem]] = defaultdict(list)
    sources: dict[str, set[str]] = defaultdict(set)
    for path in sorted(GO_SOURCE_DIR.rglob("*.go")):
        if path.name.endswith("_test.go"):
            continue
        package, items = parse_go_file(path)
        if not package:
            continue
        by_package[package].extend(items)
        sources[package].add(path.parent.relative_to(REPO_DIR).as_posix())

    pages = [
        GoPage(
            package=package,
            title=f"{go_package_title(package)} Package",
            slug=go_slug(package),
            sources=sorted(sources[package]),
            items=sorted(
                items,
                key=lambda item: (
                    go_kind_order(item.kind),
                    item.name,
                    item.line,
                ),
            ),
        )
        for package, items in by_package.items()
        if items
    ]
    pages.sort(key=lambda page: page.package)
    return pages


def parse_go_file(path: Path) -> tuple[str, list[GoItem]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    package = ""
    items: list[GoItem] = []
    pending_doc: list[str] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not package:
            package_match = re.match(r"package\s+([A-Za-z_]\w*)", stripped)
            if package_match:
                package = package_match.group(1)
        if stripped.startswith("//"):
            pending_doc.append(stripped.removeprefix("//").lstrip())
            idx += 1
            continue
        if stripped.startswith("/*"):
            block, end_idx = collect_go_block_comment(lines, idx)
            pending_doc.extend(block)
            idx = end_idx + 1
            continue
        if is_go_declaration(stripped):
            signature, end_idx = collect_go_declaration(lines, idx)
            kind, name, receiver = parse_go_item_kind_name(
                signature, pending_doc
            )
            if name and is_go_exported_item(kind, name, signature):
                items.append(
                    GoItem(
                        name=name,
                        kind=kind,
                        signature=signature,
                        doc=clean_go_doc(pending_doc),
                        source=path.relative_to(REPO_DIR).as_posix(),
                        line=idx + 1,
                        receiver=receiver,
                    )
                )
            pending_doc = []
            idx = end_idx + 1
            continue
        if stripped:
            pending_doc = []
        idx += 1
    return package, items


def collect_go_block_comment(
    lines: list[str], start: int
) -> tuple[list[str], int]:
    body: list[str] = []
    idx = start
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if idx == start:
            stripped = stripped.removeprefix("/*")
        if "*/" in stripped:
            body.append(stripped.split("*/", 1)[0].strip(" *"))
            return body, idx
        body.append(stripped.strip(" *"))
        idx += 1
    return body, idx - 1


def is_go_declaration(stripped: str) -> bool:
    return bool(re.match(r"(?:func|type|const|var)\b", stripped))


def collect_go_declaration(lines: list[str], start: int) -> tuple[str, int]:
    first = lines[start].strip()
    if first.startswith("func "):
        return collect_go_function_signature(lines, start)
    if re.match(r"(?:const|var|type)\s*\(", first):
        return collect_go_paren_block(lines, start)
    if (
        " struct {" in first
        or first.endswith(" struct {")
        or " interface {" in first
    ):
        signature, end_idx = collect_go_brace_block(lines, start)
        return compact_go_type_signature(signature), end_idx
    return first, start


def collect_go_function_signature(
    lines: list[str], start: int
) -> tuple[str, int]:
    chunks: list[str] = []
    paren_depth = 0
    bracket_depth = 0
    idx = start
    while idx < len(lines):
        stripped = lines[idx].strip()
        chunks.append(stripped)
        for char_idx, char in enumerate(stripped):
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(paren_depth - 1, 0)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(bracket_depth - 1, 0)
            elif char == "{" and paren_depth == 0 and bracket_depth == 0:
                chunks[-1] = stripped[:char_idx].rstrip()
                return normalize_go_signature("\n".join(chunks)), idx
        idx += 1
    return normalize_go_signature("\n".join(chunks)), idx - 1


def collect_go_paren_block(lines: list[str], start: int) -> tuple[str, int]:
    chunks = []
    depth = 0
    idx = start
    while idx < len(lines):
        line = lines[idx].rstrip()
        chunks.append(line.strip())
        depth += line.count("(") - line.count(")")
        if idx > start and depth <= 0:
            break
        idx += 1
    return "\n".join(chunks), idx


def collect_go_brace_block(lines: list[str], start: int) -> tuple[str, int]:
    chunks = []
    depth = 0
    idx = start
    while idx < len(lines):
        line = lines[idx].rstrip()
        chunks.append(line.strip())
        depth += line.count("{") - line.count("}")
        if idx > start and depth <= 0:
            break
        idx += 1
    return "\n".join(chunks), idx


def compact_go_type_signature(signature: str) -> str:
    first_line = signature.splitlines()[0]
    if first_line.endswith("{"):
        return f"{first_line[:-1].rstrip()} {{ ... }}"
    return signature


def normalize_go_signature(signature: str) -> str:
    return "\n".join(line.rstrip() for line in signature.splitlines()).strip()


def parse_go_item_kind_name(
    signature: str, doc_lines: list[str]
) -> tuple[str, str, str]:
    first_line = signature.splitlines()[0].strip()
    func_match = re.match(
        r"func\s+(?:\((?P<receiver>[^)]+)\)\s*)?(?P<name>[A-Za-z_]\w*)",
        first_line,
    )
    if func_match:
        receiver = go_receiver_type(func_match.group("receiver") or "")
        name = func_match.group("name")
        if receiver:
            return "method", f"{receiver}.{name}", receiver
        return "function", name, ""
    type_match = re.match(r"type\s+(?P<name>[A-Za-z_]\w*)", first_line)
    if type_match:
        return "type", type_match.group("name"), ""
    var_match = re.match(r"var\s+(?P<name>[A-Za-z_]\w*)", first_line)
    if var_match:
        return "variable", var_match.group("name"), ""
    const_match = re.match(r"const\s+(?P<name>[A-Za-z_]\w*)", first_line)
    if const_match:
        return "constant", const_match.group("name"), ""
    if first_line == "const (":
        return (
            "constant",
            go_block_heading(
                doc_lines, go_block_export_name(signature, "Constants")
            ),
            "",
        )
    if first_line == "var (":
        return (
            "variable",
            go_block_heading(
                doc_lines, go_block_export_name(signature, "Variables")
            ),
            "",
        )
    return "", "", ""


def go_receiver_type(receiver: str) -> str:
    receiver = receiver.strip()
    if not receiver:
        return ""
    parts = receiver.split()
    candidate = parts[-1] if parts else receiver
    candidate = candidate.lstrip("*")
    candidate = re.sub(r"\[.*", "", candidate)
    return candidate


def go_block_heading(doc_lines: list[str], fallback: str) -> str:
    doc = clean_go_doc(doc_lines)
    first = doc.splitlines()[0] if doc else ""
    if first and len(first) <= 64:
        return humanize_slug(slugify(first))
    return fallback


def go_block_export_name(signature: str, fallback: str) -> str:
    for line in signature.splitlines()[1:]:
        match = re.match(r"([A-Z]\w*)\s+([A-Z]\w*)\b", line.strip())
        if match:
            return f"{match.group(2)} {fallback}"
        match = re.match(r"([A-Z]\w*)\b", line.strip())
        if match:
            return f"{match.group(1)} {fallback}"
    return fallback


def is_go_exported_item(kind: str, name: str, signature: str) -> bool:
    if kind in {"constant", "variable"} and name in {"Constants", "Variables"}:
        return go_block_has_exported_name(signature)
    simple_name = name.split(".")[-1]
    return bool(simple_name and simple_name[0].isupper())


def go_block_has_exported_name(signature: str) -> bool:
    for line in signature.splitlines()[1:]:
        match = re.match(r"([A-Za-z_]\w*)", line.strip())
        if match and match.group(1)[0].isupper():
            return True
    return False


def clean_go_doc(lines: list[str]) -> str:
    return "\n".join(line.rstrip() for line in trim_blank_lines(lines)).strip()


def group_go_items(items: list[GoItem]) -> dict[str, list[GoItem]]:
    labels = {
        "constant": "Constants",
        "variable": "Variables",
        "type": "Types",
        "function": "Functions",
        "method": "Methods",
    }
    grouped: dict[str, list[GoItem]] = defaultdict(list)
    for item in items:
        grouped[labels.get(item.kind, "Other")].append(item)
    return {
        labels[kind]: grouped[labels[kind]]
        for kind in ["constant", "variable", "type", "function", "method"]
        if grouped.get(labels[kind])
    }


def go_kind_order(kind: str) -> int:
    return {
        "constant": 0,
        "variable": 1,
        "type": 2,
        "function": 3,
        "method": 4,
    }.get(kind, 9)


def render_go_item(item: GoItem) -> list[str]:
    lines = [
        f"### {heading_text(item.name)}",
        "",
        "```go",
        item.signature,
        "```",
        "",
    ]
    if item.doc:
        lines.extend(render_markdown_doc(item.doc))
        lines.append("")
    lines.extend([f"_Source: `{item.source}:{item.line}`_", ""])
    return trim_blank_lines(lines)


def go_package_title(package: str) -> str:
    if package == "cuvs":
        return "cuVS"
    return humanize_slug(package.replace("_", "-"))


def go_slug(package: str) -> str:
    return slugify(package)


def update_api_navigation() -> None:
    docs_yml = REPO_DIR / "fern" / "docs.yml"
    text = docs_yml.read_text(encoding="utf-8")
    start_marker = '  - section: "API Reference"\n'
    end_marker = '  - section: "Advanced Topics"\n'
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start == -1 or end == -1:
        raise RuntimeError(
            "Could not find API Reference navigation block in fern/docs.yml"
        )

    lines = [
        '  - section: "API Reference"',
        '    path: "./pages/api_docs.md"',
        "    contents:",
    ]
    for title, directory, _, _, _ in API_NAV_SECTIONS:
        lines.extend(
            [
                f'      - section: "{title}"',
                f'        path: "./pages/{directory}/index.md"',
                "        contents:",
            ]
        )
        for slug in read_api_index_slugs(FERN_PAGES / directory / "index.md"):
            lines.extend(
                [
                    f'          - page: "{api_nav_page_title(directory, slug)}"',
                    f'            path: "./pages/{directory}/{api_page_route(directory, slug)}.md"',
                ]
            )

    replacement = "\n".join(lines) + "\n"
    docs_yml.write_text(
        text[:start] + replacement + text[end:], encoding="utf-8"
    )
    print("Updated fern/docs.yml API Reference navigation")


def read_api_index_slugs(index_path: Path) -> list[str]:
    slugs: list[str] = []
    route_prefix = api_route_prefix(index_path.parent.name)
    for line in index_path.read_text(encoding="utf-8").splitlines():
        match = re.search(
            rf"\]\((?:\./|/api-reference/{re.escape(route_prefix)}-)([^)#]+)",
            line,
        )
        if match:
            slugs.append(match.group(1).removesuffix(".md"))
    return slugs


def api_doc_url(directory: str, slug: str) -> str:
    return f"/api-reference/{api_page_route(directory, slug)}"


def api_page_route(directory: str, slug: str) -> str:
    return f"{api_route_prefix(directory)}-{slug}"


def api_frontmatter(route: str) -> list[str]:
    return ["---", f"slug: api-reference/{route}", "---", ""]


def api_nav_page_title(directory: str, slug: str) -> str:
    return api_route_title(slug)


def api_section_route(directory: str) -> str:
    for _, api_directory, route, _, _ in API_NAV_SECTIONS:
        if api_directory == directory:
            return route
    raise ValueError(f"Unknown API docs directory: {directory}")


def api_title_prefix(directory: str) -> str:
    for _, api_directory, _, title_prefix, _ in API_NAV_SECTIONS:
        if api_directory == directory:
            return title_prefix
    raise ValueError(f"Unknown API docs directory: {directory}")


def api_route_prefix(directory: str) -> str:
    for _, api_directory, _, _, route_prefix in API_NAV_SECTIONS:
        if api_directory == directory:
            return route_prefix
    raise ValueError(f"Unknown API docs directory: {directory}")


def api_route_title(slug: str) -> str:
    replacements = {
        "api": "API",
        "cpp": "C++",
        "cuvs": "cuVS",
        "gpu": "GPU",
        "hnsw": "HNSW",
        "ivf": "IVF",
        "kmeans": "Kmeans",
        "mg": "Multi GPU",
        "nn": "NN",
        "pca": "PCA",
        "pq": "PQ",
        "vq": "VQ",
    }
    words = []
    for word in slug.split("-"):
        words.append(replacements.get(word, word.capitalize()))
    return " ".join(words).replace('"', '\\"')


def render_markdown_doc(doc: str) -> list[str]:
    lines: list[str] = []
    in_code = False
    for raw_line in doc.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            lines.append(line)
            in_code = not in_code
            continue
        if in_code:
            lines.append(line)
        else:
            heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
            if heading_match:
                level = min(len(heading_match.group(1)) + 3, 6)
                lines.append(
                    f"{'#' * level} {escape_text(heading_match.group(2))}"
                )
            else:
                lines.append(escape_text(line))
    return trim_blank_lines(lines)


def source_line(entry: DoxygenEntry) -> str:
    return f"_Source: `{entry.source}:{entry.line}`_"


def render_param_table(
    params: list[dict[str, str]],
    include_direction: bool,
    symbol_links: dict[str, str] | None = None,
) -> list[str]:
    if symbol_links is None:
        symbol_links = {}
    headers = ["Name"]
    if include_direction:
        headers.append("Direction")
    headers.extend(["Type", "Description"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for param in params:
        row = [f"`{escape_code(param.get('name', ''))}`"]
        if include_direction:
            row.append(escape_text(param.get("direction", "")))
        row.append(render_type_reference(param.get("type", ""), symbol_links))
        description = param.get("description", "")
        if param.get("default"):
            description = (
                f"{description} Default: `{param['default']}`.".strip()
            )
        row.append(render_table_description(description))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_type_reference(value: str, symbol_links: dict[str, str]) -> str:
    rendered = f"`{escape_code(value)}`"
    link = find_native_symbol_link(value, symbol_links)
    return f"[{rendered}]({link})" if link else rendered


def find_native_symbol_link(
    value: str, symbol_links: dict[str, str]
) -> str | None:
    if not value or not symbol_links:
        return None
    matches = [
        (len(name), name, link)
        for name, link in symbol_links.items()
        if native_symbol_occurs(value, name)
    ]
    if not matches:
        return None
    return max(matches)[2]


def native_symbol_occurs(value: str, name: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", value))


def symbol_anchor_line(name: str) -> str:
    return f'<a id="{symbol_anchor(name)}"></a>'


def symbol_anchor(name: str) -> str:
    return slugify(name)


def native_page_slug(source: str) -> str:
    source = re.sub(r"^(?:c|cpp)/include/cuvs/", "", source)
    source = re.sub(r"\.(?:h|hpp|cuh)$", "", source)
    return slugify(source)


def native_page_title(source: str) -> str:
    source = re.sub(r"^(?:c|cpp)/include/cuvs/", "", source)
    source = re.sub(r"\.(?:h|hpp|cuh)$", "", source)
    parts = source.split("/")
    leaf = parts[-1]
    title = humanize_slug(leaf.replace("_", "-"))
    replacements = {
        "C Api": "C API",
        "Ivf": "IVF",
        "Pq": "PQ",
        "Pca": "PCA",
        "Hnsw": "HNSW",
        "Nn": "NN",
        "Mg": "Multi-GPU",
        "Kmeans": "K-Means",
    }
    for old, new in replacements.items():
        title = title.replace(old, new)
    if len(parts) > 1 and parts[-2] not in {"core"}:
        category = humanize_slug(parts[-2].replace("_", "-"))
        if title.lower() not in category.lower():
            title = f"{title}"
    return title


def humanize_slug(value: str) -> str:
    return " ".join(
        word.capitalize() for word in re.split(r"[-_/]+", value) if word
    )


def java_slug(klass: JavaClass) -> str:
    return slugify(f"{klass.package}.{klass.name}")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "api"


def indentation(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def split_top_level(text: str, sep: str) -> list[str]:
    chunks: list[str] = []
    depth = {"(": 0, "[": 0, "{": 0, "<": 0}
    start = 0
    for idx, char in enumerate(text):
        update_depth(depth, char)
        if char == sep and not any(depth.values()):
            chunks.append(text[start:idx])
            start = idx + 1
    chunks.append(text[start:])
    return chunks


def update_depth(depth: dict[str, int], char: str) -> None:
    if char in "([{<":
        depth[char] += 1
    elif char == ")":
        depth["("] = max(depth["("] - 1, 0)
    elif char == "]":
        depth["["] = max(depth["["] - 1, 0)
    elif char == "}":
        depth["{"] = max(depth["{"] - 1, 0)
    elif char == ">":
        depth["<"] = max(depth["<"] - 1, 0)


def escape_code(value: str) -> str:
    return (
        str(value).replace("|", "\\|").replace("`", "\\`").replace("\n", " ")
    )


def escape_text(value: str) -> str:
    escaped = (
        str(value)
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", " ")
    )
    return restore_math_placeholders(escaped)


def restore_math_placeholders(value: str) -> str:
    return MATH_PLACEHOLDER_RE.sub(
        lambda match: bytes.fromhex(match.group(1)).decode("utf-8"),
        value,
    )


def heading_text(value: str) -> str:
    return escape_text(value).replace("`", "\\`").strip() or "API"


def trim_blank_lines(lines: list[str]) -> list[str]:
    lines = list(lines)
    while lines and not str(lines[0]).strip():
        lines.pop(0)
    while lines and not str(lines[-1]).strip():
        lines.pop()
    return lines


def write_page(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(trim_blank_lines(lines)).rstrip() + "\n", encoding="utf-8"
    )
    print(f"Wrote {path.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    raise SystemExit(main())
