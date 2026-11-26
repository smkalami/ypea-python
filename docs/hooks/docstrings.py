"""Griffe extension for the docs site: turn the plain text docstrings of ypea into Markdown.

The docstrings use underlined section titles and ``name : text`` parameter lines, a loose
form of the NumPy style. This extension rewrites them before rendering:

* an underlined title becomes a bold paragraph,
* ``name : text`` lines in a section (or followed by an indented line) become list items,
  and a short first line without a final period, followed by an indented capitalized
  description, is shown as a type,
* single ``*`` between two characters (``a*x1``) is escaped, so it is not read as emphasis.

Docstrings are rendered with ``docstring_style: null``, so this is the only parsing.
"""

from __future__ import annotations

import re

import griffe

TITLE_RULE = re.compile(r"^\s*(-{3,}|={3,})\s*$")
ITEM = re.compile(r"^([A-Za-z_*][\w, *]*?) : (.*)$")
STAR = re.compile(r"(?<=[\w)\]])\*(?=[\w(])")


def _escape_stars(line):
    # leave inline code alone
    parts = line.split("``")
    return "``".join(p if i % 2 else STAR.sub(r"\\*", p) for i, p in enumerate(parts))


def convert(text):
    lines = text.splitlines()
    out, in_section, item, loose = [], False, False, False
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and not line.startswith(" ") and TITLE_RULE.match(nxt):
            out += ["", f"**{line.strip()}**", ""]
            in_section, item = True, False
            i += 2
            continue
        # outside a section, "name : type" followed by an indented description is an item too
        m = ITEM.match(line) if in_section or item or nxt.startswith(" ") else None
        if m:
            if not (in_section or item):
                out.append("")
            in_section = True
            name, rest = m.groups()
            desc = nxt.strip()
            if loose:
                out.append("")  # the last item held a blank line, so this one must start a new block
            loose = False
            names = ", ".join(f"`{n.strip()}`" for n in name.split(","))
            if nxt.startswith(" ") and desc[:1].isupper() and len(rest) < 45 and not rest.endswith("."):
                out.append(f"- {names} ({rest}):")
            else:
                out.append(f"- {names}: {_escape_stars(rest)}")
            item = True
        elif item and line.startswith(" "):
            indent = len(line) - len(line.lstrip())
            out.append(" " * (4 + max(0, indent - 8)) + _escape_stars(line.strip()))
        else:
            if item and line.strip():
                # a paragraph after the list ends the section
                out.append("")
                item = in_section = False
            elif item:
                loose = True
            out.append(_escape_stars(line))
        i += 1
    return "\n".join(out)


class LooseNumpyDocstrings(griffe.Extension):
    def on_instance(self, *, obj, **kwargs):
        if obj.docstring is not None:
            obj.docstring.value = convert(obj.docstring.value)
