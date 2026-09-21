"""CASP submission identity, read from the environment.

The group code and contact email are per-account, so they are not checked in.
Set them once in your shell:

    export CASP_GROUP_ID=NNNN-NNNN-NNNN   # issued when you register a group
    export CASP_EMAIL=you@example.edu     # the address CASP has on file

The group code lands in the AUTHOR record of every submission file; the email
is what the submission endpoint authenticates against. Getting either wrong
means the prediction center silently files your models under someone else's
group, so both are required rather than defaulted.
"""

import os
import re
import sys

GROUP_ID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}$")


def group_id(required: bool = True) -> str:
    """Return $CASP_GROUP_ID. Empty string if unset and ``required`` is False."""
    value = os.environ.get("CASP_GROUP_ID", "").strip()
    if not value:
        if required:
            sys.exit(
                "CASP_GROUP_ID is not set. Export your CASP group code first:\n"
                "    export CASP_GROUP_ID=NNNN-NNNN-NNNN"
            )
        return ""
    if not GROUP_ID_RE.match(value):
        sys.exit(f"CASP_GROUP_ID={value!r} is not in NNNN-NNNN-NNNN form")
    return value


def email(required: bool = True) -> str:
    """Return $CASP_EMAIL. Empty string if unset and ``required`` is False."""
    value = os.environ.get("CASP_EMAIL", "").strip()
    if not value:
        if required:
            sys.exit(
                "CASP_EMAIL is not set. Export the address registered with CASP:\n"
                "    export CASP_EMAIL=you@example.edu"
            )
        return ""
    if "@" not in value:
        sys.exit(f"CASP_EMAIL={value!r} does not look like an address")
    return value
