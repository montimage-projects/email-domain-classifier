"""Shared parsing helpers for CSV-sourced fields.

The raw CEAS_08 ``urls`` column stores the strings ``'0'``/``'1'`` rather than
booleans, and processed datasets may carry a ``has_url`` flag in several
spellings. ``parse_url_flag`` is the single place that interprets them, so the
classifier, analyzer and processor cannot drift apart.
"""

# Strings that explicitly mean "no URL" (compared case-insensitively, after
# stripping). Blank and whitespace-only cells are covered by "".
_URL_FLAG_FALSE = frozenset({"", "0", "false", "no", "off"})


def parse_url_flag(value: object) -> bool:
    """Return whether a raw ``urls``/``has_url`` cell means "URL present".

    Strings: ``'0'``, ``'false'``, ``'no'``, ``'off'`` and blank/whitespace are
    false (case-insensitive); ``'1'``, ``'true'``, ``'yes'``, ``'on'`` and any
    other non-empty text (e.g. a real URL) are true. Non-string values fall
    back to ``bool(value)``.
    """
    if isinstance(value, str):
        return value.strip().lower() not in _URL_FLAG_FALSE
    return bool(value)
