import re
from typing import Optional


_RE_SPLIT = re.compile(r"[;,/|]+")
_RE_WS = re.compile(r"\s+")
_RE_CYR = re.compile(r"[А-Яа-яІіЇїЄє]")


def _clean(s: str) -> str:
    return _RE_WS.sub(" ", s.strip())


def _norm(s: str) -> str:
    return _clean(s).casefold()


def _has_cyrillic(s: str) -> bool:
    return bool(_RE_CYR.search(s or ""))


_COLOR_MAP = {
    "blue": "blue",
    "dark blue": "blue",
    "light blue": "blue",
    "navy": "blue",
    "azure": "blue",
    "синій": "blue",
    "синя": "blue",
    "сині": "blue",
    "голубий": "blue",
    "блакитний": "blue",
    "red": "red",
    "dark red": "red",
    "light red": "red",
    "червоний": "red",
    "сіра": "gray",
    "сірий": "gray",
    "серый": "gray",
    "gray": "gray",
    "grey": "gray",
    "green": "green",
    "зелений": "green",
    "black": "black",
    "чорний": "black",
    "white": "white",
    "білий": "white",
    "brown": "brown",
    "коричневий": "brown",
    "yellow": "yellow",
    "жовтий": "yellow",
    "unknown": "unknown",
    "n/a": "unknown",
    "none": "unknown",
    "невідомий": "unknown",
    "невідомо": "unknown",
}

_LABEL_MAP = {
    "warehouse": "warehouse",
    "storage": "warehouse",
    "storage facility": "warehouse",
    "depot": "warehouse",
    "склад": "warehouse",
    "складське приміщення": "warehouse",
    "building": "building",
    "structure": "building",
    "будівля": "building",
    "car": "car",
    "vehicle": "car",
    "auto": "car",
    "авто": "car",
    "машина": "car",
    "легковик": "car",
    "truck": "truck",
    "lorry": "truck",
    "вантажівка": "truck",
    "bus": "bus",
    "автобус": "bus",
    "train": "train",
    "поїзд": "train",
    "ship": "ship",
    "boat": "ship",
    "судно": "ship",
    "корабель": "ship",
    "object": "object",
}

_UK_COLOR = {
    "blue": "синій",
    "red": "червоний",
    "gray": "сірий",
    "green": "зелений",
    "black": "чорний",
    "white": "білий",
    "brown": "коричневий",
    "yellow": "жовтий",
    "unknown": "невідомий",
}

_UK_LABEL = {
    "warehouse": "склад",
    "building": "будівля",
    "car": "авто",
    "truck": "вантажівка",
    "bus": "автобус",
    "train": "поїзд",
    "ship": "судно",
    "object": "обʼєкт",
}


def _color_by_contains(s: str) -> Optional[str]:
    if "blue" in s or "син" in s or "блак" in s or "голуб" in s:
        return "blue"
    if "red" in s or "черв" in s:
        return "red"
    if "gray" in s or "grey" in s or "сір" in s or "сер" in s:
        return "gray"
    if "green" in s or "зелен" in s:
        return "green"
    if "black" in s or "чорн" in s:
        return "black"
    if "white" in s or "біл" in s:
        return "white"
    if "brown" in s or "корич" in s:
        return "brown"
    if "yellow" in s or "жовт" in s:
        return "yellow"
    if "unknown" in s or "невідом" in s or "n/a" in s:
        return "unknown"
    return None


def normalize_roof_color(value: Optional[str], lang_hint: str = "auto") -> Optional[str]:
    if value is None:
        return None
    s = _norm(str(value))
    if not s:
        return None
    parts = [p.strip() for p in _RE_SPLIT.split(s) if p.strip()]
    if not parts:
        parts = [s]
    for p in parts:
        mapped = _COLOR_MAP.get(p)
        if mapped:
            return mapped
        mapped = _color_by_contains(p)
        if mapped:
            return mapped
    return None


def normalize_label(label: str) -> str:
    if label is None:
        return "object"
    s = _norm(str(label))
    if not s:
        return "object"
    return _LABEL_MAP.get(s, s)


def to_uk_color(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    norm = normalize_roof_color(value)
    if norm and norm in _UK_COLOR:
        return _UK_COLOR[norm]
    if _has_cyrillic(str(value)):
        return _clean(str(value))
    return None


def to_uk_label(label: str) -> Optional[str]:
    if label is None:
        return None
    norm = normalize_label(label)
    if norm in _UK_LABEL:
        return _UK_LABEL[norm]
    if _has_cyrillic(str(label)):
        return _clean(str(label))
    return norm
