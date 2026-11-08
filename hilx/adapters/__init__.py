"""Domain adapters, by name. A version picks one with `adapter:` (default: sql)."""

from hilx.adapters.sql import SQLAdapter

ADAPTERS = {"sql": SQLAdapter()}
