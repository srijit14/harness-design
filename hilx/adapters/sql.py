"""Adapter for HiL-Bench SQL: Scale's SQL tools, the submit_sql answer, and literals in a SQL query."""

import csv
import re
import sqlite3
from pathlib import Path

import sqlglot
from sqlglot import exp

import hilx.scale  # noqa: F401  (puts Scale's code on the import path)
from sweagent.agent.agents import DefaultSQLAgent

from hilx.core.entities import Entity
from hilx.core.values import DATE, NUMBER_IN_TEXT, Value, is_trivial


class SQLAdapter:
    name = "sql"
    # Which source type each SQL tool's output is. Tools not listed (submit_sql) are not evidence.
    tool_sources = {
        "get_business_info": "given",
        "ask_human": "given",
        "get_database_info": "structure",
        "get_table_info": "structure",
        "get_column_info": "structure",
        "execute_sql": "data",
    }
    source_words = {
        "given": "the task, the business notes, the human's answers",
        "structure": "the schema",
        "data": "query results",
        "data_any": "any query result you saw",
    }
    # Name words too generic to make two columns in different tables look alike (similar_alternatives).
    plumbing_words = {"id", "ref", "name", "code", "key", "url", "date", "time", "year", "type", "number", "num"}

    def submission(self, action: str) -> str:
        return DefaultSQLAgent._extract_submission_body(action)

    def extract_values(self, sql: str) -> tuple[list[Value], bool]:
        """Literals in a SQL query. Falls back to a regex if parsing fails."""
        sql = re.sub(r"\bENDOFSQL\b", "", sql).strip().rstrip(";")
        try:
            trees = [t for t in sqlglot.parse(sql, read="sqlite") if t is not None]
        except sqlglot.errors.SqlglotError:  # ParseError and TokenError (an unparsable query crashed a v2 run)
            trees = []
        if not trees:
            found = [Value(m.group(1), "string", "") for m in re.finditer(r"'((?:[^']|'')*)'", sql)]
            found += [Value(m.group(0), "number", "") for m in NUMBER_IN_TEXT.finditer(re.sub(r"'(?:[^']|'')*'", "", sql))]
            return [v for v in found if not is_trivial(v)], False

        values, groups = [], {}
        for tree in trees:
            for lit in tree.find_all(exp.Literal):
                text = lit.this
                parent = lit.parent
                if lit.is_number:
                    if isinstance(parent, exp.Neg):
                        text, parent = "-" + text, parent.parent
                    kind = "number"
                else:
                    if isinstance(parent, (exp.Like, exp.ILike)):
                        text = text.strip("%")
                    kind = "date" if DATE.match(text) else "string"
                # CASE output labels ('Yes'/'No') are the agent's own wording, not a choice about the data.
                if isinstance(parent, exp.If) and lit is not parent.this or isinstance(parent, exp.Case) and lit is parent.args.get("default"):
                    continue
                group = None
                if isinstance(parent, exp.In):
                    group = groups.setdefault(id(parent), len(groups))
                context = (parent.sql(dialect="sqlite") if parent is not None else text)[:120]
                value = Value(text, kind, context, group)
                if not is_trivial(value):
                    values.append(value)
        return values, True

    # --- entities, for the "similar alternatives" check -------------------------------------------

    def catalog(self, task_dir: Path) -> list[Entity]:
        """Every table and column in the task's database, with the descriptions Scale's schema tools show."""
        db = next(task_dir.glob("*.sqlite"))
        tdesc = {r["table_name"].lower(): r["table_description"] for r in _csv(task_dir / "table_descriptions.csv")}
        out = []
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            for t in tables:
                out.append(Entity(t, None, tdesc.get(t.lower(), "")))
                cdesc = {r["original_column_name"]: r.get("combined_description", "") for r in _csv(task_dir / f"{t.lower()}.csv")}
                for (_, col, *_rest) in con.execute(f'PRAGMA table_info("{t}")'):
                    out.append(Entity(t, col, cdesc.get(col, "")))
        return out

    def entities_used(self, sql: str, catalog: list[Entity]) -> tuple[list[tuple[Entity, str]], bool]:
        """Tables and columns the query uses, each with a role: "content" (selected, filtered, grouped,
        aggregated) or "join" (only compared with another column, i.e. a join key). Returns (uses, parsed_ok)."""
        sql = re.sub(r"\bENDOFSQL\b", "", sql).strip().rstrip(";")
        try:
            trees = [t for t in sqlglot.parse(sql, read="sqlite") if t is not None]
        except sqlglot.errors.SqlglotError:  # ParseError and TokenError (an unparsable query crashed a v2 run)
            return [], False
        by_table = {e.container.lower(): e for e in catalog if e.name is None}
        by_col = {(e.container.lower(), e.name.lower()): e for e in catalog if e.name}
        uses: dict[Entity, str] = {}
        for tree in trees:
            aliases = {}
            for t in tree.find_all(exp.Table):
                if t.name.lower() in by_table:
                    aliases[(t.alias_or_name).lower()] = t.name.lower()
                    uses.setdefault(by_table[t.name.lower()], "content")
            in_query = set(aliases.values())
            for c in tree.find_all(exp.Column):
                col = c.name.lower()
                if c.table:
                    tables = [aliases.get(c.table.lower())]
                else:
                    tables = [t for t in in_query if (t, col) in by_col]
                    tables = tables if len(tables) == 1 else []
                e = by_col.get((tables[0], col)) if tables and tables[0] else None
                if e is None:
                    continue
                p = c.parent
                is_join = isinstance(p, exp.EQ) and isinstance(p.left, exp.Column) and isinstance(p.right, exp.Column)
                if is_join:
                    uses.setdefault(e, "join")
                else:
                    uses[e] = "content"
        return list(uses.items()), True


def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))
