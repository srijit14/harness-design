"""Checks the checkpoint can run. Each is check(answer, evidence, adapter, context, **options) -> (flags, parsed_ok).
`context` holds per-task inputs a check may need (e.g. the adapter's catalog of the environment).
A version turns checks on by name in its config (configs/versions/*.yaml)."""

from hilx.checks import similar_alternatives, unsourced_values

CHECKS = {
    unsourced_values.NAME: unsourced_values.check,
    similar_alternatives.NAME: similar_alternatives.check,
}
