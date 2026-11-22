from __future__ import annotations

import typing as t
from collections import defaultdict

from sqlglot import exp
from sqlglot.helper import first
from sqlglot.lineage import Node
from sqlglot.lineage import lineage as sqlglot_lineage
from sqlglot.optimizer import Scope, build_scope, qualify

from sqlmesh.core.dialect import normalize_mapping_schema, normalize_model_name

if t.TYPE_CHECKING:
    from sqlglot.schema import MappingSchema

    from sqlmesh.core.context import Context
    from sqlmesh.core.model import Model


CACHE: t.Dict[str, t.Tuple[int, exp.Expr, Scope, MappingSchema]] = {}


def lineage(
    column: str | exp.Column,
    model: Model,
    trim_selects: bool = True,
    **kwargs: t.Any,
) -> Node:
    scope: t.Optional[Scope]
    cached = CACHE.get(model.name)
    if cached and cached[0] == id(model):
        _, query, scope, schema = cached
    else:
        query = t.cast(exp.Query, model.render_query_or_raise().copy())

        if model.managed_columns:
            query = query.select(
                *(
                    exp.alias_(exp.cast(exp.Null(), to=col_type), col)
                    for col, col_type in model.managed_columns.items()
                    if col not in query.named_selects
                ),
                copy=False,
            )

        schema = normalize_mapping_schema(model.mapping_schema, dialect=model.dialect)
        query = qualify.qualify(
            query,
            dialect=model.dialect,
            schema=schema,
            **{"validate_qualify_columns": False, **kwargs},
        )

        scope = build_scope(query)

        if scope:
            CACHE[model.name] = (id(model), query, scope, schema)

    return sqlglot_lineage(
        column,
        sql=query,
        schema=schema,
        scope=scope,
        trim_selects=trim_selects,
        dialect=model.dialect,
        copy=False,
    )


def column_dependencies(
    context: Context, model_name: str, column: str | exp.Column
) -> t.Dict[str, t.Set[str]]:
    model = context.get_model(model_name)
    parents = defaultdict(set)

    for node in lineage(column, model, trim_selects=False).walk():
        if node.downstream:
            continue

        table = node.expression.find(exp.Table)
        if table:
            name = normalize_model_name(
                table, default_catalog=context.default_catalog, dialect=model.dialect
            )
            parents[name].add(exp.to_column(node.name).name)
    return dict(parents)


def column_description(
    context: Context, model_name: str, column: str, quote_column: bool = False
) -> t.Optional[str]:
    """Returns a column's description, inferring if needed."""
    model = context.get_model(model_name)

    if not model:
        return None

    if column in model.column_descriptions:
        return model.column_descriptions[column]

    dependencies = column_dependencies(context, model_name, exp.column(column, quoted=quote_column))

    if len(dependencies) != 1:
        return None

    parent, columns = first(dependencies.items())

    if len(columns) != 1:
        return None

    return column_description(context, parent, first(columns))
