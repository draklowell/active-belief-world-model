import json
from functools import partial

from jinja2 import Environment, PackageLoader, StrictUndefined

environment = Environment(
    loader=PackageLoader("prompts", "templates"),
    undefined=StrictUndefined,
    autoescape=False,
)
# Add a filter to convert Python objects to JSON strings without escaping non-ASCII characters
# and retain key order.
environment.filters["tojson"] = partial(json.dumps, ensure_ascii=False)


def render(template_name: str, **context: object) -> str:
    """Render a packaged prompt with strict variables and unescaped text.

    Args:
        template_name: Template path relative to this package's templates directory.
        context: Values substituted without interpreting their contents as templates.

    Returns:
        Rendered prompt with its literal formatting preserved.
    """
    return environment.get_template(template_name).render(**context)
