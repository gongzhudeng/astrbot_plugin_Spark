def is_slash_prefixed_message(components) -> bool:
    """Return whether the first non-empty source text starts with a slash."""
    for component in components or []:
        text = getattr(component, "text", None)
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if not stripped:
            continue
        return stripped.startswith("/")
    return False
