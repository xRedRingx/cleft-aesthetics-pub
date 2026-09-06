"""Config schema and loader. The config is the provenance record."""

from .schema import (
    DECLARED_PATH_KINDS,
    ENV_REFERENCE,
    INPUT_SPEC,
    PORTABLE_PATH_KINDS,
    SCHEMA,
    SUPPORTED_SCHEMA_VERSIONS,
    TASK_SPECS,
    ConfigError,
    Field,
    classify_declared_path,
    dump_config,
    expand_input_paths,
    load_config,
    validate,
)

__all__ = [
    "ConfigError",
    "DECLARED_PATH_KINDS",
    "ENV_REFERENCE",
    "Field",
    "INPUT_SPEC",
    "PORTABLE_PATH_KINDS",
    "SCHEMA",
    "SUPPORTED_SCHEMA_VERSIONS",
    "TASK_SPECS",
    "classify_declared_path",
    "dump_config",
    "expand_input_paths",
    "load_config",
    "validate",
]
