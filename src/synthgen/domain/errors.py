"""Typed errors raised across the project. Every failure surfaces as a subclass of SynthgenError."""


class SynthgenError(Exception):
    """Base class for every error this project raises on purpose."""


class ConfigError(SynthgenError):
    """Settings or environment values are missing, malformed or out of range."""


class MetadataError(SynthgenError):
    """Metadata is malformed or disagrees with the source database."""


class UnknownIdentifierError(MetadataError):
    """A table or column name is not present in the source schema."""


class DependencyCycleError(MetadataError):
    """Tables reference each other in a cycle with no nullable key to break it."""


class ConnectorError(SynthgenError):
    """The source database cannot be opened or a read query failed."""


class StorageError(SynthgenError):
    """The state database cannot be read or written."""


class NotFoundError(SynthgenError):
    """A requested job, artifact or source does not exist."""


class SeedError(SynthgenError):
    """The fake source databases could not be built."""


class SpecCompilationError(SynthgenError):
    """A baseline spec cannot be compiled from the profiles; a code or profile defect."""


class EngineError(SynthgenError):
    """The engine cannot generate data that satisfies the spec."""
