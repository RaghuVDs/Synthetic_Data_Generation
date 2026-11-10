"""Typed errors raised across the project. Every failure surfaces as a subclass of SynthgenError."""


class SynthgenError(Exception):
    """Base class for every error this project raises on purpose."""


class ConfigError(SynthgenError):
    """Settings or environment values are missing, malformed or out of range."""
