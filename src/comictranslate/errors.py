class ComicTranslateError(RuntimeError):
    """Base exception for expected pipeline failures."""


class ConfigurationError(ComicTranslateError):
    """The input, output, runtime, or model configuration is invalid."""


class DetectionError(ComicTranslateError):
    """Text and bubble detection failed."""


class TranslationError(ComicTranslateError):
    """The translation service or its response was invalid."""


class MaskError(ComicTranslateError):
    """Text-mask inference or assembly failed."""


class InpaintingError(ComicTranslateError):
    """LaMa inpainting failed."""


class TextLayoutError(ComicTranslateError):
    """A translated string cannot fit inside its target region."""

