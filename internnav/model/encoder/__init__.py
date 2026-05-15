from importlib import import_module


__all__ = []


def _optional_import(module_name, attr_name):
    try:
        module = import_module(module_name, package=__name__)
        value = getattr(module, attr_name)
        globals()[attr_name] = value
        __all__.append(attr_name)
    except Exception:
        return None
    return value


_optional_import('.bert_backbone', 'PositionalEncoding')
_optional_import('.distance_encoder', 'DistanceNetwork')
_optional_import('.image_clip_encoder', 'ImageEncoder')
_optional_import('.instruction_encoder', 'InstructionEncoder')
_optional_import('.instruction_longCLIP_encoder', 'InstructionLongCLIPEncoder')
_optional_import('.instruction_roberta_encoder', 'LanguageEncoder')
_optional_import('.vision_language_encoder', 'VisionLanguageEncoder')
