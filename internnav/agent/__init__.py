from internnav.agent.base import Agent

__all__ = ['Agent']


def _optional_import(module_name, attr_name):
    try:
        module = __import__(module_name, fromlist=[attr_name])
        attr = getattr(module, attr_name)
        globals()[attr_name] = attr
        __all__.append(attr_name)
    except Exception:
        return None
    return attr


_optional_import('internnav.agent.cma_agent', 'CmaAgent')
_optional_import('internnav.agent.dialog_agent', 'DialogAgent')
_optional_import('internnav.agent.internvla_n1_agent', 'InternVLAN1Agent')
_optional_import('internnav.agent.rdp_agent', 'RdpAgent')
_optional_import('internnav.agent.seq2seq_agent', 'Seq2SeqAgent')
