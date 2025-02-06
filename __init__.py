from .nodes import *
from .nodes2 import StyleAdvancedApply

NODE_CLASS_MAPPINGS = {
    "YC_LG_Redux": YC_LG_Redux,
    "StyleAdvancedApply": StyleAdvancedApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YC_LG_Redux": "YC_LG_Redux_Advance",
    "StyleAdvancedApply": "Style Advanced Apply",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 
