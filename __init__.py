from .tele_style_node import TeleStyleLoader, TeleStyleVideoInference

NODE_CLASS_MAPPINGS = {
    "TeleStyleLoader": TeleStyleLoader,
    "TeleStyleVideoInference": TeleStyleVideoInference
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TeleStyleLoader": "TeleStyle Model Loader",
    "TeleStyleVideoInference": "TeleStyle Video Transfer"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']