from .beats import BEATsTransferLearningModel   
from .hftt import get_hftt_model


_backbone_class_map = {
    'beats': BEATsTransferLearningModel,
    'hftt': get_hftt_model,
}


def get_backbone_class(key):
    if key in _backbone_class_map:
        return _backbone_class_map[key]
    else:
        raise ValueError('Invalid backbone: {}'.format(key))