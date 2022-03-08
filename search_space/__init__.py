from .ofa_ss import OFASearchSpace
from .alphanet_ss import AlphaNetSearchSpace
from .proxyless_ss import ProxylessSearchSpace

_name_to_class_dict = {'ofa': OFASearchSpace, 'alphanet': AlphaNetSearchSpace, 'proxyless': ProxylessSearchSpace}

def make_search_space(name, **kwargs):
    return _name_to_class_dict[name](**kwargs)