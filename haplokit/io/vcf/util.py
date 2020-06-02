import numpy as np

def vcfstr(obj):
    if isinstance(obj, str):
        if obj:
            return obj
        else:
            return '.'
    elif hasattr(obj, '__iter__'):
        if len(obj) == 0:
            return '.'
        else:
            return ','.join(map(vcfstr, obj))
    elif obj is None:
        return '.'
    else:
        return str(obj)


def vcfround(obj, decimals):
    if hasattr(obj, '__iter__'):
        return [vcfround(o, decimals) for o in obj]
    elif isinstance(obj, float):
        return np.round(obj, decimals)
    else:
        return obj
