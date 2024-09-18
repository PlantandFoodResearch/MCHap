import numpy as np


def vcfstr(obj, precision=3):
    # handle large int and float arrays
    if isinstance(obj, np.ndarray):
        if len(obj) == 0:
            return "."
        elif np.issubdtype(obj.dtype, np.floating):
            obj = obj.round(precision)
            # trim any decimal values of 0 and replace nans with '.'
            string = ",".join(obj.astype("U16")).replace("nan", ".").replace(".0,", ",")
            if string[-2:] == ".0":
                return string[:-2]
            else:
                return string
        elif np.issubdtype(obj.dtype, np.integer):
            return ",".join(obj.astype("U16"))
    # handle other objects
    if isinstance(obj, str):
        if obj:
            return obj
        else:
            return "."
    elif hasattr(obj, "__iter__"):
        if len(obj) == 0:
            return "."
        else:
            return ",".join(map(vcfstr, obj))
    elif obj is None:
        return "."
    elif isinstance(obj, float):
        if np.isnan(obj):
            return "."
        obj = np.round(obj, precision)
        i = int(obj)
        if i == obj:
            return str(i)
        else:
            return str(obj)
    else:
        return str(obj)
