import re
import numpy as np


_COMPARATOR = {
    "=": np.equal,
    "==": np.equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "!=": np.not_equal,
}


def parse_allele_filter(string):
    """Parse a simple filter string of the form <field><operator><value>.

    Parameters
    ----------
    string : str
        A filter string.

    Returns
    -------
    field
        Field name.
    func
        Numpy function corresponding to the operator.
    value
        Numerical value for comparison with observations.
    """
    pattern = "^(\w+)(=|>|<|==|!=|>=|<|<=|<>)(\d*[.,]?\d*)$"  # noqa: W605
    match = re.search(pattern, string)
    if match:
        field = match.group(1)
        operator = match.group(2)
        if operator in _COMPARATOR:
            operator = _COMPARATOR[operator]
        else:
            raise ValueError(f"Invalid operator in allele filter '{operator}'")
        value = match.group(3)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"Non-numerical value in allele filter '{value}'")
    else:
        raise ValueError(f"Invalid allele filter '{string}'")
    return field, operator, value


def apply_allele_filter(record, field, func, value):
    """
    Apply a simple allele filter to a VCF record.

    Parameters
    ----------
    record
        A pysam VariantRecord object.
    field
        Name of field used to filter alleles.
    func
        Numpy comparison function returning booleans.
    value
        Numerical value for comparison with observations.

    Returns
    -------
    keep
        A boolean array indicating which alleles to keep.
    """
    meta = record.header.info.get(field)
    if meta is None:
        raise ValueError(f"Allele filter field not found in header '{field}'")
    length = meta.number
    if length not in {"R", "A"}:
        raise ValueError(f"Allele filter of field of invalid length '{length}'")
    alts = record.alts
    if alts is None:
        n_alts = 0
    else:
        n_alts = len(alts)
    observations = record.info.get(field)
    if observations is None:
        keep = np.ones(1 + n_alts, dtype=bool)
    elif length == "R":
        assert len(observations) == 1 + n_alts
        keep = func(observations, value)
    elif length == "A":
        assert len(observations) == n_alts
        keep = np.ones(1 + n_alts, dtype=bool)
        keep[1:] = func(observations, value)
    return keep
