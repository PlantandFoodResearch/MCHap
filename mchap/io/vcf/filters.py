from dataclasses import dataclass


@dataclass(frozen=True)
class VariantFilter(object):
    id: str
    descr: str

    def __str__(self):
        template = '##FILTER=<ID={id},Description="{descr}">'
        return template.format(id=self.id, descr=self.descr)


PASS = VariantFilter("PASS", "All filters passed")
NAA = VariantFilter(
    "NAA", "No alleles assembled with probability greater than threshold"
)

VARIANT_FILTERS = dict(
    PASS=PASS,
    NAA=NAA,
)
