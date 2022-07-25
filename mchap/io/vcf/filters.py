from dataclasses import dataclass


@dataclass(frozen=True)
class VariantFilter(object):
    id: str
    descr: str

    def __str__(self):
        template = '##FILTER=<ID={id},Description="{descr}">'
        return template.format(id=self.id, descr=self.descr)


PASS = VariantFilter("PASS", "All filters passed")
NOA = VariantFilter("NOA", "No observed alleles at locus")
AF0 = VariantFilter("AF0", "All alleles have prior allele frequency of zero")

VARIANT_FILTERS = dict(
    PASS=PASS,
    NOA=NOA,
    AF0=AF0,
)
