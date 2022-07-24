from dataclasses import dataclass
import pysam

from mchap.application import baseclass
from mchap.io import LocusPrior


@dataclass
class program(baseclass.program):
    haplotype_frequencies_tag: str = None
    use_haplotype_frequencies_prior: bool = False
    skip_rare_haplotypes: float = None

    def loci(self):
        with pysam.VariantFile(self.vcf) as f:
            for record in f.fetch():
                locus = LocusPrior.from_variant_record(
                    record,
                    frequency_tag=self.haplotype_frequencies_tag,
                    frequency_min=self.skip_rare_haplotypes,
                )
                yield locus
