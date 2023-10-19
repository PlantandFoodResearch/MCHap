from dataclasses import dataclass
import pysam

from mchap.application import baseclass
from mchap.io import LocusPrior


@dataclass
class program(baseclass.program):
    haplotype_frequencies_tag: str = None
    filter_input_haplotypes: str = None

    def loci(self):
        with pysam.VariantFile(self.vcf) as f:
            for record in f.fetch():
                locus = LocusPrior.from_variant_record(
                    record,
                    frequency_tag=self.haplotype_frequencies_tag,
                    allele_filter=self.filter_input_haplotypes,
                )
                yield locus
