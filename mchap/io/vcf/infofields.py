from dataclasses import dataclass


@dataclass(frozen=True)
class InfoField(object):
    id: str
    number: str
    type: str
    descr: str

    def __str__(self):
        template = '##INFO=<ID={id},Number={number},Type={type},Description="{descr}">'
        return template.format(
            id=self.id, number=self.number, type=self.type, descr=self.descr
        )


# INFO fields
NS = InfoField(id="NS", number=1, type="Integer", descr="Number of samples with data")
DP = InfoField(id="DP", number=1, type="Integer", descr="Combined depth across samples")
PS = InfoField(id="PS", number=1, type="Integer", descr="Phased set for all samples")
AC = InfoField(
    id="AC",
    number="A",
    type="Integer",
    descr="Allele count in genotypes, for each ALT allele, in the same order as listed",
)
AN = InfoField(
    id="AN",
    number=1,
    type="Integer",
    descr="Total number of alleles in called genotypes",
)
UAN = InfoField(
    id="UAN",
    number=1,
    type="Integer",
    descr="Total number of unique alleles in called genotypes",
)
MCI = InfoField(
    id="MCI",
    number=1,
    type="Integer",
    descr="Number of samples with incongruent Markov chain replicates",
)
AF = InfoField(id="AF", number="A", type="Float", descr="Allele Frequency")
AFP = InfoField(
    id="AFP", number="R", type="Float", descr="Posterior mean allele frequencies"
)
ACP = InfoField(id="ACP", number="R", type="Float", descr="Posterior allele counts")
AFPRIOR = InfoField(
    id="AFPRIOR", number="R", type="Float", descr="Prior allele frequencies"
)
AOP = InfoField(
    id="AOP",
    number="R",
    type="Float",
    descr="Posterior probability of allele occurring across all samples",
)
AOPSUM = InfoField(
    id="AOPSUM",
    number="R",
    type="Float",
    descr="Posterior estimate of the number of samples containing an allele",
)
AA = InfoField(id="AA", number=1, type="String", descr="Ancestral allele")
END = InfoField(id="END", number=1, type="Integer", descr="End position on CHROM")
NVAR = InfoField(
    id="NVAR",
    number=1,
    type="Integer",
    descr="Number of input variants within assembly locus",
)
SNVPOS = InfoField(
    id="SNVPOS",
    number=".",
    type="Integer",
    descr="Relative (1-based) positions of SNVs within haplotypes",
)
AD = InfoField(
    id="AD",
    number="R",
    type="Integer",
    descr="Total read depth for each allele",
)
ADMF = InfoField(
    id="ADMF",
    number="R",
    type="Float",
    descr="Mean of sample allele frequencies calculated from read depth",
)
RCOUNT = InfoField(
    id="RCOUNT",
    number=1,
    type="Integer",
    descr="Total number of observed reads across all samples",
)
REFMASKED = InfoField(
    id="REFMASKED",
    number=0,
    type="Flag",
    descr="Reference allele is masked",
)
SNVDP = InfoField(
    id="SNVDP",
    number=".",
    type="Integer",
    descr="Read depth at each SNV position",
)

DEFAULT_FIELDS = [
    AN,
    UAN,
    AC,
    REFMASKED,
    NS,
    MCI,
    DP,
    RCOUNT,
    END,
    NVAR,
    SNVPOS,
]

OPTIONAL_FIELDS = [AFPRIOR, ACP, AFP, AOP, AOPSUM, SNVDP]

ALL_FIELDS = DEFAULT_FIELDS + OPTIONAL_FIELDS
