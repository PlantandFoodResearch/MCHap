import numpy as np
from dataclasses import dataclass

from mchap.io.util import qual_of_prob


@dataclass(frozen=True)
class FormatField(object):
    id: str
    number: str
    type: str
    descr: str

    def __str__(self):
        template = (
            '##FORMAT=<ID={id},Number={number},Type={type},Description="{descr}">'
        )
        return template.format(
            id=self.id, number=self.number, type=self.type, descr=self.descr
        )


# FORMAT fields
GT = FormatField(id="GT", number=1, type="String", descr="Genotype")
GQ = FormatField(id="GQ", number=1, type="Integer", descr="Genotype quality")
PHQ = FormatField(id="PHQ", number=1, type="Integer", descr="Phenotype quality")
DP = FormatField(id="DP", number=1, type="Integer", descr="Read depth")
PS = FormatField(id="PS", number=1, type="Integer", descr="Phase set")
FT = FormatField(
    id="FT",
    number=1,
    type="String",
    descr="Filter indicating if this genotype was called",
)
RCOUNT = FormatField(
    id="RCOUNT",
    number=1,
    type="Integer",
    descr="Total count of read pairs within haplotype interval",
)
RCALLS = FormatField(
    id="RCALLS",
    number=1,
    type="Integer",
    descr="Total count of read base calls matching a known variant",
)
GPM = FormatField(
    id="GPM", number=1, type="Float", descr="Genotype posterior mode probability"
)
PHPM = FormatField(
    id="PHPM", number=1, type="Float", descr="Phenotype posterior mode probability"
)
DOSEXP = FormatField(
    id="DOSEXP", number=".", type="Float", descr="Mode phenotype expected dosage"
)
MEC = FormatField(id="MEC", number=1, type="Integer", descr="Minimum error correction")
MECP = FormatField(
    id="MECP", number=1, type="Float", descr="Minimum error correction proportion"
)
AD = FormatField(
    id="AD",
    number="R",
    type="Integer",
    descr="Number of reads unambiguously assigned to each allele based on MEC",
)
GL = FormatField(id="GL", number="G", type="Float", descr="Genotype likelihoods")
GP = FormatField(
    id="GP", number="G", type="Float", descr="Genotype posterior probabilities"
)
AFP = FormatField(
    id="AFP", number="R", type="Float", descr="Posterior mean allele frequencies"
)
AOP = FormatField(
    id="AOP",
    number="R",
    type="Float",
    descr="Posterior probability of allele occurring",
)
MCI = FormatField(
    id="MCI",
    number=1,
    type="Integer",
    descr="Replicate Markov-chain incongruence, 0 = none, 1 = incongruence, 2 = putative CNV",
)
KMERCOV = FormatField(
    id="KMERCOV",
    number=3,
    type="Float",
    descr="Minimum proportion of read-SNV 1-, 2-, and 3-mers found in genotype at any position.",
)
MCAP = FormatField(
    id="MCAP",
    number="R",
    type="Float",
    descr="Posterior probability of allele-presence from assembly MCMC",
)

HEADER_FORMAT_FIELDS = dict(
    GT=GT,
    GQ=GQ,
    PHQ=PHQ,
    DP=DP,
    PS=PS,
    FT=FT,
    RCOUNT=RCOUNT,
    RCALLS=RCALLS,
    GPM=GPM,
    PHPM=PHPM,
    DOSEXP=DOSEXP,
    MEC=MEC,
    MECP=MECP,
    AD=AD,
    GL=GL,
    GP=GP,
    AFP=AFP,
    AOP=AOP,
    MCI=MCI,
    KMERCOV=KMERCOV,
    MCAP=MCAP,
)


def haplotype_depth(variant_depths):
    if len(variant_depths) == 0:
        # 0 variants
        return None
    else:
        depth = np.mean(variant_depths)
        return int(depth)


def quality(prob):
    if prob is None:
        return None
    else:
        return qual_of_prob(prob)


def probabilities(obj, decimals):
    if hasattr(obj, "__iter__"):
        return [probabilities(o, decimals) for o in obj]
    elif isinstance(obj, float):
        return np.round(obj, decimals)
    else:
        return obj
