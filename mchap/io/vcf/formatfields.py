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
SQ = FormatField(id="SQ", number=1, type="Integer", descr="Genotype support quality")
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
SPM = FormatField(
    id="SPM",
    number=1,
    type="Float",
    descr="Genotype support posterior mode probability",
)
DOSEXP = FormatField(
    id="DOSEXP", number=".", type="Float", descr="Mode genotype support expected dosage"
)
MEC = FormatField(id="MEC", number=1, type="Integer", descr="Minimum error correction")
MECP = FormatField(
    id="MECP", number=1, type="Float", descr="Minimum error correction proportion"
)
AD = FormatField(
    id="AD",
    number="R",
    type="Integer",
    descr="Read depth for each allele",
)
GL = FormatField(id="GL", number="G", type="Float", descr="Genotype likelihoods")
GP = FormatField(
    id="GP", number="G", type="Float", descr="Genotype posterior probabilities"
)
ACP = FormatField(id="ACP", number="R", type="Float", descr="Posterior allele counts")
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

SNVDP = FormatField(
    id="SNVDP",
    number=".",
    type="Integer",
    descr="Read depth at each SNV position",

PEDERR = FormatField(
    id="PEDERR",
    number=1,
    type="Float",
    descr="Posterior probability of pedigree error between an individual and its specified parents",
)

HEADER_FORMAT_FIELDS = dict(
    GT=GT,
    GQ=GQ,
    SQ=SQ,
    DP=DP,
    PS=PS,
    FT=FT,
    RCOUNT=RCOUNT,
    RCALLS=RCALLS,
    GPM=GPM,
    SPM=SPM,
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
    PEDERR=PEDERR,
)

DEFAULT_FIELDS = [
    GT,
    GQ,
    SQ,
    DP,
    RCOUNT,
    RCALLS,
    MEC,
    MECP,
    GPM,
    SPM,
    MCI,
]

OPTIONAL_FIELDS = [ACP, AFP, AOP, GP, GL, SNVDP]


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
