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
        template = '##FORMAT=<ID={id},Number={number},Type={type},Description="{descr}">'
        return template.format(
            id=self.id,
            number=self.number,
            type=self.type,
            descr=self.descr
        )


# FORMAT fields
GT = FormatField(id='GT', number=1, type='String', descr='Genotype')
GQ = FormatField(id='GQ', number=1, type='Integer', descr='Genotype quality')
PHQ = FormatField(id='PHQ', number=1, type='Integer', descr='Phenotype quality')
DP = FormatField(id='DP', number=1, type='Integer', descr='Read depth')
PS = FormatField(id='PS', number=1, type='Integer', descr='Phase set')
FT = FormatField(id='FT', number=1, type='String', descr='Filter indicating if this genotype was called')
RC = FormatField(id='RC', number=1, type='Integer', descr='Total count of read pairs within haplotype interval')
GP = FormatField(id='GP', number='G', type='Float', descr='Genotype posterior probabilities')
GPM = FormatField(id='GPM', number=1, type='Float', descr='Genotype posterior mode probability')
PPM = FormatField(id='PPM', number=1, type='Float', descr='Penotype posterior mode probability')
MPGP = FormatField(id='MPGP', number='.', type='Float', descr='Genotype posterior probabilities of genotypes contributing to the posterior mode phenotype')
MPED = FormatField(id='MPED', number='.', type='Float', descr='Mode phenotype expected dosage')


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
    if hasattr(obj, '__iter__'):
        return [probabilities(o, decimals) for o in obj]
    elif isinstance(obj, float):
        return np.round(obj, decimals)
    else:
        return obj
