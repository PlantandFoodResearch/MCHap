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
            id=self.id,
            number=self.number,
            type=self.type,
            descr=self.descr
        )


# INFO fields
NS = InfoField(id='NS', number=1, type='Integer', descr='Number of samples with data') 
DP = InfoField(id='DP', number=1, type='Integer', descr='Combined depth across samples')
AC = InfoField(id='AC', number='A', type='Integer', descr='Allele count in genotypes, for each ALT allele, in the same order as listed')
AN = InfoField(id='AN', number=1, type='Integer', descr='Total number of alleles in called genotypes')
AF = InfoField(id='AF', number='A', type='Float', descr='Allele Frequency')
AA = InfoField(id='AA', number=1, type='String', descr='Ancestral allele')
END = InfoField(id='END', number=1, type='Integer', descr='End position on CHROM')
SNVPOS = InfoField(id='SNVPOS', number='.', type='Integer', descr='Relative (1-based) positions of SNVs within haplotypes')
