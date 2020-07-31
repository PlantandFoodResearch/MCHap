import numpy as np 
from dataclasses import dataclass

from haplokit.io.vcf.util import vcfstr


@dataclass(frozen=True)
class VCFHeader(object):
    meta: tuple = ()
    contigs: tuple = ()
    filters: tuple = ()
    info_fields: tuple = ()
    format_fields: tuple = ()
    samples: tuple = ()
        
    def columns(self, samples=True):
        cols = (
            'CHROM',
            'POS',
            'ID',
            'REF',
            'ALT',
            'QUAL',
            'FILTER',
            'INFO',
            'FORMAT',
        )
        
        if samples:
            return cols + self.samples
        else:
            return cols
            
    def lines(self):

        for obj in self.meta:
            yield str(obj)

        for obj in self.contigs:
            yield str(obj)

        for obj in self.filters:
            yield str(obj)

        for obj in self.info_fields:
            yield str(obj)

        for obj in self.format_fields:
            yield str(obj)

        yield '#' + '\t'.join(self.columns())

    def __str__(self):
        return '\n'.join(self.lines())


@dataclass(frozen=True)
class VCFRecord(object):
    header: VCFHeader
    chrom: str = None 
    pos: int = None 
    id: str = None 
    ref: str = None 
    alt: tuple = None 
    qual: int = None 
    filter: str = None 
    info: dict = None # key: value
    format: dict = None # sample: key: value

    def __str__(self):

        # order info based on header
        info = dict() if self.info is None else self.info
        ikeys = [f.id for f in self.header.info_fields]
        info = ';'.join(['{}={}'.format(str(k), vcfstr(info.get(k))) for k in ikeys])

        # order sample data based on header
        format_ = dict() if self.format is None else self.format
        samples = self.header.samples
        fkeys = [f.id for f in self.header.format_fields]
        samples = '\t'.join([':'.join([vcfstr(format_.get(s, {}).get(k)) for k in fkeys]) for s in samples])
        format = ':'.join(fkeys)

        line = [
            vcfstr(self.chrom),
            vcfstr(self.pos),
            vcfstr(self.id),
            vcfstr(self.ref),
            vcfstr(self.alt),
            vcfstr(self.qual),
            vcfstr(self.filter),
            info,
            format,
            samples,
        ]
        return '\t'.join(line)


@dataclass
class VCF(object):
    header: VCFHeader
    records: list
        
    def append(self, *args, **kwargs):
        record = VCFRecord(self.header, *args, **kwargs)
        self.records.append(record)
        
    def lines(self, header=True, records=True):

        if header:
            yield from self.header.lines()
        if records:
            for record in self.records:
                yield str(record)

    def __str__(self):
        return '\n'.join(self.lines())
