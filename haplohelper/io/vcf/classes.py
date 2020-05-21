import numpy as np 
from dataclasses import dataclass
from Bio import bgzf


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
        

def vcfstr(obj):
    if isinstance(obj, str):
        if obj:
            return obj
        else:
            return '.'
    elif hasattr(obj, '__iter__'):
        if len(obj) == 0:
            return '.'
        else:
            return ','.join(map(vcfstr, obj))
    elif obj is None:
        return '.'
    else:
        return str(obj)


@dataclass
class VCF(object):
    header: VCFHeader
    data: list
        
    def append(self, chrom=None, pos=None, id=None, ref=None, alt=None, qual=None, filter=None, info=None, format=None):
        
        info = dict() if info is None else info
        info_keys = (f.id for f in self.header.info_fields)
        info_ = tuple((k, info.get(k)) for k in info_keys)
        
        format = dict() if format is None else format
        sample_dicts = ((s, format.get(s, {})) for s in self.header.samples)
        format_keys = tuple(f.id for f in self.header.format_fields)
        format_ = tuple((s, tuple((k, d.get(k)) for k in format_keys)) for s, d in sample_dicts)
        
        self.data.append((chrom, pos, id, ref, alt, qual, filter, info_, format_))
        
    def data_lines(self):
        
        sample_order = tuple(self.header.samples)
        info_keys = tuple(f.id for f in self.header.info_fields)
        format_keys = tuple(f.id for f in self.header.format_fields)
        
        for tup in self.data:
            
            assert len(tup) == 9
            line = list(map(vcfstr, tup[0:7]))
            
            info = tup[7]
            format_=tup[8]
            
            # ensure info data in correct order
            assert info_keys == tuple(k for k, _ in info)
            
            # info string 
            line.append(';'.join(('{}={}'.format(k, vcfstr(v)) for k, v in info)))
            
            # format string
            line.append(':'.join(format_keys))
            
            # ensure samples in correct order
            assert sample_order == tuple(s for s, _ in format_)
            
            # iter through sample specific format data
            for _, sample_data in format_:
            
                # ensure format data in correct order
                assert format_keys == tuple(k for k, _ in sample_data)
                
                # sample format string
                line.append(':'.join((vcfstr(v) for _, v in sample_data)))
                
            yield '\t'.join(line)


    def lines(self):

        yield from self.header.lines()
        yield from self.data_lines()


    def write(self, path, bgzip=False):

        open_ = bgzf.open if bgzip else open

        with open_(path, 'w') as f:
            for line in self.lines():
                f.write(line + '\n')
            