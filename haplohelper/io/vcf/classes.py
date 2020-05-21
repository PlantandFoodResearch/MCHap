import numpy as np 
from dataclasses import dataclass

@dataclass(frozen=True)
class MetaHeader(object):
    id: str
    descr: str

    def header(self):
        return '##{id}={descr}'.format(id=self.id, descr=self.descr)


@dataclass(frozen=True)
class ContigHeader(object):
    id: str
    length: int

    def header(self):
        return '##contig=<ID={id},length={length}>'.format(id=self.id, length=self.length)


@dataclass(frozen=True)
class FilterHeader(object):
    id: str
    descr: str

    def header(self):
        template = '##FILTER=<ID={id},Description="{descr}">'
        return template.format(
            id=self.id,
            descr=self.descr
        )


@dataclass(frozen=True)
class InfoField(object):
    id: str
    number: str
    type: str
    descr: str
    dtype: np.dtype = np.object

    def header(self):
        template = '##INFO=<ID={id},Number={number},Type={type},Description="{descr}">'
        return template.format(
            id=self.id,
            number=self.number,
            type=self.type,
            descr=self.descr
        )


@dataclass(frozen=True)
class FormatField(object):
    id: str
    number: str
    type: str
    descr: str
    dtype: np.dtype = np.object

    def header(self):
        template = '##FORMAT=<ID={id},Number={number},Type={type},Description="{descr}">'
        return template.format(
            id=self.id,
            number=self.number,
            type=self.type,
            descr=self.descr
        )


@dataclass(frozen=True)
class Genotype(object):
    alleles: tuple
    phased: bool = False

    def __str__(self):
        sep = '|' if self.phased else '/'
        return sep.join((str(i) if i >= 0 else '.' for i in self.alleles))


@dataclass(frozen=True)
class FilterCall(object):
    id: str
    failed: bool
    applied: bool = True
    
    def __str__(self):
        if self.applied:
            return self.id if self.failed else 'PASS'
        else: 
            return '.'


@dataclass(frozen=True)
class FilterCallSet(object):
    calls: tuple
        
    def __str__(self):
        calls = [call for call in self.calls if call.applied]

        if len(calls) == 0:
            return '.'
        else:
            failed = [call for call in calls if call.failed]
            
            if failed:
                return ','.join(map(str, failed))
            else:
                return 'PASS'


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
            yield obj.header()

        for obj in self.contigs:
            yield obj.header()

        for obj in self.filters:
            yield obj.header()

        for obj in self.info_fields:
            yield obj.header()

        for obj in self.format_fields:
            yield obj.header()

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

            