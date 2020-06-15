import re
import numpy as np
from dataclasses import dataclass


meta_pattern = re.compile(r'''\#\#\s*(?P<id>[^,]+)\s*=\s*(?P<descr>[^,]+)\s*''')

column_pattern = re.compile(
    r'''\#\#\s*COLUMN=<
    ID=(?P<id>.+),\s*
    Type=(?P<type>.+),\s*
    Description="(?P<descr>.*)"
    >''', 
    re.VERBOSE
)

pedigree_pattern = re.compile(
    r'''\#\#\s*PEDIGREE=<(?P<column>[^,]+)=(?P<node>[^,]+),\s*(?P<edges>.+)>''', 
    re.VERBOSE
)

pedigree_edge_pattern = re.compile(r'''(?P<edge>[^,]+)=(?P<node>[^,]+)''')


@dataclass(frozen=True)
class MetaHeader(object):
    id: str
    descr: str
    
    def __str__(self):
        return '## {id}={descr}'.format(id=self.id, descr=self.descr)
    
    @classmethod
    def from_string(cls, string):
        data = meta_pattern.match(string).groupdict()
        return cls(**data)
    
@dataclass(frozen=True)
class ColumnHeader(object):
    id: str
    type: str
    descr: str
    
    @classmethod
    def from_string(cls, string):
        data = column_pattern.match(string).groupdict()
        return cls(**data)
    
    def __str__(self):
        return '## COLUMN=<ID={id},Type={type},Description="{descr}">'.format(
            id=self.id, 
            type=self.type,
            descr=self.descr,
        )

@dataclass(frozen=True)
class PedigreeHeader(object):
    column: str
    node: str
    edges: tuple # (PedigreeEdge, )
        
    def __str__(self):
        edges = ','.join([str(edge for edge in self.edges)])
        return '## PEDIGREE=<{column}={node},{edges}>'.format(
            column=self.column,
            node=self.node,
            edges=edges,
        )

    @classmethod
    def from_string(cls, string):
        data = pedigree_pattern.match(string).groupdict()
        edges = pedigree_edge_pattern.findall(data['edges'])
        data['edges'] = [PedigreeEdge(e, n) for e, n in edges]
        return cls(**data)
        

@dataclass(frozen=True)
class PedigreeEdge(object):
    edge: str
    node: str
    
    def __str__(self):
        return '{edge}={node}'.format(edge=self.edge, node=self.node)


TYPE_DISPATCH = {
    'integer': np.int,
    'float': np.float,
    'string': np.object,
}

@dataclass(frozen=True)
class BioTargetsHeader(object):
    meta: tuple
    columns: tuple
    pedigree: tuple = ()
        
    def column_names(self):
        return [col.id for col in self.columns]
        
    def iter_lines(self):
        for line in self.meta:
            yield str(line)
        for line in self.pedigree:
            yield str(line)
        for line in self.columns:
            yield str(line)
        yield '\t'.join(self.column_names())
        
    def __str__(self):
        return '\n'.join(self.iter_lines())
    
    def dtype(self):
        return np.dtype([(col.id, TYPE_DISPATCH[col.type.lower()]) for col in self.columns])
        
@dataclass
class BioTargetsFile(object):
    header: BioTargetsHeader
    array: np.ndarray
        
    def iter_lines(self):
        yield from self.header.iter_lines()
        for row in self.array:
            yield '\t'.join(map(str, row))
        
    def __str__(self):
        return '\n'.join(self.iter_lines())
    

def read_biotargets(path):
    with open(path) as f:
        lines = f.readlines()
    
    meta = list()
    pedigree = list()
    columns= dict()  # order by colnames line
    
    for i, line in enumerate(lines):
        if line.startswith('##'):
            if line.startswith('## COLUMN'):
                obj = ColumnHeader.from_string(line)
                columns[obj.id] = obj
            elif line.startswith('## PEDIGREE'):
                pedigree.append(PedigreeHeader.from_string(line.strip()))
            else:
                meta.append(MetaHeader.from_string(line.strip()))
        else:
            break

    col_names = lines[i].strip().split()
    i += 1
    
    header = BioTargetsHeader(
        meta=tuple(meta),
        columns=tuple(columns[col] for col in col_names),
        pedigree=tuple(pedigree)
    )
    
    array = np.array([tuple(row.strip().split()) for row in lines[i:]], dtype=header.dtype())
    
    return BioTargetsFile(header, array)
