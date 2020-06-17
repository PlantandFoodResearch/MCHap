import re
import warnings
import networkx as nx
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
        edges = ','.join([str(edge) for edge in self.edges])
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
    'integer': int,
    'float': float,
    'string': str,
}

@dataclass(frozen=True)
class BioTargetsHeader(object):
    meta: tuple
    columns: tuple
    pedigree: tuple = ()
        
    def column_names(self):
        return [col.id for col in self.columns]

    def column_types(self):
        return [TYPE_DISPATCH[col.type.lower()] for col in self.columns]
            
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

    def pedigree_column(self):
        column = {node.column for node in self.pedigree}
        assert len(column) == 1
        column = column.pop()
        return column


@dataclass
class BioTargetsFile(object):
    header: BioTargetsHeader
    data: list
        
    def iter_lines(self, header=True):
        if header:
            yield from self.header.iter_lines()
        for row in self.data:
            yield '\t'.join(('' if field is None else str(field) for field in row))

    def iter_dicts(self):
        columns = self.header.column_names()
        for tup in self.data:
            yield dict(zip(columns, tup))
        
    def __str__(self):
        return '\n'.join(self.iter_lines())

    def pedigree_digraph(self, data=None, warn=True, edge_labels=True):
        if data is None:
            data = []
        
        ped_column = self.header.pedigree_column()
        ped_col_header = [col for col in self.header.columns if col.id==ped_column]
        assert len(ped_col_header) == 1
        ped_col_header=ped_col_header[0]
        ped_col_type = TYPE_DISPATCH[ped_col_header.type.lower()]
        
        # add columns as nodes
        graph = nx.DiGraph()
        for row in self.iter_dicts():
            node = row[ped_column]
            if node:
                d = {col: row[col] for col in data}
                graph.add_node(node, **d)
        
        # add pedigree edges
        message = 'Pedigree node "{}" not found in column "{}".'
        for obj in self.header.pedigree:
            child = ped_col_type(obj.node)
            if warn and (child not in graph):
                warnings.warn(message.format(child, ped_column))
            for edge in obj.edges:
                relation = edge.edge
                parent = ped_col_type(edge.node)
                if warn and (parent not in graph):
                    warnings.warn(message.format(parent, ped_column))
                
                if edge_labels:
                    graph.add_edge(parent, child, label=relation)
                else:
                    graph.add_edge(parent, child)

        return graph
    

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

    col_names = lines[i].strip().split('\t')
    i += 1
    
    header = BioTargetsHeader(
        meta=tuple(meta),
        columns=tuple(columns[col] for col in col_names),
        pedigree=tuple(pedigree)
    )
    
    col_types = header.column_types()
    data = []

    for row in lines[i:]:
        strings = row.strip().split('\t')
        data.append(tuple(t(s) if s else None for t, s in zip(col_types, strings)))

    return BioTargetsFile(header, data)

