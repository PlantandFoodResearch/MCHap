import numpy as np
import networkx as nx
import graphviz as gv
import pysam
import warnings

from haplokit.io.biotargetsfile import read_biotargets


def ancestors(graph, *args, stop=None):
    """Iterate over all ancestor nodes in a DiGraph given
    one or more initial Nodes.
    """
    if stop is None:
        stop = set()
    
    for node in args:
        if node in stop:
            pass
        else:
            yield node
            stop.add(node)
            preds = graph.predecessors(node)
            yield from ancestors(graph, *preds, stop=stop)


def insert_sample_haplotypes(graph, variant, sample_map=None, default_ploidy=2):
    """
    
    Parameters
    ----------
    graph : DiGraph
        Networkx DiGraph of pedigree structure.
    variant_record : VariantRecord
        A Pysam VariantRecord object.
    sample_map : dict, optional
        Optional map of sample names to pedigree item names.

    """
    # map alleles to lists of chars
    haps = (variant.ref, ) + variant.alts
    positions = variant.info['VP']
    alleles = {}
    for i, hap in enumerate(haps):
        alleles[i] = [hap[pos] for pos in positions]
    # add null allele
    alleles[None] = ['-' for _ in positions]

    # keep track of visited nodes
    nodes = set()

    # blank out samples dict
    for ped_item in graph.nodes():
        nodes.add(ped_item)
        graph.node[ped_item]['SAMPLES'] = dict()
    
    # insert sample alleles into graph
    for sample, record in variant.samples.items():
        if sample_map:
            ped_item = sample_map[sample]
        else:
            ped_item = sample
            
        array = [alleles[a] for a in record['GT']]
        
        if ped_item not in graph:
            warnings.warn('Node "{}" not found in pedigree and will be skipped')
        else:
            graph.node[ped_item]['SAMPLES'] = {sample: array}
            if ped_item in nodes:
                nodes.remove(ped_item)

    # add a default null sample to items without samples
    for ped_item in nodes:
        if len(graph.node[ped_item]['SAMPLES']) == 0:
            ploidy = graph.node[ped_item].get('ploidy')
            if not isinstance(ploidy, int):
                ploidy = default_ploidy
            array = [alleles[None] for _ in range(ploidy)]
            graph.node[ped_item]['SAMPLES']['None'] = array
        else:
            assert False
        



_BASE_COLORS = {
    'A': '#e00000',  #  Red
    'C': '#00c000',  #  Green
    'G': '#5050ff',  #  Blue
    'T': '#e6e600',  #  Yellow
    'N': 'black',
    'Z': 'grey',
    '-': 'grey',
    '.': 'grey'
}

_BASE_TEMPLATE = '<TD bgcolor="{color}"> {char} </TD>'


def _haplotype_string(vector, symbol=True):   
    return '<TR>\n' + '\n'.join((_BASE_TEMPLATE.format(
            color=_BASE_COLORS[char],
            char=char if symbol else ' '  # 3 spaces to keep things square
        ) for char in vector)) + '\n</TR>'


def _genotype_string(array, symbol=True):
    head = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">\n'
    body = '\n'.join((_haplotype_string(vector, symbol=symbol) for vector in array))
    tail = '\n</TABLE>>'
    return head + body + tail


def as_graphviz(ped):

    gvg = gv.Digraph('G', node_attr={'shape': 'plaintext'})
    gvg.attr(compound='true')
    gvg.attr(nodesep='1')
    gvg.attr(ranksep='1')

    for node, data in ped.nodes(data=True):

        cluster = 'cluster_{}'.format(node)
        samples = data['SAMPLES'].copy()

        with gvg.subgraph(name=cluster) as sg:
            sg.attr(label='{}'.format(node))
            sg.attr(style='filled', color='lightgrey')
            sg.node_attr.update(style='filled', color='grey')
            for i, (sample, array) in enumerate(samples.items()):
                name = '{}_{}'.format(node, i)
                genotype = _genotype_string(array)
                sg.node(name, genotype)

    for parent, child in ped.edges():

        gvg.edge(
            '{}_0'.format(parent), 
            '{}_0'.format(child), 
            ltail='cluster_{}'.format(parent), 
            lhead='cluster_{}'.format(child), 
        )
        
    return gvg

