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


def as_haplotype_graphviz(graph, variant, sample_map=None, default_ploidy=2):
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
    
    # map of pedigree item to sample to haplotype chars
    ped_sample_arrays = {}
    for sample, record in variant.samples.items():
        array = [alleles[a] for a in record['GT']]
        if sample_map:
            ped_item = sample_map[sample]
        else:
            ped_item = sample
        if ped_item not in ped_sample_arrays:
            ped_sample_arrays[ped_item] = {}
        ped_sample_arrays[ped_item][sample] = array
    
    # add null haps for pedigree items without sampels
    for ped_item, data in graph.nodes(data=True):
        # get ploidy if present
        if ped_item not in ped_sample_arrays:
            ploidy = data.get('ploidy')
            if not isinstance(ploidy, int):
                # fall back to default ploidy
                ploidy = default_ploidy
            # create array of null haplotypes
            array = [alleles[None] for _ in range(ploidy)]
            ped_sample_arrays[ped_item]={'None': array}
    
    # create graphviz
    gvg = gv.Digraph('G', node_attr={'shape': 'plaintext'})
    gvg.attr(compound='true')
    gvg.attr(nodesep='1')
    gvg.attr(ranksep='1')
    
    # create subgraph of nodes for each sample
    for node, samples in ped_sample_arrays.items():
        cluster = 'cluster_{}'.format(node)
        with gvg.subgraph(name=cluster) as sg:
            sg.attr(label='{}'.format(node))
            sg.attr(style='filled', color='lightgrey')
            sg.node_attr.update(style='filled', color='grey')
            for i, (sample, array) in enumerate(samples.items()):
                name = '{}_{}'.format(node, i)
                genotype = _genotype_string(array)
                sg.node(name, genotype)
    
    # copy edges from initial graph
    for parent, child in graph.edges():
        gvg.edge(
            '{}_0'.format(parent), 
            '{}_0'.format(child), 
            ltail='cluster_{}'.format(parent), 
            lhead='cluster_{}'.format(child), 
        )
    return gvg
