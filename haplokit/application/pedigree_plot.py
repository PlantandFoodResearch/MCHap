import os
import numpy as np
import networkx as nx
import graphviz as gv
import pysam
import warnings
from dataclasses import dataclass

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


@dataclass
class program(object):
    digraph: nx.DiGraph
    variants: list
    sample_map: dict = None
    default_ploidy: int = 2
    label: str = 'label'
    simplify_haplotypes: bool = False
    transpose_haplotypes: bool = False
    show_sample_names: bool = False
    variant_names: bool = False
    nodesep: int = 1
    ranksep: int = 1
    rankdir: str = 'TB'
    output_format: str = 'pdf'
    output_path: str = None

    @classmethod
    def cli(cls):
        pass

    def iter_graphs(self):
        for variant in self.variants:
            if self.variant_names and variant.id:
                name = variant.id
            else:
                name = '{}_{}'.format(
                    variant.chrom,
                    variant.pos
                )
                if variant.stop:
                    name = '{}_{}'.format(
                        name,
                        variant.stop,
                    )

            graph = graph_variant(
                digraph=self.digraph, 
                variant=variant, 
                sample_map=self.sample_map, 
                default_ploidy=self.default_ploidy, 
                label=self.label, 
                simplify_haplotypes=self.simplify_haplotypes,
                transpose_haplotypes=self.transpose_haplotypes, 
                show_sample_names=self.show_sample_names,
                nodesep=self.nodesep,
                ranksep=self.ranksep,
                rankdir=self.rankdir,
            )

            yield name, graph

    def run(self):
        if not os.path.isdir(self.output_path):
            raise IOError('{} is not a directory.'.format(self.output_path))
        for name, graph in self.iter_graphs():
            path = self.output_path + '/' + name + '.gv'
            graph.render(path, format=self.output_format, cleanup=True)


def graph_variant(
        digraph, 
        variant, 
        sample_map=None, 
        default_ploidy=2, 
        label='label', 
        simplify_haplotypes=False,
        transpose_haplotypes=False, 
        show_sample_names=False,
        nodesep=1,
        ranksep=1,
        rankdir='TB'
    ):
    
    # map alleles to lists of chars
    haps = (variant.ref, ) + variant.alts
    positions = variant.info['VP']
    if simplify_haplotypes:
        counts = [(pos, len({h[pos] for h in haps})) for pos in positions]
        positions = [pos for pos, count in counts if count > 1]
    alleles = {}
    for i, hap in enumerate(haps):
        alleles[i] = [hap[pos] for pos in positions]
    # add null allele
    alleles[None] = ['-' for _ in positions]    
    
    # map of pedigree item to sample to haplotype chars
    ped_sample_arrays = {}
    for sample, record in variant.samples.items():
        array = np.array([alleles[a] for a in record['GT']])
        if transpose_haplotypes:
            array = array.transpose()
        if sample_map:
            ped_item = sample_map[sample]
        else:
            ped_item = sample
        if ped_item not in ped_sample_arrays:
            ped_sample_arrays[ped_item] = {}
        ped_sample_arrays[ped_item][sample] = array
    
    # add null haps for pedigree items without sampels
    for ped_item, data in digraph.nodes(data=True):
        # get ploidy if present
        if ped_item not in ped_sample_arrays:
            ploidy = data.get('ploidy')
            if not isinstance(ploidy, int):
                # fall back to default ploidy
                ploidy = default_ploidy
            # create array of null haplotypes
            array = np.array([alleles[None] for _ in range(ploidy)])
            if transpose_haplotypes:
                array = array.transpose()
            ped_sample_arrays[ped_item]={'None': array}
    
    # create graphviz
    gvg = gv.Digraph('G', node_attr={'shape': 'plaintext'})
    gvg.attr(compound='true')
    gvg.attr(nodesep=str(nodesep))
    gvg.attr(ranksep=str(ranksep))
    gvg.attr(rankdir=rankdir)
    
    # get labels from graph (default to node name)
    labels = {}
    for node, data in digraph.nodes(data=True):
        labels[node] = data.get(label, node)
    
    # create subgraph of nodes for each sample
    for node, samples in ped_sample_arrays.items():
        cluster = 'cluster_{}'.format(node)
        with gvg.subgraph(name=cluster) as sg:
            sg.attr(label=str(labels[node]))
            sg.attr(style='filled', color='lightgrey')
            sg.node_attr.update(style='filled', color='grey')
            for i, (sample, array) in enumerate(samples.items()):
                name = '{}_{}'.format(node, i)
                genotype = _genotype_string(array)
                if show_sample_names:
                    sub_cluster = '{}_{}'.format(cluster, sample)
                    with sg.subgraph(name=sub_cluster) as ssg:
                        ssg.attr(label=str(sample))
                        ssg.attr(style='filled', color='grey')
                        ssg.node_attr.update(style='filled', color='grey')
                        ssg.node(name, genotype)
                else:
                    sg.node(name, genotype)

    # copy edges from initial graph
    for parent, child in digraph.edges():
        gvg.edge(
            '{}_{}'.format(parent, len(ped_sample_arrays[parent])//2), 
            '{}_{}'.format(child, len(ped_sample_arrays[child])//2), 
            ltail='cluster_{}'.format(parent), 
            lhead='cluster_{}'.format(child), 
        )
  
    return gvg
