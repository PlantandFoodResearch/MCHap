import os
import sys
import argparse
import numpy as np
import networkx as nx
import graphviz as gv
import pysam
import warnings
from dataclasses import dataclass


def read_pedigree(path, label_column=0):
    """File containing map of pedigree node name to father and mother nodes.
    Missing parents should be specified as '.'.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split('\t') for line in lines]
    digraph = nx.DiGraph()
    for line in lines:
        assert len(line) >= 3
        child, father, mother = line[0], line[1], line[2]
        digraph.add_node(child, label=line[label_column])
        if father != '.':
            digraph.add_edge(father, child)
        if mother != '.':
            digraph.add_edge(mother, child)
    return digraph


def read_sample_pedigree_map(path):
    """File containing map of VCF sample names to a pedigree nodes.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split('\t') for line in lines]
    sample_pedigree = {sample: node for sample, node in lines}
    return sample_pedigree


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


_FILTER_COLORS = {
    True: 'orange',
    False: 'grey',
    None: 'grey',
}


_BASE_COLORS = {
    'A': '#e00000',  #  Red
    'C': '#00c000',  #  Green
    'G': '#5050ff',  #  Blue
    'T': '#e6e600',  #  Yellow
    'N': 'black',
    'Z': 'lightgrey',
    '-': 'lightgrey',
    '.': 'lightgrey',
    ' ': 'lightgrey',
}

_BASE_TEMPLATE = '<TD bgcolor="{color}">{char}</TD>'


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
    vcf: str
    digraph: nx.DiGraph
    region: str = None
    sample_map: dict = None
    style: str = 'haplotype'
    default_ploidy: int = 2
    label: str = 'label'
    simplify_haplotypes: bool = False
    transpose_haplotypes: bool = False
    show_sample_names: bool = False
    variant_names: bool = False
    label_font: str = "Times-Roman"
    haplotype_font: str = "Mono"
    nodesep: int = 1
    ranksep: int = 1
    rankdir: str = 'TB'
    output_format: str = 'pdf'
    output_path: str = None

    @classmethod
    def cli(cls, command):
        """Program initialisation from cli command
        
        e.g. `program.cli(sys.argv)`
        """
        parser = argparse.ArgumentParser(
            'Plot sample hapotypes within a pedigree'
        )

        parser.set_defaults(plot_gt=False)
        parser.add_argument(
            '--GT',
            dest='plot_gt',
            action='store_true',
            help='Plot GT field instead of haplotypes.'
        )

        parser.add_argument(
            '--vcf',
            type=str,
            nargs=1,
            default=[],
            help='VCF file of haplotypes.',
        )

        parser.add_argument(
            '--region',
            type=str,
            nargs=1,
            default=[None],
            help='Graph haplotypes in specified region.',
        )

        parser.add_argument(
            '--ped',
            type=str,
            nargs=1,
            default=[],
            help='Tab seperated file mapping child nodes to father and mother nodes.',
        )

        parser.add_argument(
            '--sample-pedigree-map',
            type=str,
            nargs=1,
            default=[None],
            help='File containing map of VCF sample names to a pedigree nodes.',
        )

        parser.add_argument(
            '--default-ploidy',
            type=int,
            nargs=1,
            default=[2],
            help='Default ploidy value to use for pedigree items not in the VCF.',
        )

        parser.add_argument(
            '--label-column',
            type=int,
            nargs=1,
            default=[0],
            help='Column of pedigree file to use as pedigree item labels.',
        )

        parser.set_defaults(simplify_haplotypes=False)
        parser.add_argument(
            '--simplify-haplotypes',
            dest='simplify_haplotypes',
            action='store_true',
            help='Remove SNPs that do not vary across haplotypes.'
        )

        parser.set_defaults(transpose_haplotypes=False)
        parser.add_argument(
            '--transpose-haplotypes',
            dest='transpose_haplotypes',
            action='store_true',
            help='Transpose haplotypes into a vertical orientation.'
        )

        parser.set_defaults(show_sample_names=False)
        parser.add_argument(
            '--show-sample-names',
            dest='show_sample_names',
            action='store_true',
            help='Show names of individual samples in the output graphic.'
        )

        parser.set_defaults(use_variant_names=False)
        parser.add_argument(
            '--use-variant-names',
            dest='use_variant_names',
            action='store_true',
            help='Use variant names (if present) for output file names.'
        )

        parser.add_argument(
            '--label-font',
            type=str,
            nargs=1,
            default=["Times-Roman"],
            help='Font for pedigree and sample labels.',
        )

        parser.add_argument(
            '--haplotype-font',
            type=str,
            nargs=1,
            default=["Mono"],
            help='Font for haplotype characters.',
        )

        parser.add_argument(
            '--nodesep',
            type=int,
            nargs=1,
            default=[1],
            help='Graphviz paramter for space seperating nodes.',
        )

        parser.add_argument(
            '--ranksep',
            type=int,
            nargs=1,
            default=[1],
            help='Graphviz paramter for space seperating ranks.',
        )

        parser.add_argument(
            '--rankdir',
            type=str,
            nargs=1,
            default=['TB'],
            help='Graphviz paramter for direction of graph.',
        )

        parser.add_argument(
            '--format',
            type=str,
            nargs=1,
            default=['pdf'],
            help='Output format (default = "pdf").',
        )

        parser.add_argument(
            '-o', '--output',
            type=str,
            nargs=1,
            help='Output directory.',
        )

        if len(command) < 3:
            parser.print_help()
            sys.exit(1)
        args = parser.parse_args(command[2:])


        digraph = read_pedigree(args.ped[0], label_column=args.label_column[0])

        if args.sample_pedigree_map[0]:
            sample_map = read_sample_pedigree_map(args.sample_pedigree_map[0])
        else:
            sample_map = None

        if args.plot_gt:
            style = 'genotype'
        else:
            style = 'haplotype'

        return cls(
            vcf=args.vcf[0],
            digraph=digraph,
            region=args.region[0],
            sample_map=sample_map,
            style=style,
            default_ploidy=args.default_ploidy[0],
            simplify_haplotypes=args.simplify_haplotypes,
            transpose_haplotypes=args.transpose_haplotypes,
            show_sample_names=args.show_sample_names,
            variant_names=args.use_variant_names,
            label_font=args.label_font[0],
            haplotype_font=args.haplotype_font[0],
            nodesep=args.nodesep[0],
            ranksep=args.ranksep[0],
            rankdir=args.rankdir[0],
            output_format=args.format[0],
            output_path=args.output[0],
        )

    def iter_graphs(self):
        with pysam.VariantFile(self.vcf) as vcf:
            if self.region:
                variants = vcf.fetch(self.region)
            else:
                variants = vcf.fetch()

            for variant in variants:
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
                if self.style == 'genotype':
                    graph = variant_genotype_graph(
                        digraph=self.digraph, 
                        variant=variant,
                        sample_map=self.sample_map,
                        label=self.label,
                        show_sample_names=self.show_sample_names,
                        nodesep=self.nodesep,
                        ranksep=self.ranksep,
                        rankdir=self.rankdir,
                    )
                else:
                    graph = variant_haplotype_graph(
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
        return True


def variant_genotype_graph(
        digraph, 
        variant,
        sample_map=None,
        label='label', 
        show_sample_names=False,
        label_font="Times-Roman",
        genotype_font="Mono",
        nodesep=1,
        ranksep=1,
        rankdir='TB'
    ):
    # map of pedigree item to sample to genotype
    ped_sample_genotypes = {}
    ped_sample_filtered = {}
    for sample, record in variant.samples.items():
        tup = tuple('.' if a is None else str(a) for a in record['GT'])
        sep = '|' if record.phased else '/'
        genotype = sep.join(tup)
        if sample_map:
            ped_item = sample_map[sample]
        else:
            ped_item = sample
        if ped_item not in ped_sample_genotypes:
            ped_sample_genotypes[ped_item] = {}
        ped_sample_genotypes[ped_item][sample] = genotype
        filt = record.get('FT')
        if filt is None:
            pass
        elif filt == 'PASS':
            filt = False
        else:
            filt = True
        if ped_item not in ped_sample_filtered:
            ped_sample_filtered[ped_item] = {}
        ped_sample_filtered[ped_item][sample] = filt

    # add null genotype for pedigree items without sampels
    for ped_item, data in digraph.nodes(data=True):
        # get ploidy if present
        if ped_item not in ped_sample_genotypes:
            ped_sample_genotypes[ped_item]={'None': '.'}
            ped_sample_filtered[ped_item]={'None': None}

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
    for node, samples in ped_sample_genotypes.items():
        cluster = 'cluster_{}'.format(node)
        with gvg.subgraph(name=cluster) as sg:
            sg.attr(label=str(labels[node]))
            sg.attr(style='filled', color='lightgrey')
            sg.node_attr.update(style='filled', color='grey')
            for i, (sample, genotype) in enumerate(samples.items()):
                name = '{}_{}'.format(node, i)
                node_color = _FILTER_COLORS[ped_sample_filtered[node][sample]]
                if show_sample_names:
                    sub_cluster = '{}_{}'.format(cluster, sample)                    
                    with sg.subgraph(name=sub_cluster) as ssg:
                        ssg.attr(label=str(sample))
                        ssg.attr(style='filled', color=node_color)
                        ssg.node_attr.update(style='filled', color=node_color)
                        ssg.node(name, genotype, fontname=genotype_font, color=node_color)
                else:
                    sg.node(name, genotype, fontname=genotype_font, color=node_color)

    # copy edges from initial graph
    for parent, child in digraph.edges():
        gvg.edge(
            '{}_{}'.format(parent, len(ped_sample_genotypes[parent])//2), 
            '{}_{}'.format(child, len(ped_sample_genotypes[child])//2), 
            ltail='cluster_{}'.format(parent), 
            lhead='cluster_{}'.format(child), 
        )
  
    return gvg


def variant_haplotype_graph(
        digraph, 
        variant, 
        sample_map=None, 
        default_ploidy=2, 
        label='label', 
        simplify_haplotypes=False,
        transpose_haplotypes=False, 
        show_sample_names=False,
        label_font="Times-Roman",
        haplotype_font="Mono",
        nodesep=1,
        ranksep=1,
        rankdir='TB'
    ):
    
    # map alleles to lists of chars
    haps = (variant.ref, )
    if variant.alts:
        haps += variant.alts
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
    ped_sample_filtered = {}
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
        filt = record.get('FT')
        if filt is None:
            pass
        elif filt == 'PASS':
            filt = False
        else:
            filt = True
        if ped_item not in ped_sample_filtered:
            ped_sample_filtered[ped_item] = {}
        ped_sample_filtered[ped_item][sample] = filt
    
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
            ped_sample_filtered[ped_item]={'None': None}
    
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
                node_color = _FILTER_COLORS[ped_sample_filtered[node][sample]]
                if show_sample_names:
                    sub_cluster = '{}_{}'.format(cluster, sample)                    
                    with sg.subgraph(name=sub_cluster) as ssg:
                        ssg.attr(label=str(sample))
                        ssg.attr(style='filled', color=node_color)
                        ssg.node_attr.update(style='filled', color=node_color)
                        ssg.node(name, genotype, fontname=haplotype_font, color=node_color)
                else:
                    sg.node(name, genotype, fontname=haplotype_font, color=node_color)

    # copy edges from initial graph
    for parent, child in digraph.edges():
        gvg.edge(
            '{}_{}'.format(parent, len(ped_sample_arrays[parent])//2), 
            '{}_{}'.format(child, len(ped_sample_arrays[child])//2), 
            ltail='cluster_{}'.format(parent), 
            lhead='cluster_{}'.format(child), 
        )
  
    return gvg
