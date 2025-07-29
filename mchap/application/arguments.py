import copy
import pysam
import os
from dataclasses import dataclass

from mchap.constant import PFEIFFER_ERROR
from mchap.io import extract_sample_ids
import mchap.io.vcf.infofields as INFO
import mchap.io.vcf.formatfields as FORMAT


@dataclass
class Argument(object):
    cli: str
    kwargs: dict

    def add_to(self, parser):
        raise NotImplementedError


@dataclass
class Parameter(Argument):
    def add_to(self, parser):
        """Add parameter to a parser object."""
        kwargs = copy.deepcopy(self.kwargs)
        parser.add_argument(
            self.cli,
            **kwargs,
        )
        return parser


@dataclass
class BooleanFlag(Argument):
    def add_to(self, parser):
        """Add boolean flag to a parser object."""
        dest = self.kwargs["dest"]
        action = self.kwargs["action"]
        if action == "store_true":
            default = False
        elif action == "store_false":
            default = True
        else:
            raise ValueError('Action must be "store_true" or "store_false".')
        parser.set_defaults(**{dest: default})
        parser.add_argument(
            self.cli,
            **self.kwargs,
        )
        return parser


haplotypes = Parameter(
    "--haplotypes",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Tabix indexed VCF file containing haplotype/MNP/SNP variants to be "
            "re-called among input samples."
        ),
    ),
)

region = Parameter(
    "--region",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Specify a single target region with the format contig:start-stop. "
            "This region will be a single variant in the output VCF. "
            "This argument can not be combined with the --targets argument."
        ),
    ),
)


region_id = Parameter(
    "--region-id",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Specify an identifier for the locus specified with the "
            "--region argument. This id will be reported in the output VCF."
        ),
    ),
)

targets = Parameter(
    "--targets",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Bed file containing multiple genomic intervals for haplotype assembly. "
            "First three columns (contig, start, stop) are mandatory. "
            "If present, the fourth column (id) will be used as the variant id in "
            "the output VCF."
            "This argument can not be combined with the --region argument."
        ),
    ),
)

variants = Parameter(
    "--variants",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Tabix indexed VCF file containing SNP variants to be used in "
            "assembly. Assembled haplotypes will only contain the reference and "
            "alternate alleles specified within this file."
        ),
    ),
)

reference = Parameter(
    "--reference",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help="Indexed fasta file containing the reference genome.",
    ),
)

bam = Parameter(
    "--bam",
    dict(
        type=str,
        nargs="+",
        default=[],
        help=(
            "Bam file(s) to use in analysis. "
            "This may be (1) a list of one or more bam filepaths, "
            "(2) a plain-text file containing a single bam filepath on each line, "
            "(3) a plain-text file containing a sample identifier and its "
            "corresponding bam filepath on each line separated by a tab. "
            "If options (1) or (2) are used then all samples within each bam will be used within the analysis. "
            "If option (3) is used then only the specified sample will be extracted from each bam file and "
            "An error will be raised if a sample is not found within its specified bam file."
        ),
    ),
)


ploidy = Parameter(
    "--ploidy",
    dict(
        type=str,
        nargs=1,
        default=["2"],
        help=(
            "Specify sample ploidy (default = 2)."
            "This may be (1) a single integer used to specify the ploidy of all samples or "
            "(2) a file containing a list of all samples and their ploidy. "
            "If option (2) is used then each line of the plaintext file must "
            "contain a single sample identifier and the ploidy of that sample separated by a tab."
        ),
    ),
)


inbreeding = Parameter(
    "--inbreeding",
    dict(
        type=str,
        nargs=1,
        default=["0.0"],
        help=(
            "Specify expected sample inbreeding coefficient (default = 0.0)."
            "This may be (1) a single floating point value in the interval [0, 1] "
            "used to specify the inbreeding coefficient of all samples or "
            "(2) a file containing a list of all samples and their inbreeding coefficient. "
            "If option (2) is used then each line of the plaintext file must "
            "contain a single sample identifier and the inbreeding coefficient of that sample separated by a tab."
        ),
    ),
)

sample_parents = Parameter(
    "--sample-parents",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "A file containing a list of samples and their parents "
            "used to indicate pedigree structure. "
            "Each line should contain a sample identifier followed by both "
            "parent identifiers separated by tabs. "
            "A period '.' is used to indicate unknown parents."
        ),
    ),
)

gamete_ploidy = Parameter(
    "--gamete-ploidy",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Ploidy of the gametes contributing to each sample. "
            "By default it is assumed that the ploidy of each gamete "
            "is equal to half the ploidy of the sample derived from that gamete. "
            "If all gametes have the same ploidy then a single number can be specified. "
            "If gametic ploidy is variable then these values must be specified with a "
            "file containing a list of samples and the ploidy of the gametes they were "
            "derived from. "
            "Each line of this file should contain a sample identifier followed by the ploidy "
            "of gametes derived from each parent in the same order as specified "
            "using the --sample-parents argument. "
            "For each sample, the ploidy of the two gametes must sum to the ploidy "
            "of that sample. "
        ),
    ),
)

gamete_ibd = Parameter(
    "--gamete-ibd",
    dict(
        type=str,
        nargs=1,
        default=["0.0"],
        help=(
            "Excess IBD/homozygosity due to meiotic processes. "
            "By default this variable is 0.0 for all gametes and non-zero values "
            "may only be specified for gametes with a ploidy of 2. "
            "This value must be in the interval [0, 1] with a value of 0.0 indicating "
            "no excess ibd/homozygosity and a value of 1.0 indicating that the gamete "
            "must be fully homozygous."
            "If a single value is specified then this will be applied to all gametes. "
            "If this value varies between gametes then it must be specified with a "
            "file containing a list of samples and the value associated with each gamete "
            "those samples are derived from. "
            "Each line should contain a sample identifier followed by value of each "
            "gamete in the same order as specified using the --sample-parents argument. "
        ),
    ),
)

gamete_error = Parameter(
    "--gamete-error",
    dict(
        type=str,
        nargs=1,
        default=["0.01"],
        help=(
            "An error term associated with each parent-child pair indicating the "
            "probability that that gamete was not derived from the specified parent. "
            "By default this variable is 0.01 for all parent-child pairs. "
            "This value must be in the interval [0, 1] and should generally be > 0 and < 1."
            "If a single value is specified then this will be applied to all parent-child pairs. "
            "If this value varies between gametes then it must be specified with a "
            "file containing a list of samples and the value associated with each gamete "
            "those samples are derived from. "
            "Each line should contain a sample identifier followed by value of each "
            "gamete in the same order as specified using the --sample-parents argument. "
        ),
    ),
)

sample_pool = Parameter(
    "--sample-pool",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "WARNING: this is an experimental feature!!! "
            "Pool samples together into a single genotype. "
            "This may be (1) the name of a single pool for all samples or "
            "(2) a file containing a list of all samples and their assigned pool. "
            "If option (2) is used then each line of the plaintext file must "
            "contain a single sample identifier and the name of a pool separated by a tab."
            "Samples may be assigned to multiple pools by using the same sample name on "
            "multiple lines."
            "Each pool will treated as a single genotype by combining "
            "all reads from its constituent samples. "
            "Note that the pool names should be used in place of the samples names when "
            "assigning other per-sample parameters such as ploidy or inbreeding coefficients."
        ),
    ),
)

base_error_rate = Parameter(
    "--base-error-rate",
    dict(
        nargs=1,
        type=float,
        default=[PFEIFFER_ERROR],
        help=(
            "Expected base error rate of read sequences (default = {}). "
            "The default value comes from Pfeiffer et al 2018 "
            "and is a general estimate for Illumina short reads."
        ).format(PFEIFFER_ERROR),
    ),
)

ignore_base_phred_scores = BooleanFlag(
    "--use-base-phred-scores",
    dict(
        dest="ignore_base_phred_scores",
        action="store_false",
        help=(
            "Flag: use base phred-scores as a source of base error rate. "
            "This will use the phred-encoded per base scores in addition "
            "to the general error rate specified by the "
            "--base-error-rate argument. "
            "Using this option can slow down assembly speed."
        ),
    ),
)

haplotype_posterior_threshold = Parameter(
    "--haplotype-posterior-threshold",
    dict(
        type=float,
        nargs=1,
        default=[0.20],
        help=(
            "Posterior probability required for a haplotype to be included in "
            "the output VCF as an alternative allele. "
            "The posterior probability of each haplotype is assessed per individual "
            "and calculated as the probability of that haplotype being present "
            "with one or more copies in that individual."
            "A haplotype is included as an alternate allele if it meets this "
            "posterior probability threshold in at least one individual. "
            "This parameter is the main mechanism to control the number of "
            "alternate alleles in ech VCF record and hence the number of genotypes "
            "assessed when recalculating likelihoods and posterior distributions "
            "(default = 0.20)."
        ),
    ),
)

prior_frequencies = Parameter(
    "--prior-frequencies",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Optionally specify an INFO field within the input VCF file to "
            "designate as prior allele frequencies for the input haplotypes. "
            "This can be any numerical field of length 'R' and these "
            "values will automatically be normalized. "
        ),
    ),
)

filter_input_haplotypes = Parameter(
    "--filter-input-haplotypes",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Optionally filter input haplotypes using a string of the "
            "form '<field><operator><value>' where <field> is a numerical "
            "INFO field with length 'A' or 'R', <operator> is one of "
            "=|>|<|>=|<=|!=, and <value> is a numerical value."
        ),
    ),
)


_optional_field_descriptions = [
    "INFO/{} = {}".format(f.id, f.descr) for f in INFO.OPTIONAL_FIELDS
]
_optional_field_descriptions += [
    "FORMAT/{}: {}".format(f.id, f.descr) for f in FORMAT.OPTIONAL_FIELDS
]

report = Parameter(
    "--report",
    dict(
        type=str,
        nargs="*",
        default=[],
        help=(
            "Extra fields to report within the output VCF. "
            "The INFO/FORMAT prefix may be omitted to return both "
            "variations of the named field. Options include: "
        )
        + "; ".join(_optional_field_descriptions),
    ),
)

mapping_quality = Parameter(
    "--mapping-quality",
    dict(
        nargs=1,
        type=int,
        default=[20],
        help=("Minimum mapping quality of reads used in assembly (default = 20)."),
    ),
)

skip_duplicates = BooleanFlag(
    "--keep-duplicate-reads",
    dict(
        dest="skip_duplicates",
        action="store_false",
        help=(
            "Flag: Use reads marked as duplicates in the assembly "
            "(these are skipped by default)."
        ),
    ),
)

skip_qcfail = BooleanFlag(
    "--keep-qcfail-reads",
    dict(
        dest="skip_qcfail",
        action="store_false",
        help=(
            "Flag: Use reads marked as qcfail in the assembly "
            "(these are skipped by default)."
        ),
    ),
)

skip_supplementary = BooleanFlag(
    "--keep-supplementary-reads",
    dict(
        dest="skip_supplementary",
        action="store_false",
        help=(
            "Flag: Use reads marked as supplementary in the assembly "
            "(these are skipped by default)."
        ),
    ),
)

mcmc_chains = Parameter(
    "--mcmc-chains",
    dict(
        type=int,
        nargs=1,
        default=[2],
        help="Number of independent MCMC chains per assembly (default = 2).",
    ),
)

mcmc_temperatures = Parameter(
    "--mcmc-temperatures",
    dict(
        type=str,
        nargs="*",
        default=["1.0"],
        help=(
            "Specify inverse-temperatures to use for parallel tempered chains (default = 1.0 i.e., no tempering). "
            "This may be either (1) a list of floating point values or "
            "(2) a file containing a list of samples with mcmc inverse-temperatures. "
            "If option (2) is used then the file must contain a single sample per line "
            "followed by a list of tab separated inverse temperatures. "
            "The number of inverse-temperatures may differ between samples and any samples "
            "not included in the list will default to not using tempering."
        ),
    ),
)

mcmc_steps = Parameter(
    "--mcmc-steps",
    dict(
        type=int,
        nargs=1,
        default=[2000],
        help="Number of steps to simulate in each MCMC chain (default = 2000).",
    ),
)

mcmc_burn = Parameter(
    "--mcmc-burn",
    dict(
        type=int,
        nargs=1,
        default=[1000],
        help="Number of initial steps to discard from each MCMC chain (default = 1000).",
    ),
)

mcmc_fix_homozygous = Parameter(
    "--mcmc-fix-homozygous",
    dict(
        type=float,
        nargs=1,
        default=[0.999],
        help=(
            "Fix alleles that are homozygous with a probability greater "
            "than or equal to the specified value (default = 0.999). "
            "The probability of that a variant is homozygous in a sample is "
            "assessed independently for each variant prior to MCMC simulation. "
            'If an allele is "fixed" it is not allowed vary within the MCMC thereby '
            "reducing computational complexity."
        ),
    ),
)

mcmc_seed = Parameter(
    "--mcmc-seed",
    dict(
        type=int,
        nargs=1,
        default=[42],
        help=("Random seed for MCMC (default = 42). "),
    ),
)

mcmc_recombination_step_probability = Parameter(
    "--mcmc-recombination-step-probability",
    dict(
        type=float,
        nargs=1,
        default=[0.5],
        help=(
            "Probability of performing a recombination sub-step during "
            "each step of the MCMC. (default = 0.5)."
        ),
    ),
)

mcmc_partial_dosage_step_probability = Parameter(
    "--mcmc-partial-dosage-step-probability",
    dict(
        type=float,
        nargs=1,
        default=[0.5],
        help=(
            "Probability of performing a within-interval dosage sub-step during "
            "each step of the MCMC. (default = 0.5)."
        ),
    ),
)

mcmc_dosage_step_probability = Parameter(
    "--mcmc-dosage-step-probability",
    dict(
        type=float,
        nargs=1,
        default=[1.0],
        help=(
            "Probability of performing a dosage sub-step during "
            "each step of the MCMC. (default = 1.0)."
        ),
    ),
)

mcmc_chain_incongruence_threshold = Parameter(
    "--mcmc-chain-incongruence-threshold",
    dict(
        type=float,
        nargs=1,
        default=[0.60],
        help=(
            "Posterior probability threshold for identification of "
            "incongruent posterior modes (default = 0.60)."
        ),
    ),
)

mcmc_llk_cache_threshold = Parameter(
    "--mcmc-llk-cache-threshold",
    dict(
        type=int,
        nargs=1,
        default=[100],
        help=(
            "Threshold for determining whether to cache log-likelihoods "
            "during MCMC to improve performance. This value is computed as "
            "ploidy * variants * unique-reads (default = 100). "
            "If set to 0 then log-likelihoods will be cached for all samples "
            "including those with few observed reads which is inefficient and "
            "can slow the MCMC. "
            "If set to -1 then log-likelihood caching will be disabled for all "
            "samples."
        ),
    ),
)

read_group_field = Parameter(
    "--read-group-field",
    dict(
        nargs=1,
        type=str,
        default=["SM"],
        help=(
            'Read group field to use as sample id (default = "SM"). '
            "The chosen field determines tha sample ids required in other "
            "input files e.g. the --sample-list argument."
        ),
    ),
)

cores = Parameter(
    "--cores",
    dict(
        type=int,
        nargs=1,
        default=[1],
        help=("Number of cpu cores to use (default = 1)."),
    ),
)

basis_targets = Parameter(
    "--targets",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Bed file containing genomic intervals. "
            "Basis SNVs will only be identified from within these intervals. "
            "The first three columns (contig, start, stop) are mandatory."
        ),
    ),
)

find_snvs_maf = Parameter(
    "--maf",
    dict(
        type=float,
        nargs=1,
        default=[0.0],
        help=(
            "Minimum sample population allele frequency required to include an allele "
            "(default = 0.0). "
        ),
    ),
)

find_snvs_mad = Parameter(
    "--mad",
    dict(
        type=int,
        nargs=1,
        default=[0],
        help=(
            "Minimum sample population allele depth required to include an allele "
            "(default = 0). "
        ),
    ),
)

find_snvs_ind_maf = Parameter(
    "--ind-maf",
    dict(
        type=float,
        nargs=1,
        default=[0.1],
        help=(
            "Minimum allele frequency of an individual required to include an allele "
            "(default = 0.1). "
            "Alleles will be excluded if their frequency is lower than  "
            "this value across all samples."
        ),
    ),
)

find_snvs_ind_mad = Parameter(
    "--ind-mad",
    dict(
        type=int,
        nargs=1,
        default=[3],
        help=(
            "Minimum allele depth of an individual required to include an allele "
            "(default = 3). "
            "Alleles will be excluded if their depth is lower than  "
            "this value across all samples."
        ),
    ),
)

find_snvs_min_ind = Parameter(
    "--min-ind",
    dict(
        type=int,
        nargs=1,
        default=[1],
        help=(
            "Minimum number of individuals required to meet the --ind-maf and --ind-mad thresholds "
            "(default = 1). "
        ),
    ),
)

# find_snvs_allele_frequency_prior = Parameter(
#     "--allele-frequency-prior",
#     dict(
#         type=str,
#         nargs=1,
#         default=["ADMF"],
#         help=(
#             "Values to use as a prior for population allele frequencies. "
#             "Must be one of {'FLAT', 'ADMF'} (default = 'ADMF') "
#             "Where FLAT indicates a flat prior and ADMF use of the mean "
#             "of sample allele frequencies calculated from allele depth."
#         ),
#     ),
# )


SAMPLE_FLATPRIOR_ARGUMENTS = [
    bam,
    ploidy,
    sample_pool,
]

SAMPLE_DIRMUL_ARGUMENTS = [
    bam,
    ploidy,
    inbreeding,
    sample_pool,
]

LOCI_DENOVO_ARGUMENTS = [
    reference,
    region,
    region_id,
    targets,
    variants,
]

LOCI_KNOWN_ARGUMENTS = [
    reference,
    haplotypes,
    prior_frequencies,
    filter_input_haplotypes,
]

READ_ENCODING_ARGUMENTS = [
    base_error_rate,
    ignore_base_phred_scores,
    mapping_quality,
    skip_duplicates,
    skip_qcfail,
    skip_supplementary,
    read_group_field,
]

MCMC_ARGUMENTS = [
    mcmc_chains,
    mcmc_steps,
    mcmc_burn,
    mcmc_seed,
    mcmc_chain_incongruence_threshold,
]

OUTPUT_ARGUMENTS = [
    report,
]

CORES_ARGUMENTS = [
    cores,
]

ASSEMBLE_MCMC_PARSER_ARGUMENTS = (
    SAMPLE_FLATPRIOR_ARGUMENTS
    + LOCI_DENOVO_ARGUMENTS
    + READ_ENCODING_ARGUMENTS
    + MCMC_ARGUMENTS
    + [
        mcmc_fix_homozygous,
        mcmc_llk_cache_threshold,
        mcmc_recombination_step_probability,
        mcmc_dosage_step_probability,
        mcmc_partial_dosage_step_probability,
        mcmc_temperatures,
        haplotype_posterior_threshold,
    ]
    + OUTPUT_ARGUMENTS
    + CORES_ARGUMENTS
)

CALL_EXACT_PARSER_ARGUMENTS = (
    SAMPLE_DIRMUL_ARGUMENTS
    + LOCI_KNOWN_ARGUMENTS
    + READ_ENCODING_ARGUMENTS
    + OUTPUT_ARGUMENTS
    + CORES_ARGUMENTS
)

CALL_MCMC_PARSER_ARGUMENTS = (
    SAMPLE_DIRMUL_ARGUMENTS
    + LOCI_KNOWN_ARGUMENTS
    + READ_ENCODING_ARGUMENTS
    + MCMC_ARGUMENTS
    + OUTPUT_ARGUMENTS
    + CORES_ARGUMENTS
)

CALL_PEDIGREE_MCMC_PARSER_ARGUMENTS = (
    SAMPLE_FLATPRIOR_ARGUMENTS  # inbreeding is not supported yet
    + [
        sample_parents,
        gamete_ploidy,
        gamete_ibd,
        gamete_error,
    ]
    + LOCI_KNOWN_ARGUMENTS
    + READ_ENCODING_ARGUMENTS
    + MCMC_ARGUMENTS
    + OUTPUT_ARGUMENTS
    + CORES_ARGUMENTS
)


def parse_sample_pools(samples, sample_bams, sample_pool_argument):
    if sample_pool_argument is None:
        # pools of single samples
        sample_bams = {k: [(k, v)] for k, v in sample_bams.items()}
        return samples, sample_bams
    if not os.path.isfile(sample_pool_argument):
        # pool of all samples
        samples = [sample_pool_argument]
        sample_bams = {sample_pool_argument: [(k, v) for k, v in sample_bams.items()]}
        return samples, sample_bams
    else:
        # custom pools
        with open(sample_pool_argument) as f:
            lines = [line.strip().split("\t") for line in f.readlines()]
        pools = list()
        pool_bams = dict()
        samples_in_pools = set()
        for sample, pool in lines:
            samples_in_pools.add(sample)
            bam = sample_bams[sample]
            if pool not in pools:
                # start new pool
                pools.append(pool)
                pool_bams[pool] = [(sample, bam)]
            else:
                # add sample to pool
                pool_bams[pool].append((sample, bam))
        # validation
        sample_with_bams = set(samples)
        diff = sample_with_bams - samples_in_pools
        if diff:
            raise ValueError(
                f"The following samples have not been assigned to a pool: {diff}"
            )
        diff = samples_in_pools - sample_with_bams
        if diff:
            raise ValueError(
                f"The following names in the sample-pool file do not match a known sample : {diff}"
            )
        return pools, pool_bams


def parse_sample_bam_paths(
    bam_argument, sample_pool_argument, read_group_field, reference_path
):
    """Combine arguments relating to sample bam file specification.

    Parameters
    ----------
    argument : list[str]
        list of bam filepaths or single plaintext filepath
    read_group_field : str

    Returns
    -------
    samples : list
        List of samples.
    sample_bam : dict
        Dict mapping samples to bam paths.
    """

    # case of list of bam paths
    textfile = False
    if len(bam_argument) == 1:
        try:
            pysam.AlignmentFile(bam_argument[0], reference_filename=reference_path)
        except ValueError:
            # not a bam
            textfile = True
        else:
            bams = bam_argument
    else:
        bams = bam_argument
    if not textfile:
        sample_bams = extract_sample_ids(
            bams, id=read_group_field, reference_path=reference_path
        )
        samples = list(sample_bams)

    # case of plain-text filepath
    if textfile:
        with open(bam_argument[0]) as f:
            lines = [line.strip().split("\t") for line in f.readlines()]
        n_fields = len(lines[0])
        for line in lines:
            if len(line) != n_fields:
                raise ValueError("Inconsistent number of fields")
        if n_fields == 1:
            # list of bam paths
            bams = [line[0] for line in lines]
            sample_bams = extract_sample_ids(
                bams, id=read_group_field, reference_path=reference_path
            )
            samples = list(sample_bams)
        elif n_fields == 2:
            # list of sample-bam pairs
            samples = [line[0] for line in lines]
            sample_bams = dict(lines)
        else:
            raise ValueError("Too many fields")

    # handle sample pooling
    samples, sample_bams = parse_sample_pools(
        samples, sample_bams, sample_pool_argument
    )

    return samples, sample_bams


def parse_sample_value_map(argument, samples, type):
    """Combine arguments specified for a default value and sample-value map file.

    Parameters
    ----------
    argument : str
        Argument to parse.
    samples : list
        List of sample names
    type : type
        Type of the specified values (float or int).

    Returns
    -------
    sample_values : dict
        Dict mapping samples to values.
    """
    if (type is int) and argument.isdigit():
        value = int(argument)
        return {s: value for s in samples}
    if (type is float) and argument.replace(".", "", 1).isdigit():
        value = float(argument)
        return {s: value for s in samples}
    data = dict()
    with open(argument) as f:
        for line in f.readlines():
            sample, value = line.strip().split("\t")
            data[sample] = type(value)
    for s in samples:
        if s not in data:
            raise ValueError("Sample '{}' not found in file '{}'".format(s, argument))
    return data


def parse_pedigree_arguments(
    samples,
    sample_bams,
    ploidy_argument,
    sample_parents_argument,
    gamete_ploidy_argument,
    gamete_ibd_argument,
    gamete_error_argument,
):
    """Parse arguments related to pedigree specification.

    Parameters
    ----------
    samples : list
        List of ordered samples.
    sample_bams : dict[str, str]
        Dict mapping sample names to bam filepaths.
    ploidy_argument : str
        Filepath or default value for for ploidy.
    inbreeding_argument : str
        Filepath or default value for for inbreeding.
    sample_parents_argument : str
        Filepath to tab-separated values describing pedigree.
    gamete_ploidy_argument : str
        Filepath or default value for for gametic ploidy.
    gamete_ibd_argument,
        Filepath or default value for for gametic IBD excess.
    gamete_error_argument,
        Filepath or default value for for gametic error.

    Returns
    -------
    data : dict
        A dictionary containing the following:
        - 'samples': list of ordered samples.
        - 'sample_bams': dict mapping sample names to bam filepaths.
        - 'sample_ploidy' dict mapping sample names to ploidy.
        - 'sample_inbreeding' dict mapping sample names to inbreeding.
        - 'sample_parents': dict mapping sample names to parent sample names.
        - 'gamete_ploidy': dict mapping sample names to ploidy of contributing gametes.
        - 'gamete_ibd': dict mapping sample names to excess IBD of contributing gametes.
        - 'gamete_error': dict mapping sample names to parental error of contributing gametes.

    Note
    ----
    Samples listed in the 'sample_parents_argument' file will be appended
    to the 'samples' list and 'sample_bams' dict if they are not already present.
    A None value is used to indicate that no bam file is available for that sample.
    """
    # parse pedigree, inserting new samples as required
    known_samples = set(samples)
    sample_parents = dict()
    with open(sample_parents_argument) as f:
        for line in f.readlines():
            sample, p, q = line.strip().split("\t")
            if sample not in known_samples:
                # add a dummy sample
                samples.append(sample)
                sample_bams[sample] = []
                known_samples.add(sample)
            p = None if p == "." else p
            q = None if q == "." else q
            sample_parents[sample] = (p, q)

    # parse ploidy and inbreeding to ensure any additional samples are included
    sample_ploidy = parse_sample_value_map(
        ploidy_argument,
        samples,
        type=int,
    )
    sample_inbreeding = {s: 0.0 for s in samples}

    # gamete ploidy
    gamete_ploidy = dict()
    if gamete_ploidy_argument is None:
        # assume it is half the individuals ploidy
        for sample in samples:
            ploidy = sample_ploidy[sample]
            if ploidy % 2:
                raise ValueError(
                    "Gamete ploidy must be specified for individuals with odd ploidy"
                )
            tau = ploidy // 2
            gamete_ploidy[sample] = (tau, tau)
    elif gamete_ploidy_argument.isdigit():
        # constant for all samples
        tau = int(gamete_ploidy_argument)
        for sample in samples:
            gamete_ploidy[sample] = (tau, tau)
    else:
        # must be a file
        with open(gamete_ploidy_argument) as f:
            for line in f.readlines():
                sample, tau_p, tau_q = line.strip().split("\t")
                gamete_ploidy[sample] = (int(tau_p), int(tau_q))

    # gamete ibd
    gamete_ibd = dict()
    if gamete_ibd_argument.replace(".", "", 1).isdigit():
        # constant for all samples
        lambda_ = float(gamete_ibd_argument)
        for sample in samples:
            gamete_ibd[sample] = (lambda_, lambda_)
    else:
        # must be a file
        with open(gamete_ibd_argument) as f:
            for line in f.readlines():
                sample, lambda_p, lambda_q = line.strip().split("\t")
                gamete_ibd[sample] = (float(lambda_p), float(lambda_q))

    # gamete error
    gamete_error = dict()
    if gamete_error_argument.replace(".", "", 1).isdigit():
        # constant for all samples
        err = float(gamete_error_argument)
        for sample in samples:
            gamete_error[sample] = (err, err)
    else:
        # must be a file
        with open(gamete_error_argument) as f:
            for line in f.readlines():
                sample, err_p, err_q = line.strip().split("\t")
                gamete_error[sample] = (float(err_p), float(err_q))

    return dict(
        samples=samples,
        sample_bams=sample_bams,
        sample_ploidy=sample_ploidy,
        sample_inbreeding=sample_inbreeding,
        sample_parents=sample_parents,
        gamete_ploidy=gamete_ploidy,
        gamete_ibd=gamete_ibd,
        gamete_error=gamete_error,
    )


def parse_sample_temperatures(mcmc_temperatures_argument, samples):
    """Parse inverse temperatures for MCMC simulation
    with parallel-tempering.
    Parameters
    ----------
    mcmc_temperatures_argument : str
        Value(s) for mcmc_temperatures.
    samples : list
        List of samples.
    Returns
    -------
    sample_temperatures : dict
        Dict mapping each sample to a list of temperatures (floats).
    """
    if len(mcmc_temperatures_argument) > 1:
        # must be a list of temps
        floats = True
    elif mcmc_temperatures_argument[0].replace(".", "", 1).isdigit():
        # must be a single temp
        floats = True
    else:
        floats = False
    if floats:
        temps = [float(s) for s in mcmc_temperatures_argument]
        temps.sort()
        assert temps[0] > 0.0
        assert temps[-1] <= 1.0
        if temps[-1] != 1.0:
            temps.append(1.0)
        return {s: temps for s in samples}
    # case of a file, default to 1.0
    data = {s: [1.0] for s in samples}
    with open(mcmc_temperatures_argument[0]) as f:
        for line in f.readlines():
            values = line.strip().split("\t")
            sample = values[0]
            temps = [float(v) for v in values[1:]]
            temps.sort()
            assert temps[0] > 0.0
            assert temps[-1] <= 1.0
            if temps[-1] != 1.0:
                temps.append(1.0)
            data[sample] = temps
    assert len(samples) == len(data)
    return data


def parse_report_fields(report_argument):
    if report_argument is None:
        report_argument = set()
    else:
        report_argument = set(report_argument)
    info_fields = INFO.DEFAULT_FIELDS.copy()
    for f in INFO.OPTIONAL_FIELDS:
        id = f.id
        if (id in report_argument) or (f"INFO/{id}" in report_argument):
            info_fields.append(f)

    format_fields = FORMAT.DEFAULT_FIELDS.copy()
    for f in FORMAT.OPTIONAL_FIELDS:
        id = f.id
        if (id in report_argument) or (f"FORMAT/{id}" in report_argument):
            format_fields.append(f)
    return info_fields, format_fields


def collect_default_program_arguments(arguments, skip_inbreeding=False):
    # must have some source of error in reads
    if arguments.ignore_base_phred_scores:
        if arguments.base_error_rate[0] == 0.0:
            raise ValueError(
                "Cannot ignore base phred scores if --base-error-rate is 0"
            )
    # merge sample specific data with defaults
    samples, sample_bams = parse_sample_bam_paths(
        arguments.bam,
        arguments.sample_pool[0],
        arguments.read_group_field[0],
        reference_path=arguments.reference[0],
    )
    sample_ploidy = parse_sample_value_map(
        arguments.ploidy[0],
        samples,
        type=int,
    )
    if skip_inbreeding:
        sample_inbreeding = None
    else:
        sample_inbreeding = parse_sample_value_map(
            arguments.inbreeding[0],
            samples,
            type=float,
        )
    info_fields, format_fields = parse_report_fields(arguments.report)
    return dict(
        samples=samples,
        sample_bams=sample_bams,
        sample_ploidy=sample_ploidy,
        sample_inbreeding=sample_inbreeding,
        ref=arguments.reference[0],
        read_group_field=arguments.read_group_field[0],
        base_error_rate=arguments.base_error_rate[0],
        ignore_base_phred_scores=arguments.ignore_base_phred_scores,
        mapping_quality=arguments.mapping_quality[0],
        skip_duplicates=arguments.skip_duplicates,
        skip_qcfail=arguments.skip_qcfail,
        skip_supplementary=arguments.skip_supplementary,
        info_fields=info_fields,
        format_fields=format_fields,
        n_cores=arguments.cores[0],
    )


def collect_call_exact_program_arguments(arguments):
    data = collect_default_program_arguments(arguments)
    data["vcf"] = arguments.haplotypes[0]
    data["random_seed"] = None
    data["prior_frequencies_tag"] = arguments.prior_frequencies[0]
    data["filter_input_haplotypes"] = arguments.filter_input_haplotypes[0]
    return data


def collect_default_mcmc_program_arguments(arguments):
    return dict(
        mcmc_chains=arguments.mcmc_chains[0],
        mcmc_steps=arguments.mcmc_steps[0],
        mcmc_burn=arguments.mcmc_burn[0],
        mcmc_incongruence_threshold=arguments.mcmc_chain_incongruence_threshold[0],
        random_seed=arguments.mcmc_seed[0],
    )


def collect_call_mcmc_program_arguments(arguments):
    data = collect_default_program_arguments(arguments)
    data.update(collect_default_mcmc_program_arguments(arguments))
    data["vcf"] = arguments.haplotypes[0]
    data["prior_frequencies_tag"] = arguments.prior_frequencies[0]
    data["filter_input_haplotypes"] = arguments.filter_input_haplotypes[0]
    return data


def collect_call_pedigree_mcmc_program_arguments(arguments):
    # TODO: re-add the inbreeding option when supported
    data = collect_default_program_arguments(arguments, skip_inbreeding=True)
    data["format_fields"] += FORMAT.PEDIGREE_FIELDS
    data.update(collect_default_mcmc_program_arguments(arguments))
    data["vcf"] = arguments.haplotypes[0]
    data["prior_frequencies_tag"] = arguments.prior_frequencies[0]
    data["filter_input_haplotypes"] = arguments.filter_input_haplotypes[0]
    assert data["sample_inbreeding"] is None
    data.update(
        parse_pedigree_arguments(
            samples=data["samples"],
            sample_bams=data["sample_bams"],
            ploidy_argument=arguments.ploidy[0],
            sample_parents_argument=arguments.sample_parents[0],
            gamete_ploidy_argument=arguments.gamete_ploidy[0],
            gamete_ibd_argument=arguments.gamete_ibd[0],
            gamete_error_argument=arguments.gamete_error[0],
        )
    )
    return data


def collect_assemble_mcmc_program_arguments(arguments):
    # target and regions cant be combined
    if (arguments.targets[0] is not None) and (arguments.region[0] is not None):
        raise ValueError("Cannot combine --targets and --region arguments.")
    data = collect_default_program_arguments(
        arguments, skip_inbreeding=True
    )  # flat prior
    data.update(collect_default_mcmc_program_arguments(arguments))
    sample_mcmc_temperatures = parse_sample_temperatures(
        arguments.mcmc_temperatures, samples=data["samples"]
    )
    data.update(
        dict(
            bed=arguments.targets[0],
            vcf=arguments.variants[0],
            sample_mcmc_temperatures=sample_mcmc_temperatures,
            region=arguments.region[0],
            region_id=arguments.region_id,
            # mcmc_alpha,
            # mcmc_beta,
            mcmc_fix_homozygous=arguments.mcmc_fix_homozygous[0],
            mcmc_recombination_step_probability=arguments.mcmc_recombination_step_probability[
                0
            ],
            mcmc_partial_dosage_step_probability=arguments.mcmc_partial_dosage_step_probability[
                0
            ],
            mcmc_dosage_step_probability=arguments.mcmc_dosage_step_probability[0],
            mcmc_llk_cache_threshold=arguments.mcmc_llk_cache_threshold[0],
            haplotype_posterior_threshold=arguments.haplotype_posterior_threshold[0],
        )
    )
    return data
