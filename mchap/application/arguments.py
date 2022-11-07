import copy
import pysam
from dataclasses import dataclass

from mchap.constant import PFEIFFER_ERROR
from mchap.io import extract_sample_ids


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


sample_pool = Parameter(
    "--sample-pool",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "A name used to pool all sample reads into a single sample. "
            "WARNING: this is an experimental feature."
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

haplotype_frequencies = Parameter(
    "--haplotype-frequencies",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Optionally specify an INFO field within the input VCF file to "
            "designate as allele frequencies for the input haplotypes. "
            "This can be any numerical field of length 'R' and these "
            "values will automatically be normalized. "
            "This parameter has no affect on the output by itself but "
            "is required by some other parameters."
        ),
    ),
)

haplotype_frequencies_prior = BooleanFlag(
    "--haplotype-frequencies-prior",
    dict(
        dest="haplotype_frequencies_prior",
        action="store_true",
        help=(
            "Flag: Use haplotype frequencies to inform prior distribution. "
            "This requires that the --haplotype-frequencies parameter is also specified."
        ),
    ),
)

skip_rare_haplotypes = Parameter(
    "--skip-rare-haplotypes",
    dict(
        type=float,
        nargs=1,
        default=[None],
        help=(
            "Optionally ignore haplotypes from the input VCF file if their frequency "
            "within that file is less than the specified value. "
            "This requires that the --haplotype-frequencies parameter is also specified."
        ),
    ),
)

report = Parameter(
    "--report",
    dict(
        type=str,
        nargs="*",
        default=[],
        help=(
            "Extra fields to report within the output VCF: "
            "AFPRIOR = prior allele frequencies; "
            "AFP = posterior mean allele frequencies; "
            "GP = genotype posterior probabilities; "
            "GL = genotype likelihoods."
        ),
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
        type=float,
        nargs="*",
        default=[1.0],
        help=(
            "A list of inverse-temperatures to use for parallel tempered chains. "
            "These values must be between 0 and 1 and will automatically be sorted in "
            "ascending order. The cold chain value of 1.0 will be added automatically if "
            "it is not specified."
        ),
    ),
)

sample_mcmc_temperatures = Parameter(
    "--sample-mcmc-temperatures",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "A file containing a list of samples with mcmc (inverse) temperatures. "
            "Each line of the file should start with a sample identifier followed by "
            "tab seperated numeric values between 0 and 1. "
            "The number of temperatures specified may vary between samples. "
            "Samples not listed in this file will use the default values specified "
            "with the --mcmc-temperatures argument."
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
            "Posterior phenotype probability threshold for identification of "
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

DEFAULT_PARSER_ARGUMENTS = [
    bam,
    ploidy,
    inbreeding,
    sample_pool,
    base_error_rate,
    ignore_base_phred_scores,
    mapping_quality,
    skip_duplicates,
    skip_qcfail,
    skip_supplementary,
    read_group_field,
    report,
    cores,
]

KNOWN_HAPLOTYPES_ARGUMENTS = [
    haplotypes,
    haplotype_frequencies,
    haplotype_frequencies_prior,
    skip_rare_haplotypes,
]

CALL_EXACT_PARSER_ARGUMENTS = KNOWN_HAPLOTYPES_ARGUMENTS + DEFAULT_PARSER_ARGUMENTS

DEFAULT_MCMC_PARSER_ARGUMENTS = DEFAULT_PARSER_ARGUMENTS + [
    mcmc_chains,
    mcmc_steps,
    mcmc_burn,
    mcmc_seed,
    mcmc_chain_incongruence_threshold,
]

CALL_MCMC_PARSER_ARGUMENTS = KNOWN_HAPLOTYPES_ARGUMENTS + DEFAULT_MCMC_PARSER_ARGUMENTS

ASSEMBLE_MCMC_PARSER_ARGUMENTS = (
    [
        region,
        region_id,
        targets,
        variants,
        reference,
    ]
    + DEFAULT_MCMC_PARSER_ARGUMENTS
    + [
        mcmc_fix_homozygous,
        mcmc_llk_cache_threshold,
        mcmc_recombination_step_probability,
        mcmc_dosage_step_probability,
        mcmc_partial_dosage_step_probability,
        mcmc_temperatures,
        sample_mcmc_temperatures,
        haplotype_posterior_threshold,
    ]
)


def parse_sample_bam_paths(bam_argument, sample_pool_argument, read_group_field):
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
            pysam.AlignmentFile(bam_argument[0])
        except ValueError:
            # not a bam
            textfile = True
        else:
            bams = bam_argument
    else:
        bams = bam_argument
    if not textfile:
        sample_bams = extract_sample_ids(bams, id=read_group_field)
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
            sample_bams = extract_sample_ids(bams, id=read_group_field)
            samples = list(sample_bams)
        elif n_fields == 2:
            # list of sample-bam pairs
            samples = [line[0] for line in lines]
            sample_bams = dict(lines)
        else:
            raise ValueError("Too many fields")

    # handle sample pooling
    if sample_pool_argument is None:
        # pools of 1
        sample_bams = {k: [(k, v)] for k, v in sample_bams.items()}
    else:
        # pool all samples
        samples = [sample_pool_argument]
        sample_bams = {sample_pool_argument: [(k, v) for k, v in sample_bams.items()]}
    # TODO: multiple pools

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
    return data


def parse_sample_temperatures(arguments, samples):
    """Parse inverse temperatures for MCMC simulation
    with parallel-tempering.

    Parameters
    ----------
    arguments
        Parsed arguments containing the "mcmc_temperatures"
        argument and optionally the "sample_mcmc_temperatures"
        argument.
    samples : list
        List of samples.

    Returns
    -------
    sample_temperatures : dict
        Dict mapping each sample to a list of temperatures (floats).

    """
    assert hasattr(arguments, "mcmc_temperatures")
    # per sample mcmc temperatures
    sample_mcmc_temperatures = dict()
    if hasattr(arguments, "sample_mcmc_temperatures"):
        path = arguments.sample_mcmc_temperatures[0]
        if path:
            with open(path) as f:
                for line in f.readlines():
                    values = line.strip().split("\t")
                    sample = values[0]
                    temps = [float(v) for v in values[1:]]
                    temps.sort()
                    assert temps[0] > 0.0
                    assert temps[-1] <= 1.0
                    if temps[-1] != 1.0:
                        temps.append(1.0)
                    sample_mcmc_temperatures[sample] = temps

    # default mcmc temperatures
    temps = arguments.mcmc_temperatures
    temps.sort()
    assert temps[0] > 0.0
    assert temps[-1] <= 1.0
    if temps[-1] != 1.0:
        temps.append(1.0)
    for sample in samples:
        if sample in sample_mcmc_temperatures:
            pass
        else:
            sample_mcmc_temperatures[sample] = temps
    return sample_mcmc_temperatures


def collect_default_program_arguments(arguments):
    # must have some source of error in reads
    if arguments.ignore_base_phred_scores:
        if arguments.base_error_rate[0] == 0.0:
            raise ValueError(
                "Cannot ignore base phred scores if --base-error-rate is 0"
            )
    # merge sample specific data with defaults
    samples, sample_bams = parse_sample_bam_paths(
        arguments.bam, arguments.sample_pool[0], arguments.read_group_field[0]
    )
    sample_ploidy = parse_sample_value_map(
        arguments.ploidy[0],
        samples,
        type=int,
    )
    sample_inbreeding = parse_sample_value_map(
        arguments.inbreeding[0],
        samples,
        type=float,
    )
    return dict(
        samples=samples,
        sample_bams=sample_bams,
        sample_ploidy=sample_ploidy,
        sample_inbreeding=sample_inbreeding,
        read_group_field=arguments.read_group_field[0],
        base_error_rate=arguments.base_error_rate[0],
        ignore_base_phred_scores=arguments.ignore_base_phred_scores,
        mapping_quality=arguments.mapping_quality[0],
        skip_duplicates=arguments.skip_duplicates,
        skip_qcfail=arguments.skip_qcfail,
        skip_supplementary=arguments.skip_supplementary,
        report_fields=arguments.report,
        n_cores=arguments.cores[0],
    )


def collect_call_exact_program_arguments(arguments):
    data = collect_default_program_arguments(arguments)
    data["vcf"] = arguments.haplotypes[0]
    data["random_seed"] = None
    data["use_haplotype_frequencies_prior"] = arguments.haplotype_frequencies_prior
    data["haplotype_frequencies_tag"] = arguments.haplotype_frequencies[0]
    data["skip_rare_haplotypes"] = arguments.skip_rare_haplotypes[0]
    return data


def collect_default_mcmc_program_arguments(arguments):
    data = collect_default_program_arguments(arguments)
    data.update(
        dict(
            mcmc_chains=arguments.mcmc_chains[0],
            mcmc_steps=arguments.mcmc_steps[0],
            mcmc_burn=arguments.mcmc_burn[0],
            mcmc_incongruence_threshold=arguments.mcmc_chain_incongruence_threshold[0],
            random_seed=arguments.mcmc_seed[0],
        )
    )
    return data


def collect_call_mcmc_program_arguments(arguments):
    data = collect_default_mcmc_program_arguments(arguments)
    data["vcf"] = arguments.haplotypes[0]
    data["use_haplotype_frequencies_prior"] = arguments.haplotype_frequencies_prior
    data["haplotype_frequencies_tag"] = arguments.haplotype_frequencies[0]
    data["skip_rare_haplotypes"] = arguments.skip_rare_haplotypes[0]
    return data


def collect_assemble_mcmc_program_arguments(arguments):
    # target and regions cant be combined
    if (arguments.targets[0] is not None) and (arguments.region[0] is not None):
        raise ValueError("Cannot combine --targets and --region arguments.")
    data = collect_default_mcmc_program_arguments(arguments)
    sample_mcmc_temperatures = parse_sample_temperatures(
        arguments, samples=data["samples"]
    )
    data.update(
        dict(
            bed=arguments.targets[0],
            vcf=arguments.variants[0],
            ref=arguments.reference[0],
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
