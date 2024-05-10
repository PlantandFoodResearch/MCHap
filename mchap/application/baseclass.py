import sys
import numpy as np
from dataclasses import dataclass
import pysam
import multiprocessing as mp

from mchap import mset
from mchap.constant import PFEIFFER_ERROR
from mchap.encoding import character
from mchap.io import (
    Locus,
    extract_read_variants,
    encode_read_alleles,
    encode_read_distributions,
    vcf,
)
import mchap.io.vcf.infofields as INFO
import mchap.io.vcf.formatfields as FORMAT
import mchap.io.vcf.columns as COLUMN

import warnings

warnings.simplefilter("error", RuntimeWarning)


LOCUS_ASSEMBLY_ERROR = (
    "Exception encountered at locus: '{name}', '{contig}:{start}-{stop}'."
)
SAMPLE_ASSEMBLY_ERROR = "Exception encountered when assembling sample '{sample}'."

KILL_SIGNAL = "MCHAP_KILL_SIGNAL"


class LocusAssemblyError(Exception):
    pass


class SampleAssemblyError(Exception):
    pass


@dataclass
class program(object):
    vcf: str
    ref: str
    samples: list
    sample_bams: dict
    sample_ploidy: dict
    sample_inbreeding: dict
    read_group_field: str = "SM"
    base_error_rate: float = PFEIFFER_ERROR
    ignore_base_phred_scores: bool = True
    mapping_quality: int = 20
    skip_duplicates: bool = True
    skip_qcfail: bool = True
    skip_supplementary: bool = True
    info_fields: list = None
    format_fields: list = None
    n_cores: int = 1
    precision: int = 3
    random_seed: int = 42
    cli_command: str = None

    @classmethod
    def cli(cls, command):
        """Program initialization from cli command

        e.g. `program.cli(sys.argv)`
        """
        raise NotImplementedError()

    def require_AFP(self):
        if {INFO.ACP, INFO.AFP, INFO.AOP, INFO.AOPSUM} & set(self.info_fields):
            return True
        if {FORMAT.ACP, FORMAT.AFP, FORMAT.AOP} & set(self.format_fields):
            return True
        return False

    def loci(self):
        raise NotImplementedError()

    def header_contigs(self):
        with pysam.VariantFile(self.vcf) as f:
            contigs = f.header.contigs.values()
        return [vcf.headermeta.ContigHeader(c.name, c.length) for c in contigs]

    def header(self):
        meta_fields = [
            vcf.headermeta.fileformat("v4.3"),
            vcf.headermeta.filedate(),
            vcf.headermeta.source(),
            vcf.headermeta.phasing("None"),
            vcf.headermeta.commandline(self.cli_command),
            vcf.headermeta.randomseed(self.random_seed),
        ]
        contigs = self.header_contigs()
        filters = [
            vcf.filters.PASS,
            vcf.filters.NOA,
            vcf.filters.AF0,
        ]
        columns = [vcf.headermeta.columns(self.samples)]
        header = (
            meta_fields
            + contigs
            + filters
            + self.info_fields
            + self.format_fields
            + columns
        )
        return [str(line) for line in header]

    def _locus_data(self, locus, sample_bams):
        """Generate a LocusAssemblyData object for a given locus
        to be populated with data relating to a single vcf record.
        """
        return LocusAssemblyData(
            locus=locus,
            samples=self.samples,
            sample_bams=sample_bams,
            sample_ploidy=self.sample_ploidy,
            sample_inbreeding=self.sample_inbreeding,
            read_calls=dict(),
            read_dists=dict(),
            read_counts=dict(),
            infofields=self.info_fields.copy(),
            formatfields=self.format_fields.copy(),
            columndata=dict(FILTER=list()),
            infodata={f: {} for f in INFO.DEFAULT_FIELDS + INFO.OPTIONAL_FIELDS},
            sampledata={f: {} for f in FORMAT.DEFAULT_FIELDS + FORMAT.OPTIONAL_FIELDS},
            precision=self.precision,
        )

    def encode_sample_reads(self, data):
        """Extract and encode reads from each sample at a locus.

        Parameters
        ----------
        data : LocusAssemblyData

        Returns
        -------
        data : LocusAssemblyData
            With sample read data.
        """
        locus = data.locus
        for sample in data.samples:
            # wrap in try clause to pass sample info back with any exception
            try:
                # extract read data for potentially pooled samples
                # each "sample" is a "pool" which is mapped to a list
                # of bam sample-path pairs.
                pairs = data.sample_bams[sample]
                read_chars, read_quals = [], []
                for name, path in pairs:
                    with pysam.AlignmentFile(
                        path, reference_filename=self.ref
                    ) as alignment_file:
                        chars, quals = extract_read_variants(
                            data.locus,
                            alignment_file=alignment_file,
                            samples=name,
                            id=self.read_group_field,
                            min_quality=self.mapping_quality,
                            skip_duplicates=self.skip_duplicates,
                            skip_qcfail=self.skip_qcfail,
                            skip_supplementary=self.skip_supplementary,
                        )[name]
                        read_chars.append(chars)
                        read_quals.append(quals)
                read_chars = np.concatenate(read_chars)
                read_quals = np.concatenate(read_quals)

                # get read stats
                read_count = read_chars.shape[0]
                data.sampledata[FORMAT.RCOUNT][sample] = read_count
                read_variant_depth = character.depth(read_chars)
                if len(read_variant_depth) == 0:
                    read_variant_depth = np.array(np.nan)
                data.sampledata[FORMAT.DP][sample] = np.round(
                    np.mean(read_variant_depth)
                )
                data.sampledata[FORMAT.SNVDP][sample] = np.round(read_variant_depth)

                # encode reads as alleles and probabilities
                read_calls = encode_read_alleles(locus, read_chars)
                data.read_calls[sample] = read_calls
                if self.ignore_base_phred_scores:
                    read_quals = None
                read_dists = encode_read_distributions(
                    locus,
                    read_calls,
                    read_quals,
                    error_rate=self.base_error_rate,
                )
                data.sampledata[FORMAT.RCALLS][sample] = np.sum(read_calls >= 0)

                # de-duplicate reads
                read_dists_unique, read_dist_counts = mset.unique_counts(read_dists)
                data.read_dists[sample] = read_dists_unique
                data.read_counts[sample] = read_dist_counts

            # end of try clause for specific sample
            except Exception as e:
                message = SAMPLE_ASSEMBLY_ERROR.format(sample=sample)
                raise SampleAssemblyError(message) from e
        return data

    def call_sample_genotypes(self, data):
        raise NotImplementedError()

    def sumarise_vcf_record(self, data):
        """Generate VCF record fields.

        Parameters
        ----------
        data : LocusAssemblyData

        Returns
        -------
        data : LocusAssemblyData
        """
        data.columndata[COLUMN.CHROM] = data.locus.contig
        data.columndata[COLUMN.POS] = data.locus.start + 1  # 0-based BED to 1-based VCF
        data.columndata[COLUMN.ID] = data.locus.name
        data.columndata[COLUMN.QUAL] = np.nan  # TODO: calculate locus level QUAL
        # postions
        data.infodata[INFO.END] = data.locus.stop
        data.infodata[INFO.NVAR] = len(data.locus.variants)
        data.infodata[INFO.SNVPOS] = (
            np.subtract(data.locus.positions, data.locus.start) + 1
        )
        # if no filters applied then locus passed
        if len(data.columndata[COLUMN.FILTER]) == 0:
            data.columndata[COLUMN.FILTER] = vcf.filters.PASS.id
        # alt allele counts
        allele_counts = np.zeros(len(data.columndata[COLUMN.ALT]) + 1, int)
        for array in data.sampledata[FORMAT.GT].values():
            for a in array:
                # don't count null alleles
                if a >= 0:
                    allele_counts[a] += 1
        data.infodata[INFO.AC] = allele_counts[1:]  # skip ref count
        # total number of alleles in called genotypes
        data.infodata[INFO.AN] = np.sum(allele_counts)
        # total number of unique alleles in called genotypes
        data.infodata[INFO.UAN] = np.sum(allele_counts > 0)
        # number of called samples
        data.infodata[INFO.NS] = sum(
            np.any(a >= 0) for a in data.sampledata[FORMAT.GT].values()
        )
        # number of samples with Markov chain incongruence
        data.infodata[INFO.MCI] = sum(
            mci > 0 for mci in data.sampledata[FORMAT.MCI].values()
        )
        # total read depth and allele depth
        if len(data.locus.variants) == 0:
            # it will be misleading to return a depth of 0 in this case
            data.infodata[INFO.DP] = np.nan
        else:
            data.infodata[INFO.DP] = np.nansum(
                list(data.sampledata[FORMAT.DP].values())
            )
        # total read count
        data.infodata[INFO.RCOUNT] = np.nansum(
            list(data.sampledata[FORMAT.RCOUNT].values())
        )
        n_allele = len(data.columndata[COLUMN.ALT]) + 1
        null_length_R = np.full(n_allele, np.nan)
        if INFO.ACP in data.infofields:
            _ACP = sum(data.sampledata[FORMAT.AFP].values())
            _ACP = null_length_R if np.isnan(_ACP).all() else _ACP
            data.infodata[INFO.ACP] = _ACP
        if INFO.AFP in data.infofields:
            # use ACP to weight frequencies of each individual by ploidy
            _AFP = sum(data.sampledata[FORMAT.ACP].values()) / sum(
                data.sample_ploidy.values()
            )
            _AFP = null_length_R if np.isnan(_AFP).all() else _AFP
            data.infodata[INFO.AFP] = _AFP
        if INFO.AOPSUM in data.infofields:
            _AOPSUM = sum(data.sampledata[FORMAT.AOP].values())
            _AOPSUM = null_length_R if np.isnan(_AOPSUM).all() else _AOPSUM
            data.infodata[INFO.AOPSUM] = _AOPSUM
        if INFO.AOP in data.infofields:
            prob_not_occurring = np.ones(len(data.columndata[COLUMN.ALT]) + 1, float)
            for occur in data.sampledata[FORMAT.AOP].values():
                prob_not_occurring = prob_not_occurring * (1 - occur)
            prob_occurring = 1 - prob_not_occurring
            data.infodata[INFO.AOP] = prob_occurring
        if INFO.SNVDP in data.infofields:
            _SNVDP = sum(data.sampledata[FORMAT.SNVDP].values())
            data.infodata[INFO.SNVDP] = _SNVDP
        return data

    def call_locus(self, locus, sample_bams):
        """Call samples at a locus and formats resulting data
        into a VCF record line.

        Parameters
        ----------
        locus
            Assembly target locus.
        sample_bams : dict
            Map for sample identifiers to bam path.

        Returns
        -------
        vcf_record : str
            VCF variant line.

        """
        data = self._locus_data(locus, sample_bams)
        self.encode_sample_reads(data)
        self.call_sample_genotypes(data)
        self.sumarise_vcf_record(data)
        return data.format_vcf_record()

    def _assemble_loci_wrapped(self, loci):
        for locus in loci:
            try:
                result = self.call_locus(locus, self.sample_bams)
            except Exception as e:
                message = LOCUS_ASSEMBLY_ERROR.format(
                    name=locus.name,
                    contig=locus.contig,
                    start=locus.start,
                    stop=locus.stop,
                )
                raise LocusAssemblyError(message) from e
            yield result

    def _run_stdout_single_core(self):
        header = self.header()
        for line in header:
            sys.stdout.write(line + "\n")
        for line in self._assemble_loci_wrapped(self.loci()):
            sys.stdout.write(line + "\n")

    def _worker(self, loci, queue):
        for line in self._assemble_loci_wrapped(loci):
            queue.put(str(line))

    def _writer(self, queue):
        while True:
            line = queue.get()
            if line == KILL_SIGNAL:
                break
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def _run_stdout_multi_core(self):

        header = self.header()

        for line in header:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()

        manager = mp.Manager()
        queue = manager.Queue()
        # add one core for the nanny process so that specified number of cores
        # matches the number of compute cores
        pool = mp.Pool(self.n_cores + 1)

        # start writer process
        _ = pool.apply_async(self._writer, (queue,))

        loci = list(self.loci())
        blocks = np.array_split(loci, self.n_cores)
        jobs = []
        for block in blocks:
            job = pool.apply_async(self._worker, (block, queue))
            jobs.append(job)

        for job in jobs:
            job.get()

        queue.put(KILL_SIGNAL)
        pool.close()
        pool.join()

    def run_stdout(self):
        if self.n_cores <= 1:
            self._run_stdout_single_core()
        else:
            self._run_stdout_multi_core()


@dataclass
class LocusAssemblyData(object):
    locus: Locus
    samples: list
    sample_bams: dict
    sample_ploidy: dict
    sample_inbreeding: dict
    read_calls: dict
    read_dists: dict
    read_counts: dict
    infofields: list
    formatfields: list
    columndata: dict
    infodata: dict
    sampledata: dict
    precision: float = 3

    def _sampledata_as_list(self, field):
        data = self.sampledata[field]
        return [data.get(s) for s in self.samples]

    def format_vcf_record(self):
        kwargs = {f.id: self.infodata[f] for f in self.infofields}
        info_string = vcf.format_info_field(precision=self.precision, **kwargs)
        kwargs = {f.id: self._sampledata_as_list(f) for f in self.formatfields}
        format_string = vcf.format_sample_field(precision=self.precision, **kwargs)
        return vcf.format_record(
            chrom=self.columndata[COLUMN.CHROM],
            pos=self.columndata[COLUMN.POS],
            id=self.columndata[COLUMN.ID],
            ref=self.columndata[COLUMN.REF],
            alt=self.columndata[COLUMN.ALT],
            qual=self.columndata[COLUMN.QUAL],
            filter=self.columndata[COLUMN.FILTER],
            info=info_string,
            format=format_string,
            precision=self.precision,
        )
