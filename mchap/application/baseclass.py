import sys
import numpy as np
from dataclasses import dataclass
import pysam
import multiprocessing as mp
from collections import OrderedDict

from mchap import mset
from mchap.constant import PFEIFFER_ERROR
from mchap.encoding import character, integer
from mchap.io import (
    Locus,
    extract_read_variants,
    encode_read_alleles,
    encode_read_distributions,
    vcf,
)
from mchap.io.vcf.infofields import HEADER_INFO_FIELDS, OPTIONAL_INFO_FIELDS
from mchap.io.vcf.formatfields import HEADER_FORMAT_FIELDS, OPTIONAL_FORMAT_FIELDS

import warnings

warnings.simplefilter("error", RuntimeWarning)


LOCUS_ASSEMBLY_ERROR = (
    "Exception encountered at locus: '{name}', '{contig}:{start}-{stop}'."
)
SAMPLE_ASSEMBLY_ERROR = "Exception encountered when assembling sample '{sample}'."


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
    report_fields: list = ()
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

    def info_fields(self):
        infofields = [
            "AN",
            "AC",
            "REFMASKED",
            "NS",
            "DP",
            "RCOUNT",
            "END",
            "NVAR",
            "SNVPOS",
        ]
        for f in OPTIONAL_INFO_FIELDS:
            id = f.id
            if (id in self.report_fields) or (f"INFO/{id}" in self.report_fields):
                infofields.append(id)
        return infofields

    def format_fields(self):
        formatfields = [
            "GT",
            "GQ",
            "PHQ",
            "DP",
            "RCOUNT",
            "RCALLS",
            "MEC",
            "MECP",
            "GPM",
            "PHPM",
            "MCI",
        ]
        for f in OPTIONAL_FORMAT_FIELDS:
            id = f.id
            if (id in self.report_fields) or (f"FORMAT/{id}" in self.report_fields):
                formatfields.append(id)
        return formatfields

    def require_AFP(self):
        requested = set(self.info_fields()) | set(self.format_fields())
        if {"ACP", "AFP", "AOP", "AOPSUM"} & requested:
            return True
        else:
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
        info_fields = [HEADER_INFO_FIELDS[field] for field in self.info_fields()]
        format_fields = [HEADER_FORMAT_FIELDS[field] for field in self.format_fields()]
        columns = [vcf.headermeta.columns(self.samples)]
        header = meta_fields + contigs + filters + info_fields + format_fields + columns
        return [str(line) for line in header]

    def _locus_data(self, locus, sample_bams):
        """Generate a LocusAssemblyData object for a given locus
        to be populated with data relating to a single vcf record.
        """
        infofields = self.info_fields()
        formatfields = self.format_fields()
        return LocusAssemblyData(
            locus=locus,
            samples=self.samples,
            sample_bams=sample_bams,
            sample_ploidy=self.sample_ploidy,
            sample_inbreeding=self.sample_inbreeding,
            infofields=infofields,
            formatfields=formatfields,
            columndata=dict(FILTER=list()),
            infodata=dict(),
            sampledata=dict(),
        )

    def encode_sample_reads(self, data):
        """Extract and encode reads from each sample at a locus.

        Parameters
        ----------
        data : LocusAssemblyData
            With relevant locus, samples, sample_bams, and sample_inbreeding attributes.

        Returns
        -------
        data : LocusAssemblyData
            With sampledata fields: "read_calls", "read_dists_unique", "read_dist_counts",
            "DP", "RCOUNT", "RCALLS".
        """
        for field in [
            "DP",
            "RCOUNT",
            "RCALLS",
            "SNVDP",
            "read_calls",
            "read_dists_unique",
            "read_dist_counts",
        ]:
            data.sampledata[field] = dict()
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
                data.sampledata["RCOUNT"][sample] = read_count
                read_variant_depth = character.depth(read_chars)
                if len(read_variant_depth) == 0:
                    read_variant_depth = np.array(np.nan)
                data.sampledata["DP"][sample] = np.round(np.mean(read_variant_depth))
                data.sampledata["SNVDP"][sample] = np.round(read_variant_depth)

                # encode reads as alleles and probabilities
                read_calls = encode_read_alleles(locus, read_chars)
                data.sampledata["read_calls"][sample] = read_calls
                if self.ignore_base_phred_scores:
                    read_quals = None
                read_dists = encode_read_distributions(
                    locus,
                    read_calls,
                    read_quals,
                    error_rate=self.base_error_rate,
                )
                data.sampledata["RCALLS"][sample] = np.sum(read_calls >= 0)

                # de-duplicate reads
                read_dists_unique, read_dist_counts = mset.unique_counts(read_dists)
                data.sampledata["read_dists_unique"][sample] = read_dists_unique
                data.sampledata["read_dist_counts"][sample] = read_dist_counts

            # end of try clause for specific sample
            except Exception as e:
                message = SAMPLE_ASSEMBLY_ERROR.format(sample=sample)
                raise SampleAssemblyError(message) from e
        return data

    def call_sample_genotypes(self, data):
        raise NotImplementedError()

    def sumarise_sample_genotypes(self, data):
        """Computes some statistics comparing called genotypes and haplotypes
        to initial read sequences.

        Parameters
        ----------
        data : LocusAssemblyData
            With sampledata fields: "alleles", "haplotypes", "read_calls".

        Returns
        -------
        data : LocusAssemblyData
            With sampledata fields: "GT" "MEC" "MECP".
        """
        for field in ["GT", "MEC", "MECP"]:
            data.sampledata[field] = dict()
        for sample in data.samples:
            # wrap in try clause to pass sample info back with any exception
            try:
                alleles = data.sampledata["alleles"][sample]
                genotype = data.sampledata["haplotypes"][sample]
                read_calls = data.sampledata["read_calls"][sample]
                gt = "/".join([str(a) if a >= 0 else "." for a in alleles])
                mec = np.sum(integer.minimum_error_correction(read_calls, genotype))
                mec_denom = np.sum(read_calls >= 0)
                mecp = (
                    np.round(mec / mec_denom, self.precision)
                    if mec_denom > 0
                    else np.nan
                )
                data.sampledata["GT"][sample] = gt
                data.sampledata["MEC"][sample] = mec
                data.sampledata["MECP"][sample] = mecp
            except Exception as e:
                message = SAMPLE_ASSEMBLY_ERROR.format(sample=sample)
                raise SampleAssemblyError(message) from e
        return data

    def sumarise_vcf_record(self, data):
        """Generate VCF record fields.

        Parameters
        ----------
        data : LocusAssemblyData
            With sampledata fields: .

        Returns
        -------
        data : LocusAssemblyData
            With infodata fields:
            "END", "NVAR", "SNVPOS", "AC", "AN", "NS", "DP", "RCOUNT"
            and "AFP" or "AOP" if specified.
        """
        # postions
        data.infodata["END"] = data.locus.stop
        data.infodata["NVAR"] = len(data.locus.variants)
        data.infodata["SNVPOS"] = (
            np.subtract(data.locus.positions, data.locus.start) + 1
        )
        # if no filters applied then locus passed
        if len(data.columndata["FILTER"]) == 0:
            data.columndata["FILTER"] = vcf.filters.PASS.id
        # alt allele counts
        allele_counts = np.zeros(len(data.columndata["ALTS"]) + 1, int)
        for array in data.sampledata["alleles"].values():
            for a in array:
                # don't count null alleles
                if a >= 0:
                    allele_counts[a] += 1
        data.infodata["AC"] = allele_counts[1:]  # skip ref count
        # total number of alleles in called genotypes
        data.infodata["AN"] = np.sum(allele_counts)
        # number of called samples
        data.infodata["NS"] = sum(
            np.any(a >= 0) for a in data.sampledata["alleles"].values()
        )
        # total read depth and allele depth
        if len(data.locus.variants) == 0:
            # it will be misleading to return a depth of 0 in this case
            data.infodata["DP"] = np.nan
        else:
            data.infodata["DP"] = np.nansum(list(data.sampledata["DP"].values()))
        # total read count
        data.infodata["RCOUNT"] = np.nansum(list(data.sampledata["RCOUNT"].values()))
        n_allele = len(data.columndata["ALTS"]) + 1
        null_length_R = np.full(n_allele, np.nan)
        if "ACP" in data.infofields:
            _ACP = sum(data.sampledata["AFP"].values())
            _ACP = null_length_R if np.isnan(_ACP).all() else _ACP
            data.infodata["ACP"] = _ACP.round(self.precision)
        if "AFP" in data.infofields:
            # use ACP to weight frequencies of each individual by ploidy
            _AFP = sum(data.sampledata["ACP"].values()) / sum(
                data.sample_ploidy.values()
            )
            _AFP = null_length_R if np.isnan(_AFP).all() else _AFP
            data.infodata["AFP"] = _AFP.round(self.precision)
        if "AOPSUM" in data.infofields:
            _AOPSUM = sum(data.sampledata["AOP"].values())
            _AOPSUM = null_length_R if np.isnan(_AOPSUM).all() else _AOPSUM
            data.infodata["AOPSUM"] = _AOPSUM.round(self.precision)
        if "AOP" in data.infofields:
            prob_not_occurring = np.ones(len(data.columndata["ALTS"]) + 1, float)
            for occur in data.sampledata["AOP"].values():
                prob_not_occurring = prob_not_occurring * (1 - occur)
            prob_occurring = 1 - prob_not_occurring
            data.infodata["AOP"] = prob_occurring.round(self.precision)
        if "SNVDP" in data.infofields:
            _SNVDP = sum(data.sampledata["SNVDP"].values())
            data.infodata["SNVDP"] = _SNVDP.round(self.precision)
        return data

    def call_locus(self, locus, sample_bams):
        """Call samples at a locus and formats resulting data
        into a VCF record line.

        Parameters
        ----------
        locus
            Assembly target locus.
        samples : list
            Sample identifiers.
        sample_bams : dict
            Map for sample identifiers to bam path.
        sample_ploidy : dict
            Map of sample identifiers to ploidy.
        sample_inbreeding : dict
            Map of sample identifiers to inbreeding.

        Returns
        -------
        vcf_record : str
            VCF variant line.

        """
        data = self._locus_data(locus, sample_bams)
        self.encode_sample_reads(data)
        self.call_sample_genotypes(data)
        self.sumarise_sample_genotypes(data)
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
            if line == "KILL":
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

        queue.put("KILL")
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
    infofields: list
    formatfields: list
    columndata: dict
    infodata: dict
    sampledata: dict

    def _sampledata_as_list(self, field):
        data = self.sampledata.get(field, dict())
        return [data.get(s) for s in self.samples]

    def format_vcf_record(self):
        info_data = OrderedDict()
        for field in self.infofields:
            info_data[field] = self.infodata.get(field)
        info_string = vcf.format_info_field(**info_data)
        format_data = OrderedDict()
        for field in self.formatfields:
            format_data[field] = self._sampledata_as_list(field)
        format_string = vcf.format_sample_field(**format_data)
        return vcf.format_record(
            chrom=self.locus.contig,
            pos=self.locus.start + 1,  # 0-based BED to 1-based VCF
            id=self.locus.name,
            ref=self.columndata.get("REF"),
            alt=self.columndata.get("ALTS"),
            qual=self.columndata.get("QUAL"),
            filter=self.columndata.get("FILTER"),
            info=info_string,
            format=format_string,
        )
