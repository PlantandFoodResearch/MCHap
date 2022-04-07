import sys
import numpy as np
from dataclasses import dataclass
import pysam
import multiprocessing as mp
from collections import OrderedDict

from mchap import mset
from mchap.encoding import character, integer
from mchap.io import (
    Locus,
    extract_read_variants,
    encode_read_alleles,
    encode_read_distributions,
    vcf,
)
from mchap.io.vcf.infofields import HEADER_INFO_FIELDS
from mchap.io.vcf.formatfields import HEADER_FORMAT_FIELDS

import warnings

warnings.simplefilter("error", RuntimeWarning)


LOCUS_ASSEMBLY_ERROR = (
    "Exception encountered at locus: '{name}', '{contig}:{start}-{stop}'."
)
SAMPLE_ASSEMBLY_ERROR = (
    "Exception encountered when assembling sample '{sample}' from file '{bam}'."
)


class LocusAssemblyError(Exception):
    pass


class SampleAssemblyError(Exception):
    pass


@dataclass
class program(object):
    vcf: str
    samples: list
    sample_bams: dict
    sample_ploidy: dict
    sample_inbreeding: dict
    read_group_field: str = "SM"
    base_error_rate: float = 0.0
    ignore_base_phred_scores: bool = False
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
            "NS",
            "DP",
            "RCOUNT",
            "END",
            "NVAR",
            "SNVPOS",
        ] + [f for f in ["AFP"] if f in self.report_fields]
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
            "KMERCOV",
            "GPM",
            "PHPM",
            "MCI",
        ] + [f for f in ["GP", "GL", "AFP", "DS"] if f in self.report_fields]
        return formatfields

    def loci(self):
        with pysam.VariantFile(self.vcf) as f:
            for record in f.fetch():
                locus = Locus.from_variant_record(record)
                yield locus

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
            vcf.filters.SamplePassFilter(),
        ]
        info_fields = [HEADER_INFO_FIELDS[field] for field in self.info_fields()]
        format_fields = [HEADER_FORMAT_FIELDS[field] for field in self.format_fields()]
        columns = [vcf.headermeta.columns(self.samples)]
        header = meta_fields + contigs + filters + info_fields + format_fields + columns
        return [str(line) for line in header]

    def _locus_data(self, locus):
        """Generate a LocusAssemblyData object for a given locus
        to be populated with data relating to a single vcf record.
        """
        infofields = self.info_fields()
        formatfields = self.format_fields()
        return LocusAssemblyData(
            locus=locus,
            samples=self.samples,
            sample_bams=self.sample_bams,
            sample_ploidy=self.sample_ploidy,
            sample_inbreeding=self.sample_inbreeding,
            infofields=infofields,
            formatfields=formatfields,
            columndata=dict(),
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
            "read_calls",
            "read_dists_unique",
            "read_dist_counts",
        ]:
            data.sampledata[field] = dict()
        locus = data.locus
        for sample in data.samples:

            # path to bam for this sample
            path = data.sample_bams[sample]

            # wrap in try clause to pass sample info back with any exception
            try:

                # extract read data
                read_chars, read_quals = extract_read_variants(
                    data.locus,
                    path,
                    id=self.read_group_field,
                    min_quality=self.mapping_quality,
                    skip_duplicates=self.skip_duplicates,
                    skip_qcfail=self.skip_qcfail,
                    skip_supplementary=self.skip_supplementary,
                )[sample]

                # get read stats
                read_count = read_chars.shape[0]
                data.sampledata["RCOUNT"][sample] = read_count
                read_variant_depth = character.depth(read_chars)
                if len(read_variant_depth) == 0:
                    # no variants to score depth
                    data.sampledata["DP"][sample] = np.nan
                else:
                    data.sampledata["DP"][sample] = np.round(
                        np.mean(read_variant_depth)
                    )

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
                path = data.sample_bams.get(sample)
                message = SAMPLE_ASSEMBLY_ERROR.format(sample=sample, bam=path)
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
            With sampledata fields: "GT" "MEC" "KMERCOV".
        """
        for field in ["GT", "MEC", "KMERCOV"]:
            data.sampledata[field] = dict()
        for sample in data.samples:
            # wrap in try clause to pass sample info back with any exception
            try:
                alleles = data.sampledata["alleles"][sample]
                genotype = data.sampledata["haplotypes"][sample]
                read_calls = data.sampledata["read_calls"][sample]
                gt = "/".join([str(a) if a >= 0 else "." for a in alleles])
                mec = np.sum(integer.minimum_error_correction(read_calls, genotype))
                cov = np.round(
                    integer.min_kmer_coverage(
                        read_calls,
                        genotype,
                        ks=[1, 2, 3],
                    ),
                    self.precision,
                )
                data.sampledata["GT"][sample] = gt
                data.sampledata["MEC"][sample] = mec
                data.sampledata["KMERCOV"][sample] = cov
            except Exception as e:
                path = data.sample_bams.get(sample)
                message = SAMPLE_ASSEMBLY_ERROR.format(sample=sample, bam=path)
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
            With columndata fields: "REF" "ALTS" and infodata fields:
            "END", "NVAR", "SNVPOS", "AC", "AN", "NS", "DP", "RCOUNT"
            and "AFP" if specified.
        """
        # postions
        data.infodata["END"] = data.locus.stop
        data.infodata["NVAR"] = len(data.locus.variants)
        data.infodata["SNVPOS"] = (
            np.subtract(data.locus.positions, data.locus.start) + 1
        )
        # sequences
        data.columndata["REF"] = data.locus.sequence
        data.columndata["ALTS"] = data.locus.alts
        # alt allele counts
        allele_counts = np.zeros(len(data.columndata["ALTS"]) + 1, int)
        for array in data.sampledata["alleles"].values():
            for a in array:
                # don't count null alleles
                if a >= 0:
                    allele_counts[a] += 1
        data.infodata["AC"] = allele_counts[1:]  # skip ref count
        # total number of alleles in called genotypes
        data.infodata["AN"] = np.sum(allele_counts > 0)
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
        if "AFP" in data.infofields:
            data.infodata["AFP"] = np.mean(
                list(data.sampledata["AFP"].values()), axis=0
            ).round(self.precision)
        return data

    def call_locus(self, locus):
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
        data = self._locus_data(locus)
        self.encode_sample_reads(data)
        self.call_sample_genotypes(data)
        self.sumarise_sample_genotypes(data)
        self.sumarise_vcf_record(data)
        return data.format_vcf_record()

    def _assemble_locus_wrapped(self, locus):
        try:
            result = self.call_locus(locus)
        except Exception as e:
            message = LOCUS_ASSEMBLY_ERROR.format(
                name=locus.name, contig=locus.contig, start=locus.start, stop=locus.stop
            )
            raise LocusAssemblyError(message) from e
        return result

    def run(self):
        header = self.header()
        pool = mp.Pool(self.n_cores)
        jobs = ((locus,) for locus in self.loci())
        records = pool.starmap(self._assemble_locus_wrapped, jobs)
        return header + records

    def _worker(self, locus, queue):
        line = str(self._assemble_locus_wrapped(locus))
        queue.put(line)
        return line

    def _writer(self, queue):
        while True:
            line = queue.get()
            if line == "KILL":
                break
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def _run_stdout_single_core(self):
        header = self.header()
        for line in header:
            sys.stdout.write(line + "\n")
        for locus in self.loci():
            line = self._assemble_locus_wrapped(locus)
            sys.stdout.write(line + "\n")

    def _run_stdout_multi_core(self):

        header = self.header()

        for line in header:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()

        manager = mp.Manager()
        queue = manager.Queue()
        pool = mp.Pool(self.n_cores)

        # start writer process
        _ = pool.apply_async(self._writer, (queue,))

        jobs = []
        for locus in self.loci():
            job = pool.apply_async(self._worker, (locus, queue))
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
