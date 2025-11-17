# Changelog

## Unreleased 

## Beta v0.11.1

Maintenance:
- Switch to `pyproject.toml` #188
- Set versions manually without SCM #188
- Update to Numpy 2 and Pandas 2
- Add Python 3.13 to build matrix

## Beta v0.11.0

Breaking Change:
- Changed the default prior used in `assemble` to a flat prior across genotypes.
  This is a more sensible prior in almost all situations because the previous
  Dirichlet-multinomial prior had to assume that all possible haplotypes (i.e.,
  combinations of SNVS) where equally likely to be present in the sample population.
  In general, this inflated the prior probability of heterozygous genotypes.
  Note that this is not a significant issue when using the `call` program with
  known haplotypes (and a reasonable estimate of their frequencies).
  The previous Dirichlet-multinomial prior can still be used via the
  `--use-dirmul-prior` argument.
- Changed the default prior used in `call` and `call-exact` to a flat prior across
  genotypes. This is a better default option than the previous Dirichlet-multinomial
  prior when sample inbreeding and prior allele frequencies are not specified.
  The previous Dirichlet-multinomial prior can still be used via the `--use-dirmul-prior`
  argument which is now used to set both the sample inbreeding and prior on allele
  frequencies. This new argument replaces the `--inbreeding` and `--prior-frequencies`
  arguments.

## Beta v0.10.0 

New Features:
- New experimental `atomize` tool for splitting haplotypes into basis SNVs #72. 
- New experimental `call-pedigree` tool fo pedigree informed genotype calling.
- Optionally specify just the `INFO` or `FORMAT` variant of a optional VCF field.
- Use `setuptools_scm` for versioning #179.

VCF Changes:
- Renamed `PHQ` and `PHPM` to `SQ` and `SPM` for clarity.
- Added `INFO/UAN` field for number of unique alleles called #174.
- Added `INFO/MCI` field for proportion of sample with Markov Chain incongruence.
- Added optional fields #174:
    * `INFO/AOPSUM` (sum of `FORMAT/AOP`).
    * `INFO/ACP` and `FORMAT/ACP`.
    * `INFO/SNVDP` and `FORMAT/SNVDP`.


## Beta v0.9.3

Bug Fixes:
- Correct usage of the `AN` field #176
- Improved error messages for io #163 #177

## Beta v0.9.2

Bug Fixes:
- Avoid holding alignment file handles open #173


## Beta v0.9.1

New Features:
- Allow complex sample pooling via a tabular file


## Beta v0.9.0

New Features:
- Added tool `mchap find-snvs` to generate a template VCF for assembly #166
- Option to report posterior probability of allele occurrence #162
- Added generic option to filter input haplotypes #168

Bug Fixes:
- Allow samples with multiple read groups #164
- Correct number of cores used when specifying multiple cores #150
- Simplify specification of prior allele frequencies #154
- Improve performance when working with CRAM files #167

CLI Changes:
- Added `mchap find-snvs` tool #166
- Added optional `--reference` argument to call and call-exact tools #167
- Replaced `--skip-rare-haplotypes` argument with `--filter-input-haplotypes` #168
- Replaced `--haplotype-frequencies` and `--haplotype-frequencies-prior` with `--prior-frequencies` #154

VCF Changes:
- Added `AOP` field to record posterior probability of an allele occurring at any copy number #162

Internal Changes:
- Changes to using multiple process to minimize file handel creation #167



## Beta v0.8.1

Bug Fixes:
- Fixed integer overflow bug when calculating the total number of unique haplotypes in mchap assemble #157

Internal Changes:
- Minor performance improvement to SNP homozygosity testing in mchap assemble



## Beta v0.8.0

New Features:
- Combine `--bam`, `--bam-list` and `--sample-bam` arguments #128
- Combine `--ploidy` and `--sample-ploidy` arguments #128
- Combine `--inbreeding` and `--sample-inbreeding` arguments #128
- Combine `--mcmc-temperatures` and `--sample-mcmc-temperatures` arguments #128
- Improvements to documentation



## Beta v0.7.0

New Features:
- Mask reference allele when it is only reported to satisfy VCF spec #146
- Optionally report prior allele frequencies in `AFPRIOR` field
- Filtering for some edge cases where genotypes should not be reported (`AF0` and `NOA`)

Bug Fixes:
- Handle edge-case where all prior allele frequencies are zero #145

VCF Changes:
- Added `REFMASKED` info flag to indicate reference allele is amsked and should be ignored
- Added `AFPRIOR` infor filed to indicate prior allele frequencies
- Added `NOA` filter to indicate loci where no alleles were observed (e.g., masked reference only)
- Added `AF0` filter to indicate invalid prior allele frequencies in which all frequencies were zero



## Beta v0.6.0

New Features:
- Optionally output posterior allele frequencies #135
- Optionally specify a prior allele frequencies in `mchap call` and `mchap call-exact` #120
- Optionally filter input haplotypes by input frequencies in `mchap call` and `mchap call-exact` #113
- Improve PMF performance for `mchap call` and `mchap call-exact` #125
- Allow pooling of all samples into single sample #140
- Allow arbitrary ploidy in PMFs #124

CLI Changes:
- Default to constant base error rate without using phred scores #127
- Replaced `--ignore-base-phred-scores` with `--use-base-phred-scores` #127
- Increased default MCMC steps and burnin to 2000 and 1000 respectively #137
- Removed `--sample-list` argument #128
- Merged `--genotype-likelihoods` and `--genotype-posteriors` into `--report` #128
- Added option to report `AFP` via `--report` #135
- Added `--sample-pool` argument #140
- Added `--haplotype-frequencies` parameter to `mchap call` and `mchap call-exact`
- Added `--haplotype-frequencies-prior` flag to `mchap call` and `mchap call-exact`
- Added `--skip-rare-haplotypes` parameter to `mchap call` and `mchap call-exact`
- Removed `pedigraph` tool #138

VCF Changes:
- Replaced `KMERCOV` with `MECP` (`MEC` / `RCALLS`) #134
- Optional `AFP` field to report posterior allele frequencies #135



## Beta v0.5.1

Bug Fixes:
- Fix addition of null allele counts to final allele #115
- Fix integer overflow with many haplotypes #117


## Beta v0.5.0

New Features:
- Added Gibbs sampler re-calling tool: `mchap call` #110, #104
- Added exact re-calling tool:mchap `call-exact` #110

CLI Changes:
- Removed exact re-calling option from `mchap assemble` #110
- Reordering of some CLI arguments to facilitate reusing arguments between sub-tools

Internal Changes:
- Move CLI parser arguments into reusable components
- Reuse CLI tool methods via inheritance
- Prior and likelihood functions for known sets of haplotypes
- Generate Locus objects from pysam variant records
- Reorganize internal sub-modules



## Beta v0.4.2

New Features:
- Per-sample specification of temperatures for parallel-tempering #99



## Beta v0.4.1

VCF Changes:
- Removed `FORMAT/FT` from header #101
- Removed `INFO/AD` and `FORMAT/AD` #98
- Dynamically omit `FORMAT/GL` and `FORMAT/GP` fields from VCF when they are not used #102



## Beta v0.4.0

New Features:
- Haplotype inclusion based on haplotype posterior threshold rather than genotype/phenotype #93
- Recalling genotypes based on observed haplotypes #93
- Removed per-sample filters and replaced with metadata for downstream filtering #96
- Options to call full genotype posterior or likelihoods over all genotypes #93
- Improve specification of per-locus assembly for better integration with asub #90
    - `--region` and `--region-id` arguments
    - `--sample-bams` argument
- Mechanism to cache log-likelihoods for reuse #21
- Enable caching of JIT compiled functions #94

Bug Fixes:
- Fix bug in which read counts are ignored when excluding SNPs based on --mcmc-fix-homozygous threshold

VCF Changes:
- Added fields:
    `INFO/DP`
    `INFO/RCOUNT`
    `INFO/NVAR`
    `FORMAT/KMERCOV`
    `FORMAT/MCI`
    `FORMAT/GL`
    `FORMAT/GP`
- Removed fields:
    `FORMAT/DOSEXP`



## Beta v0.3.0

New Features:
- Parallel-tempering #65
- Option to use inbreeding coefficient to inform prior #75
- Separate recombination and dosage swap sub-steps
- Control of recombination and dosage swap sub-steps via probabilities
- Arguments to exclude/include duplicate, qcfail and supplementary reads #74
- Added AD format field #63
- Added filters for chain in-congruence #69 #70
- Specification on constant error rate via `--base-error-rate` and `--ignore-base-phred-scores` #83

VCF Changes:
- Added AD format field #63
- Added filters for chain in-congruence #69 #70
- Rename format fields `MPED` to `DOSEXP` and `PPM` to `PHPM`
- Remove `MPGP` and `RASSIGN` format fields #63
- Fix off-by-1 error of variant positions in VCF output

Bug Fixes:
- Added missing dependencies #61
- Correction for null prior probability
- Correct Metropolis-Hastings transition probabilities #66
- Improve single core method for running assemble program
- Use simple plain-text pedigree format for pedigraph tool #49
- Correct use or region string argument in pedigraph tool #42
- Fix off-by-1 error of variant positions in VCF output
- Replace ndarray tostring calls with tobytes #62

Internal Changes:
- Add minimal set of functions to top level API #77
- Simplify main loop of assemble program #76
- Simplify MCMC file hierarchy
- Enforce formatting with Black and Flake8 #78
- Optimization by de-duplication of identical reads #83
- Removal of old unused "brute force" assembler
- Improved exception handling during assembly #88



## Alpha v0.2.3

- Fixes bug in filter introduced in v0.2.2



## Alpha v0.2.2

- Add filter for chain incongruence #64
- Add filter to identify CNV across chains #69
- Fix potential bug with masked allele llk from np.empty



## Alpha v0.2.1

- Add missing sub-module for statistics on integer encoded sequences



## Alpha v0.2.0

- Use non-normalized probability encodings for reads #56
- Remove `probabilistic` encoding sub-module
- Rename `allelic` and `symbolic` encoding sub-modules to integer and character
- Add `MEC` score to VCF output #58
- Rename `RC` format field to `RCOUNT`
- Add `RCALLS` format field to record the number of variant encoding bases among reads
- Add `RASSIGN` format field to record approximate read assignment to haplotypes based on MEC
- Improve string formatting of floats in VCF output to remove .0 from values without decimals



## Alpha v0.1.0

- Initial release
