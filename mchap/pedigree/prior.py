import numpy as np
from math import lgamma
from numba import njit

from mchap.calling.utils import allelic_dosage
from mchap.jitutils import comb, add_log_prob, ln_equivalent_permutations
from mchap.assemble.prior import log_genotype_prior
from mchap.calling.prior import calculate_alphas


@njit(cache=True)
def parental_copies(parent_alleles, progeny_alleles):
    """Count the number of parental copies of each allele present
    with a progeny genotype.

    Parameters
    ----------
    parent_alleles : ndarray, int, shape (ploidy,)
        Alleles observed within parent.
    progeny_alleles : ndarray, int, shape (ploidy,)
        Alleles observed within progeny.

    Returns
    -------
    parental_copies : ndarray, int , shape (ploidy,)
        The number of parental copies of each allele observed
        in the progeny.

    Notes
    -----
    Returned counts correspond to the first instance of each
    progeny allele and subsequent copies of that allele will
    correspond to a count of zero.
    """
    parent_ploidy = len(parent_alleles)
    progeny_ploidy = len(progeny_alleles)
    copies = np.zeros_like(progeny_alleles)
    for i in range(parent_ploidy):
        a = parent_alleles[i]
        for j in range(progeny_ploidy):
            if a == progeny_alleles[j]:
                copies[j] += 1
                break
    return copies


@njit(cache=True)
def dosage_permutations(dosage, constraint):
    """Count the number of possible permutations in which
    the observed dosage can be drawn from a constraint dosage
    without replacement.

    Parameters
    ----------
    dosage : ndarray, int, shape (ploidy,)
        Counts of each unique allele in a genotype.
    constraint : ndarray, int, shape (ploidy,)
        Initial count of each allele.

    Returns
    -------
    n : int
        Number of permutations.

    Notes
    -----
    This function can be used to determine the number of
    possible permutations in which a gametes observed
    dosage can be drawn from the parental (constraint) dosage.
    """
    n = 1
    for i in range(len(dosage)):
        n *= comb(constraint[i], dosage[i])
    return n


@njit(cache=True)
def initial_dosage(ploidy, constraint):
    """Calculate the initial dosage that fits within a constraint.

    Parameters
    ----------
    ploidy : int
        Number of alleles in dosage array.
    constraint : ndarray, int, shape (ploidy,)
        Max count of each allele.

    Returns
    -------
    dosage : ndarray, int, shape (ploidy,)
        Counts of each unique allele in a genotype.
    """
    dosage = np.zeros_like(constraint)
    for i in range(len(constraint)):
        count = min(ploidy, constraint[i])
        dosage[i] = count
        ploidy -= count
    if ploidy > 0:
        raise ValueError("Ploidy does not fit within constraint")
    return dosage


@njit(cache=True)
def increment_dosage(dosage, constraint):
    """Increment a given dosage to the next possible dosage
    within a given constraint.

    Parameters
    ----------
    dosage : ndarray, int, shape (ploidy,)
        Counts of each unique allele in a genotype.
    constraint : ndarray, int, shape (ploidy,)
        Max count of each allele.

    Raises
    ------
    ValueError
        If there are no further dosage options.

    Notes
    -----
    The dosage is incremented in place.
    """
    ploidy = len(dosage)
    i = ploidy - 1
    change = 0
    # find last non-zero value
    while dosage[i] == 0:
        i -= 1
    # lower that value
    dosage[i] -= 1
    change += 1
    # raise first available value to its right
    j = i + 1
    while (j < ploidy) and (change > 0):
        if dosage[j] < constraint[j]:
            dosage[j] += 1
            change -= 1
        j += 1
    # if no value was available to the right
    if change > 0:
        # zero out last value
        change += dosage[i]
        dosage[i] = 0
        space = constraint[i]
        # find next positive value to its left with enough space remaining
        searching = True
        while searching:
            i -= 1
            if i < 0:
                raise ValueError("Final dosage")
            if (dosage[i] > 0) and (space > change):
                dosage[i] -= 1
                change += 1
                searching = False
            else:
                space += constraint[i]
                change += dosage[i]
                dosage[i] = 0
        # fill to the right
        j = i + 1
        while change > 0:
            value = min(constraint[j] - dosage[j], change)
            dosage[j] += value
            change -= value
            j += 1
    return


@njit(cache=True)
def second_gamete_log_pmf(gamete_dose, constant_dose, n_alleles, inbreeding):
    """Log probability of an gamete of unknown origin given a known gamete and the expected
    inbreeding coefficient.

    Parameters
    ----------
    gamete_dose : ndarray, int, shape (ploidy, )
        Allele counts of the proposed gamete.
    constant_dose : ndarray, int, shape (ploidy, )
        Allele counts of the known gamete to be held as constant.

    n_alleles : int
        Number of possible alleles at this locus.
    inbreeding : float
        Expected inbreeding coefficient of the sample.

    Returns
    -------
    lprob : float
        Log-probability of the proposed gamete given the known gamete.
    """
    # TODO: prior frequencies
    assert 0 <= inbreeding < 1
    assert len(gamete_dose) == len(constant_dose)
    gamete_tau = gamete_dose.sum()
    constant_tau = constant_dose.sum()

    # if not inbred use null prior
    if inbreeding == 0:
        ln_perms = ln_equivalent_permutations(gamete_dose)
        return ln_perms - gamete_tau * np.log(n_alleles)

    # calculate the dispersion parameter for the PMF
    alpha_const = calculate_alphas(inbreeding, 1 / n_alleles)

    # alpha of each allele is the base alpha plus constant dose
    sum_alphas = constant_tau + alpha_const * n_alleles

    # left side of equation in log space
    num = lgamma(gamete_tau + 1) + lgamma(sum_alphas)
    denom = lgamma(gamete_tau + sum_alphas)
    left = num - denom

    # right side of equation
    prod = 0.0  # log(1.0)
    for i in range(len(gamete_dose)):
        dose = gamete_dose[i]
        if dose > 0:
            alpha_i = alpha_const + constant_dose[i]
            num = lgamma(dose + alpha_i)
            denom = lgamma(dose + 1) + lgamma(alpha_i)
            prod += num - denom

    # return as log probability
    return left + prod


@njit(cache=True)
def trio_log_pmf(
    progeny,
    parent_p,
    parent_q,
    tau_p,
    tau_q,
    error_p,
    error_q,
    inbreeding,
    n_alleles,
):
    """Log probability of a trio of genotypes.

    Parameters
    ----------
    progeny : ndarray, int, shape (ploidy,)
        Integer encoded alleles in the progeny genotype.
    parent_p : ndarray, int, shape (ploidy,)
        Integer encoded alleles in the first parental genotype.
    parent_q : ndarray, int, shape (ploidy,)
        Integer encoded alleles in the second parental genotype.
    tau_p : int
        Number of alleles inherited from parent_p.
    tau_q : int
        Number of alleles inherited from parent_q.
    error_p : float
        Probability that parent_p is not the correct
        parental genotype.
    error_q : float
        Probability that parent_q is not the correct
        parental genotype.
    inbreeding : float
        Expected inbreeding coefficient of the sample.
    n_alleles : int
        Number of possible alleles at this locus.

    Returns
    -------
    lprob : float
        Log-probability of the trio.

    Notes
    -----
    In the case that one or both parental genotypes are incorrect
    (as encoded by the error terms) this function assumes that the
    progeny genotype has the specified inbreeding coefficient and that
    the gametes of unknown origin are drawn from a background population
    in which all alleles are equally frequent.
    The inbreeding coefficient is does not inform a case in which both
    parents are correct (i.e., when the error terms are zero).
    """
    dosage = allelic_dosage(progeny)

    ploidy_p = len(parent_p)
    ploidy_q = len(parent_q)

    dosage_p = parental_copies(parent_p, progeny)
    dosage_q = parental_copies(parent_q, progeny)

    constraint_p = np.minimum(dosage, dosage_p)
    constraint_q = np.minimum(dosage, dosage_q)

    valid_p = constraint_p.sum() >= tau_p
    valid_q = constraint_q.sum() >= tau_q

    lerror_p = np.log(error_p)
    lerror_q = np.log(error_q)
    lcorrect_p = np.log(1 - error_p)
    lcorrect_q = np.log(1 - error_q)

    lprob = -np.inf

    # assuming both parents are valid
    if valid_p and valid_q:
        gamete_p = initial_dosage(tau_p, constraint_p)
        gamete_q = dosage - gamete_p
        while True:
            # assuming both parents are valid
            lprob_p = (
                np.log(dosage_permutations(gamete_p, dosage_p) / comb(ploidy_p, tau_p))
                + lcorrect_p
            )
            lprob_q = (
                np.log(dosage_permutations(gamete_q, dosage_q) / comb(ploidy_q, tau_q))
                + lcorrect_q
            )
            lprob_pq = lprob_p + lprob_q
            lprob = add_log_prob(lprob, lprob_pq)
            # assuming p valid and q invalid (avoids iterating gametes of p twice)
            # probability of gamete_q given gamete_p
            lprob_q = (
                second_gamete_log_pmf(
                    gamete_q, gamete_p, n_alleles=n_alleles, inbreeding=inbreeding
                )
                + lerror_q
            )
            lprob_pq = lprob_p + lprob_q
            lprob = add_log_prob(lprob, lprob_pq)
            # increment by gamete of p
            try:
                increment_dosage(gamete_p, constraint_p)
            except:  # noqa: E722
                break
            else:
                for i in range(len(gamete_q)):
                    gamete_q[i] = dosage[i] - gamete_p[i]

    # assuming p valid and q invalid (unless already done in previous loop)
    elif valid_p:
        gamete_p = initial_dosage(tau_p, constraint_p)
        gamete_q = dosage - gamete_p
        while True:
            lprob_p = (
                np.log(dosage_permutations(gamete_p, dosage_p) / comb(ploidy_p, tau_p))
                + lcorrect_p
            )
            # probability of gamete_q given gamete_p
            lprob_q = (
                second_gamete_log_pmf(
                    gamete_q, gamete_p, n_alleles=n_alleles, inbreeding=inbreeding
                )
                + lerror_q
            )
            lprob_pq = lprob_p + lprob_q
            lprob = add_log_prob(lprob, lprob_pq)
            # increment by gamete of p
            try:
                increment_dosage(gamete_p, constraint_p)
            except:  # noqa: E722
                break
            else:
                for i in range(len(gamete_q)):
                    gamete_q[i] = dosage[i] - gamete_p[i]

    # assuming p invalid and q valid
    if valid_q:
        gamete_q = initial_dosage(tau_q, constraint_q)
        gamete_p = dosage - gamete_q
        while True:
            # probability of gamete_p given gamete_q
            lprob_p = (
                second_gamete_log_pmf(
                    gamete_p, gamete_q, n_alleles=n_alleles, inbreeding=inbreeding
                )
                + lerror_p
            )
            lprob_q = (
                np.log(dosage_permutations(gamete_q, dosage_q) / comb(ploidy_q, tau_q))
                + lcorrect_q
            )
            lprob_pq = lprob_p + lprob_q
            lprob = add_log_prob(lprob, lprob_pq)
            # increment by gamete of q
            try:
                increment_dosage(gamete_q, constraint_q)
            except:  # noqa: E722
                break
            else:
                for i in range(len(gamete_p)):
                    gamete_p[i] = dosage[i] - gamete_q[i]

    # assuming both parents are invalid
    lprob_pq = (
        log_genotype_prior(dosage, n_alleles, inbreeding=inbreeding)
        + lerror_p
        + lerror_q
    )
    lprob = add_log_prob(lprob, lprob_pq)
    return lprob


@njit(cache=True)
def markov_blanket_log_probability(
    target_index,
    sample_genotypes,
    sample_ploidy,
    sample_inbreeding,
    sample_parents,
    gamete_tau,
    gamete_error,
    n_alleles,
):
    """Joint probability of pedigree items that fall within the
    Markov blanket of the specified target sample.

    Parameters
    ----------
    target_index : int
        Index of target sample.
    sample_genotypes : ndarray, int, shape (n_sample, max_ploidy)
        Genotype of each sample padded by negative values.
    sample_ploidy  : ndarray, int, shape (n_sample,)
        Sample ploidy
    sample_inbreeding  : ndarray, float, shape (n_sample,)
        Expected inbreeding coefficients
    sample_parents : ndarray, int, shape (n_samples, 2)
        Parent indices of each sample with -1 indicating
        unknown parents.
    gamete_tau : int, shape (n_samples, 2)
        Gamete ploidy associated with each pedigree edge.
    gamete_error : float, shape (n_samples, 2)
        Error rate associated with each pedigree edge.
    n_alleles : int
        Number of possible haplotype alleles at this locus.

    Returns
    -------
    log_probability : float
        Joint log probability of pedigree items that fall within the
        Markov blanket of the specified target sample.

    """
    n_samples, _ = sample_genotypes.shape
    assert 0 <= target_index < n_samples
    log_joint = 0.0
    for i in range(n_samples):
        p = sample_parents[i, 0]
        q = sample_parents[i, 1]
        if (target_index == i) or (target_index == p) or (target_index == q):
            if p >= 0:
                error_p = gamete_error[i, 0]
                genotype_p = sample_genotypes[p, 0 : sample_ploidy[p]]
            else:
                error_p = 1.0
                genotype_p = np.array([-1], dtype=sample_genotypes.dtype)
            if q >= 0:
                error_q = gamete_error[i, 1]
                genotype_q = sample_genotypes[q, 0 : sample_ploidy[q]]
            else:
                error_q = 1.0
                genotype_q = np.array([-1], dtype=sample_genotypes.dtype)
            genotype_i = sample_genotypes[i, 0 : sample_ploidy[i]]
            log_joint += trio_log_pmf(
                genotype_i,
                genotype_p,
                genotype_q,
                tau_p=gamete_tau[i, 0],
                tau_q=gamete_tau[i, 1],
                error_p=error_p,
                error_q=error_q,
                inbreeding=sample_inbreeding[i],
                n_alleles=n_alleles,
            )
    return log_joint
