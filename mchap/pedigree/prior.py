import numpy as np
from numba import njit

from mchap.jitutils import comb, add_log_prob, ln_equivalent_permutations


@njit(cache=True)
def set_allelic_dosage(genotype_alleles, genotype_dosage):
    """Return the dosage of genotype alleles encoded as integers.

    Parameters
    ----------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Genotype alleles encoded as integers
    genotype_dosage : ndarray, int, shape (ploidy, )
        Array to collect dosage
    """
    max_ploidy = len(genotype_alleles)
    genotype_dosage[:] = 0
    for i in range(max_ploidy):
        a = genotype_alleles[i]
        if a < 0:
            continue
        searching = True
        j = 0
        while searching:
            if a == genotype_alleles[j]:
                genotype_dosage[j] += 1
                searching = False
            else:
                j += 1


@njit(cache=True)
def set_parental_copies(parent_alleles, progeny_alleles, parent_copies):
    """Count the number of parental copies of each allele present
    with a progeny genotype.

    Parameters
    ----------
    parent_alleles : ndarray, int, shape (ploidy,)
        Alleles observed within parent.
    progeny_alleles : ndarray, int, shape (ploidy,)
        Alleles observed within progeny.
    parent_copies : ndarray, int, shape (ploidy,)
        Array to collect counts

    Notes
    -----
    Counts correspond to the first instance of each progeny
    allele and subsequent copies of that allele will
    correspond to a count of zero.
    """
    parent_copies[:] = 0
    for i in range(len(parent_alleles)):
        a = parent_alleles[i]
        if a < 0:
            continue
        for j in range(len(progeny_alleles)):
            if a == progeny_alleles[j]:
                parent_copies[j] += 1
                break
    return parent_copies


@njit(cache=True)
def set_inverse_gamete(dosage, gamete, inverse_gamete):
    for i in range(len(dosage)):
        inverse_gamete[i] = dosage[i] - gamete[i]


@njit(cache=True)
def dosage_frequencies(genotype, frequencies):  # TODO: reuse array
    max_ploidy = len(genotype)
    out = np.full(max_ploidy, np.nan, dtype=np.float64)
    for i in range(max_ploidy):
        a = genotype[i]
        if a >= 0:
            assert a < len(frequencies)
            out[i] = frequencies[a]
    return out


@njit(cache=True)
def log_unknown_dosage_prior(dosage, log_frequencies):
    """

    Parameters
    ----------
    dosage
        Dosage array
    log_frequencies
        Prior frequencies corresponding to dosage
    """
    lperms = ln_equivalent_permutations(dosage)
    assert len(dosage) == len(log_frequencies)
    lperm_prob = 0.0
    for i in range(len(dosage)):
        d = dosage[i]
        if d > 0:
            lperm_prob += log_frequencies[i] * d
    return lperms + lperm_prob


@njit(cache=True)
def log_unknown_const_prior(dosage, allele_index, log_frequencies):
    if dosage[allele_index] > 0:
        dosage[allele_index] -= 1
        lprob = log_unknown_dosage_prior(dosage, log_frequencies)
        dosage[allele_index] += 1
    else:
        lprob = -np.inf
    return lprob


@njit(cache=True)
def dosage_permutations(gamete_dosage, parent_dosage):
    """Count the number of possible permutations in which
    the observed gamete dosage can be drawn from a parent dosage
    without replacement.

    Parameters
    ----------
    gamete_dosage : ndarray, int, shape (ploidy,)
        Counts of each unique allele in a gamete.
    parent_dosage : ndarray, int, shape (ploidy,)
        Counts of each gamete allele in a parent.

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
    for i in range(len(gamete_dosage)):
        n *= comb(parent_dosage[i], gamete_dosage[i])
    return n


@njit(cache=True)
def set_initial_dosage(ploidy, constraint, dosage):
    """Calculate the initial dosage that fits within a constraint.

    Parameters
    ----------
    ploidy : int
        Number of alleles in dosage array.
    constraint : ndarray, int, shape (ploidy,)
        Max count of each allele.
    dosage : ndarray, int, shape (ploidy,)
        Array to collect counts
    """
    for i in range(len(dosage)):
        count = min(ploidy, constraint[i])
        dosage[i] = count
        ploidy -= count
    if ploidy > 0:
        raise ValueError("Ploidy does not fit within constraint")


@njit(cache=True)
def valid_dosage(dosage, constraint):
    """Validate that dosage is possible given constraint.

    Parameters
    ----------
    dosage : ndarray, int, shape (ploidy,)
        Counts of each unique allele in a genotype.
    constraint : ndarray, int, shape (ploidy,)
        Max count of each allele.

    Returns
    -------
    valid : bool
        True if dosage is valid
    """
    for i in range(len(dosage)):
        if dosage[i] > constraint[i]:
            return False
    return True


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
    max_ploidy = len(dosage)
    i = max_ploidy - 1
    change = 0
    # find last non-zero value
    while dosage[i] == 0:
        i -= 1
    # lower that value
    dosage[i] -= 1
    change += 1
    # raise first available value to its right
    j = i + 1
    while (j < max_ploidy) and (change > 0):
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
def duplicate_permutations(gamete_dosage, parent_dosage):
    """Count the number of possible permutations in which the
    observed gamete dosage can be drawn from a parent dosage
    assuming double reduction.

    Parameters
    ----------
    gamete_dosage : ndarray, int, shape (ploidy,)
        Counts of each unique allele in a gamete.
    parent_dosage : ndarray, int, shape (ploidy,)
        Counts of each gamete allele in a parent.

    Returns
    -------
    n : int
        Number of permutations.

    Notes
    -----
    This function is only valid for diploid gametes!
    """
    n = 0
    for i in range(len(gamete_dosage)):
        if gamete_dosage[i] == 2:
            assert n == 0
            n = parent_dosage[i]
        elif gamete_dosage[i] != 0:
            return 0
    return n


@njit(cache=True)
def gamete_log_pmf(
    gamete_dose,
    gamete_ploidy,
    parent_dose,
    parent_ploidy,
    gamete_lambda=0.0,
):
    """Log probability of a gamete drawn from a known genotype.

    Parameters
    ----------
    gamete_dose : ndarray, int
        Counts of each unique allele in the gamete.
    gamete_ploidy : int
        Ploidy (tau) of the gamete.
    parent_dose : ndarray, int
        Counts of each unique gamete allele within the parent genotype.
    parent_ploidy : int
        Ploidy of the parent.
    gamete_lambda : float
        Excess IBD probability of gamete.

    Returns
    -------
    Log-transformed probability of gamete being derived from parental genotype.

    Notes
    -----
    A non-zero lambda value is only supported for diploid gametes.
    """
    prob = (
        dosage_permutations(gamete_dose, parent_dose)
        / comb(parent_ploidy, gamete_ploidy)
    ) * (1 - gamete_lambda)
    if gamete_lambda > 0.0:
        if gamete_ploidy != 2:
            raise ValueError("Lambda parameter is only supported for diploid gametes")
        prob += (
            duplicate_permutations(gamete_dose, parent_dose) / parent_ploidy
        ) * gamete_lambda
    if prob == 0.0:
        return -np.inf
    else:
        return np.log(prob)


@njit(cache=True)
def gamete_const_log_pmf(
    allele_index,
    gamete_dose,
    gamete_ploidy,
    parent_dose,
    parent_ploidy,
    gamete_lambda=0.0,
):
    if gamete_dose[allele_index] < 1:
        # ensure prob is zero in invalid cases
        return -np.inf
    gamete_dose[allele_index] -= 1
    if gamete_ploidy == 2:
        # the ploidy of the constant portion is 1 so lambda is ignored
        # (i.e., the constant is the first allele of the pair)
        gamete_lambda = 0.0
    lprob = gamete_log_pmf(
        gamete_dose=gamete_dose,
        gamete_ploidy=gamete_ploidy - 1,
        parent_dose=parent_dose,
        parent_ploidy=parent_ploidy,
        gamete_lambda=gamete_lambda,
    )
    gamete_dose[allele_index] += 1
    return lprob


@njit(cache=True)
def gamete_allele_log_pmf(
    gamete_count,
    gamete_ploidy,
    parent_count,
    parent_ploidy,
    gamete_lambda=0.0,
):
    """Log probability of allele within a gamete drawn from a known genotype.

    Parameters
    ----------
    gamete_count : int
        Counts of allele in the gamete.
    gamete_ploidy : int
        Ploidy (tau) of the gamete.
    parent_count : int
        Counts of allele within the parent genotype.
    parent_ploidy : int
        Ploidy of the parent.
    gamete_lambda : float
        Excess IBD probability of gamete.

    Returns
    -------
    Log-transformed probability of allele within gamete being derived from parental genotype.

    Notes
    -----
    This function assumes that other alleles of gamete represent a valid partial
    gamete given the parental genotype.

    A non-zero lambda value is only supported for diploid gametes.
    """
    assert gamete_count <= gamete_ploidy
    assert parent_count <= parent_ploidy
    if gamete_count < 1:
        # ensure prob of zero in invalid cases
        return -np.inf
    if parent_count == 0:
        return -np.inf
    const_count = gamete_count - 1
    const_ploidy = gamete_ploidy - 1
    # probability given no dr
    available_count = parent_count - const_count
    available_total = parent_ploidy - const_ploidy
    prob = (available_count / available_total) * (1 - gamete_lambda)
    # probability given dr only supports diploid gametes
    if gamete_lambda > 0.0:
        if gamete_ploidy != 2:
            raise ValueError("Lambda parameter is only supported for diploid gametes")
        # gamete must have 2+ copies of this allele
        if const_count >= 1:
            # probability of dr resulting in this specific allelic copy
            prob += (const_count / const_ploidy) * gamete_lambda
    if prob == 0.0:
        return -np.inf
    else:
        return np.log(prob)


@njit(cache=True)
def trio_log_pmf(
    progeny,
    parent_p,
    parent_q,
    ploidy_p,
    ploidy_q,
    tau_p,
    tau_q,
    lambda_p,
    lambda_q,
    error_p,
    error_q,
    log_frequencies,
    dosage,
    dosage_p,
    dosage_q,
    gamete_p,
    gamete_q,
    constraint_p,
    constraint_q,
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
    log_frequencies : ndarray, float, shape (n_alleles)
        Log of prior for allele frequencies.

    Returns
    -------
    lprob : float
        Log-probability of the trio.
    """
    # handle case of clone
    error_p = 1.0 if (tau_p == 0) else error_p
    error_q = 1.0 if (tau_q == 0) else error_q

    # handel error terms
    lerror_p = np.log(error_p)
    lerror_q = np.log(error_q)
    lcorrect_p = np.log(1 - error_p) if error_p < 1.0 else -np.inf
    lcorrect_q = np.log(1 - error_q) if error_q < 1.0 else -np.inf

    # ensure frequencies correspond to dosage frequencies
    dosage_log_frequencies = dosage_frequencies(progeny, log_frequencies)

    # ploidy of 0 indicates unknown parent
    set_allelic_dosage(progeny, dosage)
    assert dosage.sum() == tau_p + tau_q  # sanity
    if ploidy_p == 0:
        dosage_p[:] = 0
    else:
        set_parental_copies(parent_p, progeny, dosage_p)
    if ploidy_q == 0:
        dosage_q[:] = 0
    else:
        set_parental_copies(parent_q, progeny, dosage_q)

    for i in range(len(progeny)):
        constraint_p[i] = min(dosage[i], dosage_p[i])
        constraint_q[i] = min(dosage[i], dosage_q[i])

    # handle lambda parameter (diploid gametes only)
    if lambda_p > 0.0:
        if tau_p != 2:
            raise ValueError(
                "Non-zero lambda is only supported for a gametic ploidy (tau) of 2"
            )
        # adjust constraint for double reduction
        for i in range(len(dosage)):
            if (dosage[i] >= 2) and (constraint_p[i] == 1):
                constraint_p[i] = 2
    if lambda_q > 0.0:
        if tau_q != 2:
            raise ValueError(
                "Non-zero lambda is only supported for a gametic ploidy (tau) of 2"
            )
        # adjust constraint for double reduction
        for i in range(len(dosage)):
            if (dosage[i] >= 2) and (constraint_q[i] == 1):
                constraint_q[i] = 2

    # used to prune code paths
    valid_p = constraint_p.sum() >= tau_p
    valid_q = constraint_q.sum() >= tau_q
    valid_p &= tau_p > 0
    valid_q &= tau_q > 0
    valid_p &= error_p < 1.0
    valid_q &= error_q < 1.0

    # accumulate log-probabilities
    lprob = -np.inf

    # assuming both parents are valid
    if valid_p and valid_q:
        set_initial_dosage(tau_p, constraint_p, gamete_p)
        set_inverse_gamete(dosage, gamete_p, gamete_q)
        while True:
            # assuming both parents are valid
            lprob_p = (
                gamete_log_pmf(
                    gamete_dose=gamete_p,
                    gamete_ploidy=tau_p,
                    parent_dose=dosage_p,
                    parent_ploidy=ploidy_p,
                    gamete_lambda=lambda_p,
                )
                + lcorrect_p
            )
            lprob_q = (
                gamete_log_pmf(
                    gamete_dose=gamete_q,
                    gamete_ploidy=tau_q,
                    parent_dose=dosage_q,
                    parent_ploidy=ploidy_q,
                    gamete_lambda=lambda_q,
                )
                + lcorrect_q
            )
            lprob_pq = lprob_p + lprob_q
            lprob = add_log_prob(lprob, lprob_pq)
            # assuming p valid and q invalid (avoids iterating gametes of p twice)
            # probability of gamete_q
            lprob_q = (
                log_unknown_dosage_prior(gamete_q, dosage_log_frequencies) + lerror_q
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
        set_initial_dosage(tau_p, constraint_p, gamete_p)
        set_inverse_gamete(dosage, gamete_p, gamete_q)
        while True:
            lprob_p = (
                gamete_log_pmf(
                    gamete_dose=gamete_p,
                    gamete_ploidy=tau_p,
                    parent_dose=dosage_p,
                    parent_ploidy=ploidy_p,
                    gamete_lambda=lambda_p,
                )
                + lcorrect_p
            )
            # probability of gamete_q
            lprob_q = (
                log_unknown_dosage_prior(gamete_q, dosage_log_frequencies) + lerror_q
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
        set_initial_dosage(tau_q, constraint_q, gamete_q)
        set_inverse_gamete(dosage, gamete_q, gamete_p)
        while True:
            # probability of gamete_p
            lprob_p = (
                log_unknown_dosage_prior(gamete_p, dosage_log_frequencies) + lerror_p
            )
            lprob_q = (
                gamete_log_pmf(
                    gamete_dose=gamete_q,
                    gamete_ploidy=tau_q,
                    parent_dose=dosage_q,
                    parent_ploidy=ploidy_q,
                    gamete_lambda=lambda_q,
                )
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
        log_unknown_dosage_prior(dosage, dosage_log_frequencies) + lerror_p + lerror_q
    )
    lprob = add_log_prob(lprob, lprob_pq)
    return lprob


@njit(cache=True)
def markov_blanket_log_probability(
    target_index,
    sample_genotypes,
    sample_ploidy,
    sample_parents,
    sample_children,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    log_frequencies,
    dosage,
    dosage_p,
    dosage_q,
    gamete_p,
    gamete_q,
    constraint_p,
    constraint_q,
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
    sample_parents : ndarray, int, shape (n_samples, 2)
        Parent indices of each sample with -1 indicating
        unknown parents.
    gamete_tau : int, shape (n_samples, 2)
        Gamete ploidy associated with each pedigree edge.
    gamete_lambda : float, shape (n_samples, 2)
        Excess IBDy associated with each pedigree edge.
    gamete_error : float, shape (n_samples, 2)
        Error rate associated with each pedigree edge.
    log_frequencies : ndarray, float, shape (n_alleles)
        Log of prior for allele frequencies.

    Returns
    -------
    log_probability : float
        Joint log probability of pedigree items that fall within the
        Markov blanket of the specified target sample.

    """
    n_samples, max_children = sample_children.shape
    assert 0 <= target_index < n_samples
    log_joint = 0.0
    for idx in range(-1, max_children):
        if idx < 0:
            # start with trio in which target is the child
            i = target_index
        else:
            # iterate through children
            i = sample_children[target_index, idx]
            if i < 0:
                # no more children
                break
        p = sample_parents[i, 0]
        q = sample_parents[i, 1]
        if p >= 0:
            error_p = gamete_error[i, 0]
            ploidy_p = sample_ploidy[p]
        else:
            error_p = 1.0
            ploidy_p = 0
        if q >= 0:
            error_q = gamete_error[i, 1]
            ploidy_q = sample_ploidy[q]
        else:
            error_q = 1.0
            ploidy_q = 0
        log_joint += trio_log_pmf(
            sample_genotypes[i],
            sample_genotypes[p],
            sample_genotypes[q],
            ploidy_p=ploidy_p,
            ploidy_q=ploidy_q,
            tau_p=gamete_tau[i, 0],
            tau_q=gamete_tau[i, 1],
            lambda_p=gamete_lambda[i, 0],
            lambda_q=gamete_lambda[i, 1],
            error_p=error_p,
            error_q=error_q,
            log_frequencies=log_frequencies,
            dosage=dosage,
            dosage_p=dosage_p,
            dosage_q=dosage_q,
            gamete_p=gamete_p,
            gamete_q=gamete_q,
            constraint_p=constraint_p,
            constraint_q=constraint_q,
        )
    return log_joint


@njit(cache=True)
def trio_allele_log_pmf(
    allele_index,
    progeny,
    parent_p,
    parent_q,
    ploidy_p,
    ploidy_q,
    tau_p,
    tau_q,
    lambda_p,
    lambda_q,
    error_p,
    error_q,
    log_frequencies,
    dosage,
    dosage_p,
    dosage_q,
    gamete_p,
    gamete_q,
    constraint_p,
    constraint_q,
):
    """Log probability of allele within a trio of genotypes.

    Warning
    -------
    This function is proportional, but not equal, to the true PMF.
    It is out by a constant which can be computed with some complexity.
    In short, this function returns P(a | c, p, q, ...) * P(c | p, q, ...)
    where 'c' is the set of progeny alleles held as constant and hence
    P(c | p, q, ...) is a constant.
    This is easier to calculate than the true PMF of P(a | c, p, q, ...)
    due to the implementation details of iterating over gametes.


    Parameters
    ----------
    allele_index : int
        Index of allele within progeny treated as variable.
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
    log_frequencies : ndarray, float, shape (n_alleles)
        Log of prior for allele frequencies.

    Returns
    -------
    lprob : float
        Log-probability.
    """
    # handle case of clone
    error_p = 1.0 if (tau_p == 0) else error_p
    error_q = 1.0 if (tau_q == 0) else error_q

    # handel error terms
    lerror_p = np.log(error_p)
    lerror_q = np.log(error_q)
    lcorrect_p = np.log(1 - error_p) if error_p < 1.0 else -np.inf
    lcorrect_q = np.log(1 - error_q) if error_q < 1.0 else -np.inf

    # ensure allele_index is index of first occurrence of that allele within progeny
    # this is required so that the allele_index can be used on dosage arrays
    assert allele_index < len(progeny)
    for i in range(len(progeny)):
        if progeny[i] == progeny[allele_index]:
            allele_index = i
            break

    # ensure frequencies correspond to dosage frequencies
    dosage_log_frequencies = dosage_frequencies(progeny, log_frequencies)

    # ploidy of 0 indicates unknown parent
    set_allelic_dosage(progeny, dosage)
    assert dosage.sum() == tau_p + tau_q  # sanity
    if ploidy_p == 0:
        dosage_p[:] = 0
    else:
        set_parental_copies(parent_p, progeny, dosage_p)
    if ploidy_q == 0:
        dosage_q[:] = 0
    else:
        set_parental_copies(parent_q, progeny, dosage_q)

    for i in range(len(progeny)):
        constraint_p[i] = min(dosage[i], dosage_p[i])
        constraint_q[i] = min(dosage[i], dosage_q[i])

    # handle lambda parameter (diploid gametes only)
    if lambda_p > 0.0:
        if tau_p != 2:
            raise ValueError(
                "Non-zero lambda is only supported for a gametic ploidy (tau) of 2"
            )
        # adjust constraint for double reduction
        for i in range(len(dosage)):
            if (dosage[i] >= 2) and (constraint_p[i] == 1):
                constraint_p[i] = 2
    if lambda_q > 0.0:
        if tau_q != 2:
            raise ValueError(
                "Non-zero lambda is only supported for a gametic ploidy (tau) of 2"
            )
        # adjust constraint for double reduction
        for i in range(len(dosage)):
            if (dosage[i] >= 2) and (constraint_q[i] == 1):
                constraint_q[i] = 2

    # used to prune code paths
    valid_p = constraint_p.sum() >= tau_p
    valid_q = constraint_q.sum() >= tau_q
    valid_p &= tau_p > 0
    valid_q &= tau_q > 0
    valid_p &= error_p < 1.0
    valid_q &= error_q < 1.0

    # accumulate log-probabilities
    # for Gibbs sampling these are calculated as probability
    # of proposed allele given other alleles held as constant
    # p(a | c).
    # here we actually calculate p(a | c) * p(c | p, q) for simplicity.
    # to do this we iterate over all possible gamete combinations
    # and model the probability of the allele in question coming from
    # each gamete in a pair.
    # therefore, the alleles held as constant usually includes alleles
    # from both gametes.
    # we need to weight for each constant combination by its probability
    # of occurring (p(c | p, q)) and then multiply this by the probability
    # of the allele given the constant (p(a | c)).
    # This effectively involves calculating three probabilities:
    # p(gamete_p | parent_p)
    # p(gamete_q_constant | parent_q, gamete_p)
    # p(gamete_q_allele | parent_q, gamete_q_constant, gamete_p)
    lprob = -np.inf

    # assuming both parents are valid
    if valid_p and valid_q:
        set_initial_dosage(tau_p, constraint_p, gamete_p)
        set_inverse_gamete(dosage, gamete_p, gamete_q)
        while True:
            # probability of each gamete given parent
            lprob_gamete_p = gamete_log_pmf(
                gamete_dose=gamete_p,
                gamete_ploidy=tau_p,
                parent_dose=dosage_p,
                parent_ploidy=ploidy_p,
                gamete_lambda=lambda_p,
            )
            lprob_const_p = gamete_const_log_pmf(
                allele_index=allele_index,
                gamete_dose=gamete_p,
                gamete_ploidy=tau_p,
                parent_dose=dosage_p,
                parent_ploidy=ploidy_p,
                gamete_lambda=lambda_p,
            )
            lprob_allele_p = gamete_allele_log_pmf(
                gamete_count=gamete_p[allele_index],
                gamete_ploidy=tau_p,
                parent_count=dosage_p[allele_index],
                parent_ploidy=ploidy_p,
                gamete_lambda=lambda_p,
            )
            lprob_gamete_q = gamete_log_pmf(
                gamete_dose=gamete_q,
                gamete_ploidy=tau_q,
                parent_dose=dosage_q,
                parent_ploidy=ploidy_q,
                gamete_lambda=lambda_q,
            )
            lprob_const_q = gamete_const_log_pmf(
                allele_index=allele_index,
                gamete_dose=gamete_q,
                gamete_ploidy=tau_q,
                parent_dose=dosage_q,
                parent_ploidy=ploidy_q,
                gamete_lambda=lambda_q,
            )
            lprob_allele_q = gamete_allele_log_pmf(
                gamete_count=gamete_q[allele_index],
                gamete_ploidy=tau_q,
                parent_count=dosage_q[allele_index],
                parent_ploidy=ploidy_q,
                gamete_lambda=lambda_q,
            )
            lprob_p = lprob_gamete_q + lprob_const_p + lprob_allele_p
            lprob_q = lprob_gamete_p + lprob_const_q + lprob_allele_q
            lprob_pq = add_log_prob(lprob_p, lprob_q) + lcorrect_p + lcorrect_q
            lprob = add_log_prob(lprob, lprob_pq)

            # assuming p valid and q invalid (avoids iterating gametes of p twice)
            lprob_gamete_q = log_unknown_dosage_prior(gamete_q, dosage_log_frequencies)
            lprob_const_q = log_unknown_const_prior(
                gamete_q, allele_index, dosage_log_frequencies
            )
            lprob_allele_q = dosage_log_frequencies[allele_index]
            lprob_p = lprob_gamete_q + lprob_const_p + lprob_allele_p
            lprob_q = lprob_gamete_p + lprob_const_q + lprob_allele_q
            lprob_pq = add_log_prob(lprob_p, lprob_q) + lcorrect_p + lerror_q
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
        set_initial_dosage(tau_p, constraint_p, gamete_p)
        set_inverse_gamete(dosage, gamete_p, gamete_q)
        while True:
            # assuming p valid and q invalid
            lprob_gamete_p = gamete_log_pmf(
                gamete_dose=gamete_p,
                gamete_ploidy=tau_p,
                parent_dose=dosage_p,
                parent_ploidy=ploidy_p,
                gamete_lambda=lambda_p,
            )
            lprob_const_p = gamete_const_log_pmf(
                allele_index=allele_index,
                gamete_dose=gamete_p,
                gamete_ploidy=tau_p,
                parent_dose=dosage_p,
                parent_ploidy=ploidy_p,
                gamete_lambda=lambda_p,
            )
            lprob_allele_p = gamete_allele_log_pmf(
                gamete_count=gamete_p[allele_index],
                gamete_ploidy=tau_p,
                parent_count=dosage_p[allele_index],
                parent_ploidy=ploidy_p,
                gamete_lambda=lambda_p,
            )
            lprob_gamete_q = log_unknown_dosage_prior(gamete_q, dosage_log_frequencies)
            lprob_const_q = log_unknown_const_prior(
                gamete_q, allele_index, dosage_log_frequencies
            )
            lprob_allele_q = dosage_log_frequencies[allele_index]
            lprob_p = lprob_gamete_q + lprob_const_p + lprob_allele_p
            lprob_q = lprob_gamete_p + lprob_const_q + lprob_allele_q
            lprob_pq = add_log_prob(lprob_p, lprob_q) + lcorrect_p + lerror_q
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
        set_initial_dosage(tau_q, constraint_q, gamete_q)
        set_inverse_gamete(dosage, gamete_q, gamete_p)
        while True:
            assert gamete_p.sum() == tau_p
            assert gamete_q.sum() == tau_q
            # assuming q valid and p invalid
            lprob_gamete_q = gamete_log_pmf(
                gamete_dose=gamete_q,
                gamete_ploidy=tau_q,
                parent_dose=dosage_q,
                parent_ploidy=ploidy_q,
                gamete_lambda=lambda_q,
            )
            lprob_const_q = gamete_const_log_pmf(
                allele_index=allele_index,
                gamete_dose=gamete_q,
                gamete_ploidy=tau_q,
                parent_dose=dosage_q,
                parent_ploidy=ploidy_q,
                gamete_lambda=lambda_q,
            )
            lprob_allele_q = gamete_allele_log_pmf(
                gamete_count=gamete_q[allele_index],
                gamete_ploidy=tau_q,
                parent_count=dosage_q[allele_index],
                parent_ploidy=ploidy_q,
                gamete_lambda=lambda_q,
            )
            lprob_gamete_p = log_unknown_dosage_prior(gamete_p, dosage_log_frequencies)
            lprob_const_p = log_unknown_const_prior(
                gamete_p, allele_index, dosage_log_frequencies
            )
            lprob_allele_p = dosage_log_frequencies[allele_index]
            lprob_p = lprob_gamete_q + lprob_const_p + lprob_allele_p
            lprob_q = lprob_gamete_p + lprob_const_q + lprob_allele_q
            lprob_pq = add_log_prob(lprob_p, lprob_q) + lerror_p + lcorrect_q
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
    # p(constant) * p(allele | constant) * 2
    lprob_const = log_unknown_const_prior(dosage, allele_index, dosage_log_frequencies)
    lprob_allele = (
        dosage_log_frequencies[allele_index] + 0.6931471805599453
    )  # log frequency of allele * 2
    lprob_pq = lprob_const + lprob_allele + lerror_p + lerror_q
    lprob = add_log_prob(lprob, lprob_pq)
    assert not np.isnan(lprob)
    return lprob


@njit(cache=True)
def markov_blanket_log_allele_probability(
    target_index,
    allele_index,
    sample_genotypes,
    sample_ploidy,
    sample_parents,
    sample_children,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    log_frequencies,
    dosage,
    dosage_p,
    dosage_q,
    gamete_p,
    gamete_q,
    constraint_p,
    constraint_q,
):
    """Joint probability of pedigree items that fall within the
    Markov blanket of the specified target sample.

    Parameters
    ----------
    target_index : int
        Index of target sample.
    allele_index : int
        Index of target allele.
    sample_genotypes : ndarray, int, shape (n_sample, max_ploidy)
        Genotype of each sample padded by negative values.
    sample_ploidy  : ndarray, int, shape (n_sample,)
        Sample ploidy
    sample_parents : ndarray, int, shape (n_samples, 2)
        Parent indices of each sample with -1 indicating
        unknown parents.
    gamete_tau : int, shape (n_samples, 2)
        Gamete ploidy associated with each pedigree edge.
    gamete_lambda : float, shape (n_samples, 2)
        Excess IBDy associated with each pedigree edge.
    gamete_error : float, shape (n_samples, 2)
        Error rate associated with each pedigree edge.
    log_frequencies : ndarray, float, shape (n_alleles)
        Log of prior for allele frequencies.

    Returns
    -------
    log_probability : float
        Joint log probability of pedigree items that fall within the
        Markov blanket of the specified target sample.

    """
    n_samples, max_children = sample_children.shape
    assert 0 <= target_index < n_samples

    # start with trio where target is progeny
    p = sample_parents[target_index, 0]
    q = sample_parents[target_index, 1]
    if p >= 0:
        error_p = gamete_error[target_index, 0]
        ploidy_p = sample_ploidy[p]
    else:
        error_p = 1.0
        ploidy_p = 0
    if q >= 0:
        error_q = gamete_error[target_index, 1]
        ploidy_q = sample_ploidy[q]
    else:
        error_q = 1.0
        ploidy_q = 0
    log_joint = trio_allele_log_pmf(
        allele_index=allele_index,
        progeny=sample_genotypes[target_index],
        parent_p=sample_genotypes[p],
        parent_q=sample_genotypes[q],
        ploidy_p=ploidy_p,
        ploidy_q=ploidy_q,
        tau_p=gamete_tau[target_index, 0],
        tau_q=gamete_tau[target_index, 1],
        lambda_p=gamete_lambda[target_index, 0],
        lambda_q=gamete_lambda[target_index, 1],
        error_p=error_p,
        error_q=error_q,
        log_frequencies=log_frequencies,
        dosage=dosage,
        dosage_p=dosage_p,
        dosage_q=dosage_q,
        gamete_p=gamete_p,
        gamete_q=gamete_q,
        constraint_p=constraint_p,
        constraint_q=constraint_q,
    )
    # loop through children
    for idx in range(max_children):
        i = sample_children[target_index, idx]
        if i < 0:
            # no more children
            break
        p = sample_parents[i, 0]
        q = sample_parents[i, 1]
        if p >= 0:
            error_p = gamete_error[i, 0]
            ploidy_p = sample_ploidy[p]
        else:
            error_p = 1.0
            ploidy_p = 0
        if q >= 0:
            error_q = gamete_error[i, 1]
            ploidy_q = sample_ploidy[q]
        else:
            error_q = 1.0
            ploidy_q = 0
        # the target is a parent
        log_joint += trio_log_pmf(
            sample_genotypes[i],
            sample_genotypes[p],
            sample_genotypes[q],
            ploidy_p=ploidy_p,
            ploidy_q=ploidy_q,
            tau_p=gamete_tau[i, 0],
            tau_q=gamete_tau[i, 1],
            lambda_p=gamete_lambda[i, 0],
            lambda_q=gamete_lambda[i, 1],
            error_p=error_p,
            error_q=error_q,
            log_frequencies=log_frequencies,
            dosage=dosage,
            dosage_p=dosage_p,
            dosage_q=dosage_q,
            gamete_p=gamete_p,
            gamete_q=gamete_q,
            constraint_p=constraint_p,
            constraint_q=constraint_q,
        )

    return log_joint
