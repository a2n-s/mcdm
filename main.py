import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def extract_data(filename: str, preferences: [(str, str), ...]) -> (
    pd.core.frame.DataFrame,
    np.ndarray,
    [(int, int), ...]):
    """
        Extracts the needed data here, from the database at http://hdr.undp.org/en/composite/HDI
        Removes the first and last lines that are unrelevant for the study as they contain no actual data.
        Also extracts a matrix of data without counrty names and translates a list of preferences into their
        indices counterparts.

        Args
        ----
        filename: str
            the name of the file where the data is located.
        preferences: [(str, str), ...]
            the preferences of the decision makers.
            for instance, the ("Oman", "Brazil") pair means that Oman is preferred to Brazil.

        Returns
        -------
            df: pandas.core.frame.DataFrame
                the whole dataframe of HDI quantities. Education have been computed as the average of
                (a) mean years of schooling for adults aged 25 years and over, and
                (b) expected years for schooling for children of school entering age
                Contains 192 rows, 4 columns and has the following head:
                                          Country  Health       Wealth  Education
                        0                  Norway   82.40  66494.25217  15.481950
                        1                 Ireland   82.31  68370.58737  15.685810
                        2             Switzerland   83.78  69393.52076  14.854626
                        3  Hong Kong, China (SAR)   84.86  62984.76553  14.604715
                        4                 Iceland   82.99  54682.38057  15.927938
            f: np.ndarray
                the sub matrix with all the fij coefficients in the direct comparisons of a subset
                of alternatives method, i.e. only the Health, Wealth and Education columns in a matrix.
            prefs: [(int, int), ...])
                the list of indices for all the preferences.

    """
    # extract the data and remove useless lines.
    df = pd.read_excel(filename,
                       header=None,
                       usecols="B,E,G,I,K",
                       skiprows=list(range(8)) + list(range(200, 271)),
                       names=["Country", "Health", "Education_exp", "Education_curr", "Wealth"])

    # replace the education* columns by their mean.
    df["Education"] = (df["Education_exp"] + df["Education_curr"])/2
    df = df.drop("Education_curr", axis=1).drop("Education_exp", axis=1)

    # normalize the three criteria between 0 and 1.
    criteria = ["Health", "Education", "Wealth"]
    for crit in criteria:
        df[crit] = (df[crit] - df[crit].min()) / (df[crit].max() - df[crit].min())

    # extract the matrix of raw data, called f in the class.
    f = df.iloc[:, 1:].values

    # interprets the preferences.
    for i, pref in enumerate(preferences):
        # extract the corresponding indices.
        preferences[i] = (df.index[df["Country"] == pref[0]].tolist()[0],
                          df.index[df["Country"] == pref[1]].tolist()[0])
    return df, f, preferences


def prepare_lin_prog(f: np.ndarray, prefs_idx: [(str, str), ...], j: int, t: int, n: int,
                     delta: float = 0.1, verbose: bool = False) -> (
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list):
    """
        Args
        ----
        f
        prefs
        j
        t
        n
        verbose

        Returns
        -------
        c, A_ub, b_ub, A_eq, b_eq, bounds
    """
    # n is the total number of variables.
    # minimize their sum with scipy.
    # first j components are the weights corresponding to the j variables.
    # last t components are the errors associated with the preferences.
    c = np.ones(n)

    # scipy receives A_ub and b_ub such that, A_ub @ x <= b_ub
    # we need to construct A_ub and b_ub s.t.
    #     (f_i0 - f_k0)*w_0 + (f_i1 - f_k1)*w_1 + ... + (f_ij - f_kj)*w_j + e_ik > delta_ik for all i,j s.t. i is prefered to k
    # which coorespond to the line
    #     f[prefs_idx[0][0], 0]-f[prefs_idx[0][1], 0], f[prefs_idx[0][0], 1]-f[prefs_idx[0][1], 1], f[prefs_idx[0][0], 2]-f[prefs_idx[0][1], 2], 1, 0, 0, 0, 0, 0, 0, 0
    # because prefs_idx[...][0] is i and prefs_idx[...][1] is k.
    # we also need to invert the matrices because the inequlity is inverted in scipy.
    A_ub = -np.array([
        [f[prefs_idx[0][0], 0]-f[prefs_idx[0][1], 0], f[prefs_idx[0][0], 1]-f[prefs_idx[0][1], 1], f[prefs_idx[0][0], 2]-f[prefs_idx[0][1], 2], 1, 0, 0, 0, 0, 0, 0, 0],
        [f[prefs_idx[1][0], 0]-f[prefs_idx[1][1], 0], f[prefs_idx[1][0], 1]-f[prefs_idx[1][1], 1], f[prefs_idx[1][0], 2]-f[prefs_idx[1][1], 2], 0, 1, 0, 0, 0, 0, 0, 0],
        [f[prefs_idx[2][0], 0]-f[prefs_idx[2][1], 0], f[prefs_idx[2][0], 1]-f[prefs_idx[2][1], 1], f[prefs_idx[2][0], 2]-f[prefs_idx[2][1], 2], 0, 0, 1, 0, 0, 0, 0, 0],
        [f[prefs_idx[3][0], 0]-f[prefs_idx[3][1], 0], f[prefs_idx[3][0], 1]-f[prefs_idx[3][1], 1], f[prefs_idx[3][0], 2]-f[prefs_idx[3][1], 2], 0, 0, 0, 1, 0, 0, 0, 0],
        [f[prefs_idx[4][0], 0]-f[prefs_idx[4][1], 0], f[prefs_idx[4][0], 1]-f[prefs_idx[4][1], 1], f[prefs_idx[4][0], 2]-f[prefs_idx[4][1], 2], 0, 0, 0, 0, 1, 0, 0, 0],
        [f[prefs_idx[5][0], 0]-f[prefs_idx[5][1], 0], f[prefs_idx[5][0], 1]-f[prefs_idx[5][1], 1], f[prefs_idx[5][0], 2]-f[prefs_idx[5][1], 2], 0, 0, 0, 0, 0, 1, 0, 0],
        [f[prefs_idx[6][0], 0]-f[prefs_idx[6][1], 0], f[prefs_idx[6][0], 1]-f[prefs_idx[6][1], 1], f[prefs_idx[6][0], 2]-f[prefs_idx[6][1], 2], 0, 0, 0, 0, 0, 0, 1, 0],
        [f[prefs_idx[7][0], 0]-f[prefs_idx[7][1], 0], f[prefs_idx[7][0], 1]-f[prefs_idx[7][1], 1], f[prefs_idx[7][0], 2]-f[prefs_idx[7][1], 2], 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    b_ub = -np.ones(t) * delta

    # scipy receives A_eq and b_eq such that, A_eq @ x == b_eq
    # we want that the weights sum to 1.
    A_eq = np.concatenate((
        np.ones((j, 1)), 
        np.zeros((t, 1)))).transpose()
    b_eq = np.ones(1)

    # all errors e_ik and weights w_j are positive.
    bounds = [(0, None)] * n

    if verbose:
        print(c, c.shape)
        print(A_ub, A_ub.shape, b_ub, b_ub.shape)
        print(A_eq, A_eq.shape, b_eq, b_eq.shape)
        print(bounds)

    return c, A_ub, b_ub, A_eq, b_eq, bounds


def main(filename, delta, verbose=False):
    # extract all the data.
    preferences = [("Oman", "Brazil"), ("Ireland", "Portugal"),
                   ("Turkey", "Ukraine"), ("Zimbabwe", "Haiti"),
                   ("Japan", "Estonia"), ("Algeria", "Panama"),
                   ("Kenya", "India"), ("Peru", "Romania")]
    df, f, prefs = extract_data(filename, preferences)

    t = len(prefs)
    j = f.shape[1]
    n = t + j

    # get everything for solving time.
    c, A_ub, b_ub, A_eq, b_eq, bounds = prepare_lin_prog(f, prefs, j, t, n, delta=delta, verbose=verbose)

    # solve the problem.
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    if verbose:
        print(res)
    w = res.x[:3]
    print(f"weights found: {w}")
    if verbose:
        print(f"sum of weights: {np.sum(w)}, sum of errors: {np.sum(res.x[3:])}")
    print()

    # scores and ranking according to the direct comparisons method.
    scores = np.dot(f, w)
    ranks = np.argsort(scores)
    ranking = [df["Country"][r] for r in ranks]
    if verbose:
        print("scores:", scores)
        print("ranks:", ranks)
        print("ranking:", ranking)
        print()

        for i, k in preferences:
            print(f"{scores[i]: 5.4f}, {scores[k]: 5.4f}", "inconsistent" if scores[i] <= scores[k] else "consistent")
        print()

    # questions 4.
    canada = df[df["Country"] == "Canada"].index
    print("canada:", ranks[canada][0])

    df["HDI"] = (df["Health"] * df["Education"] * df["Wealth"]) ** (1/3)
    sorted_df = df.sort_values("HDI")
    hdi_canada = df["HDI"][df["Country"] == "Canada"].values[0]
    hdis = sorted_df["HDI"].values[:-3][::-1]
    if verbose:
        print()
        print("hdis:", hdis)
        print("hdi of canada:", hdi_canada)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="2020_Statistical_Annex_Table_1.xlsx",
                        help="The path to the excel file (defaults to '2020_Statistical_Annex_Table_1.xlsx').")
    parser.add_argument("-d", "--delta", type=float, default=0.1,
                        help="The value of all the right hand side of the equations for the direct method (defaults to 0.1).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Triggers the full verbose if raised.")
    args = parser.parse_args()
    main(args.input, args.delta, args.verbose)
