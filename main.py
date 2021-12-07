import numpy as np
import pandas as pd
from scipy.optimize import linprog


def extract_data():
    """
        Extracts the needed data here, from the database at http://hdr.undp.org/en/composite/HDI
        Removes the first and last lines that are unrelevant for the study as they contain no actual data.
        Also extracts a matrix of data without counrty names and translates a list of preferences into their
        indices counterparts.

        Returns
        -------
        df
        f
        prefs
                  Country  Health  Education_exp  Education_curr       Wealth  Education
0                  Norway   82.40       18.06615       12.897750  66494.25217  15.481950
1                 Ireland   82.31       18.70529       12.666331  68370.58737  15.685810
2             Switzerland   83.78       16.32844       13.380812  69393.52076  14.854626
3  Hong Kong, China (SAR)   84.86       16.92947       12.279960  62984.76553  14.604715
4                 Iceland   82.99       19.08309       12.772787  54682.38057  15.927938

    """
    df = pd.read_excel("2020_Statistical_Annex_Table_1.xlsx",
                       header=None,
                       usecols="B,E,G,I,K",
                       skiprows=list(range(8)) + list(range(200, 271)),
                       names=["Country", "Health", "Education_exp", "Education_curr", "Wealth"])

    df["Education"] = (df["Education_exp"] + df["Education_curr"])/2
    print(df)
    df = df.drop("Education_curr", axis=1).drop("Education_exp", axis=1)
    criteria = ["Health", "Education", "Wealth"]
    for crit in criteria:
        df[crit] = (df[crit] - df[crit].min()) / (df[crit].max() - df[crit].min())
    f = df.iloc[:, 1:].values
    prefs = [("Oman", "Brazil"), ("Ireland", "Portugal"), ("Turkey", "Ukraine"), ("Zimbabwe", "Haiti"), ("Japan", "Estonia"), ("Algeria", "Panama"), ("Kenya", "India"), ("Peru", "Romania")]
    for i, pref in enumerate(prefs):
        prefs[i] = df.index[df["Country"] == pref[0]].tolist()[0], df.index[df["Country"] == pref[1]].tolist()[0]
    return df, f, prefs


def prepare_lin_prog(f, prefs_idx, j, t, n):
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
    b_ub = -np.ones(t) * .1
    A_eq = np.concatenate((
        np.ones((j, 1)), 
        np.zeros((t, 1)))).transpose()
    b_eq = np.ones(1)
    bounds = [(0, None)] * n

    return A_ub, b_ub, A_eq, b_eq, bounds


def main():
    df, f, prefs = extract_data()
    t = len(prefs)
    j = f.shape[1]

    n = t + j
    c = np.ones(n)

    A_ub, b_ub, A_eq, b_eq, bounds = prepare_lin_prog(f, prefs, j, t, n)

    # print(c, c.shape)
    # print(A_ub, A_ub.shape, b_ub, b_ub.shape)
    # print(A_eq, A_eq.shape, b_eq, b_eq.shape)
    # print(bounds)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    print(res)
    w = res.x[:3]
    print(f"{w = }")
    print(np.sum(w), np.sum(res.x[3:]))


if __name__ == "__main__":
    main()
