import facile
import pandas as pd


def main():
    df = pd.read_excel("2020_Statistical_Annex_Table_1.xlsx",
                       header=None,
                       usecols="B,E,G,I,K",
                       skiprows=list(range(8)) + list(range(200,271)),
                       names=["Country", "Health", "Education_exp", "Education_curr", "Wealth"])
    print(df)
    print(df.head())

    a = facile.variable(0, 330)
    b = facile.variable(0, 160)
    c = facile.variable(0, 140)
    d = facile.variable(0, 140)

    facile.constraint(a + b + c + d == 711)
    facile.constraint(a * b * c * d == 711000000)

    p = facile.array([a, b, c, d])
    assert facile.solve(p)
    print("Solution found a=%d, b=%d, c=%d, d=%d" % tuple(p.value()))


if __name__ == "__main__":
    main()
