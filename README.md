# 1. Run the code.
`python main.py -h` to access the small doc of the file.
`python main.py -d <delta>` to change the value of delta.
`python main.py -v` to trigger verbose mode.

1. What value(s) did you use for the constraint RHS δik?

I tried the following values for the deltas, note that the deltas are the same for all errors:
  5, 1, .1, .01, .001, .0001, .00001

2. Did your optimal solution exhibit any inconsistencies with respect to the pairwise com-
parisons provided? If yes, which ones?

For all the tested values of delta, i.e. Algeria is preferred to Panama,
Kenya is preferred to India and Peru is preferred to Romania, the last 3 preferences
lead to inconsistent results.
All others were consistent.

3. What were the optimal criteria weight values wj that you obtained?

Results are given in the following table:
|  delta |            w_0 |            w_1 |            w_2 |       sum of errors |
|--------|----------------|----------------|----------------|---------------------|
| 5      | 3.98534414e-07 | 9.99999097e-01 | 5.10685810e-07 | 39.766861578711726  |
| 1      | 9.64382720e-11 | 1.00000000e+00 | 6.33378767e-11 | 7.766861646909813   |
| .1     | 0.57756273     | 0.10140772     | 0.32102955     | 0.5751720378561045  |
| .01    | 3.15539045e-01 | 1.24566095e-10 | 6.84460955e-01 | 0.1202404598506865  |
| .001   | 2.78862922e-01 | 2.00731866e-10 | 7.21137078e-01 | 0.09145850295469214 |
| .0001  | 2.75195310e-01 | 2.04700592e-10 | 7.24804690e-01 | 0.0885803072599351  |
| .00001 | 2.74828549e-01 | 2.04883459e-10 | 7.25171451e-01 | 0.08829248769018429 |

4. What is Canada’s ranking out of all countries, applying your optimal weights? How does
this compare with its HDI ranking?

With the above values of delta, Canada has been respectively ranked: 161, 161, 174, 164, 170, 170 and 170.
The rank of Canada using only the HDI index is 20.
