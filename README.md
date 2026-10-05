# Band Selection Diagnostic: Re-Selection Differential (Δ)

Experimental code for the **internal-validity** diagnosis of band selection
(BS) algorithms. The core quantity is the re-selection differential

Δ = Q(S1) − Q(S2)

- **S1**: the k bands the algorithm selects on the full band set, following its
  own criterion;
- **S2**: the k bands the algorithm selects with the **same criterion** on the
  remaining bands (single-band independent criteria are sliced directly;
  multi-band / search criteria are re-run on the remaining set);
- **Q(·)**: classification accuracy under a fixed evaluation protocol (kNN).

Statistical inference: SE(Δ) (two conventions: independent binomial
approximation and paired McNemar), t = Δ/SE(Δ), and a three-way verdict
(t > +2 positive evidence / |t| ≤ 2 noise / t < −2 negative evidence). The
random-selection accuracy distribution is shown for reference only and never
enters the verdict.

## Repository layout

```
.
├── config.yaml             # all run settings (datasets / methods / classifier / k)
├── run_all.py              # main entry point
├── run_criterion_values.py # criterion-level check: J(S1) vs J(S2)
├── plot_results.py         # figures (reads results/csv)
├── requirements.txt
├── data/<dataset>/         # X.npy (all pixels, N x L) / y.npy (N, 0 = background)
├── cache/
│   ├── stats/              # shared statistics (covariance Sigma / correlation corr)
│   ├── rankings/           # cached full band rankings (independent criteria)
│   ├── selections/         # cached S1 / S2 selections
│   ├── criterion_values/   # cached criterion values ({ds}_{m}_k{k}.json: J1/J2/dJ/rel)
│   ├── accuracy/           # band subset -> accuracy table (shared across methods)
│   └── random/             # B random subsets per k
├── results/
│   ├── csv/                # <dataset>_k<k>.csv + all_datasets.csv + criterion values
│   └── logs/               # run logs
├── figures/                # figure output
└── src/
    ├── paths.py            # path conventions
    ├── config.py           # config loading
    ├── datasets.py         # data loading / stratified split
    ├── cache.py            # generic cache helpers
    ├── stats.py            # shared statistics (with pixel subsampling guard)
    ├── classifier.py       # kNN + accuracy cache table
    ├── metrics.py          # SE(Δ) / t / three-way verdict
    ├── objectives.py       # criterion values J(S)
    ├── pipeline.py         # S1/S2 construction and differential pipeline
    ├── data_prepare.py     # raw .mat -> X.npy / y.npy
    └── algorithms/         # one module per method
```

## Quick start

```bash
# 1) install dependencies
pip install -r requirements.txt

# 2) prepare data (place the raw .mat files in the repo root first)
python3 src/data_prepare.py --dataset indian_pines
python3 src/data_prepare.py --dataset paviau
python3 src/data_prepare.py --dataset salinas

# 3) run everything (all datasets x all methods x all k)
python3 run_all.py

# 4) criterion values and figures
python3 run_criterion_values.py
python3 plot_results.py
```

Common options:

```bash
python3 run_all.py --datasets IP,PU   # only selected datasets
python3 run_all.py --force            # overwrite selection/ranking/statistics caches
python3 run_all.py --jobs 4           # number of parallel workers
python3 plot_results.py --only-criterion   # only the criterion figures
```

## Methods and the independent / dependent distinction

| Method | Supervised | Family | independent | How S2 is built |
|---|---|---|---|---|
| MI (mutual information) | yes | ranking | yes | slice from the full ranking |
| ETP (histogram entropy) | no | ranking | yes | slice from the full ranking |
| MVPCA (PCA loadings) | no | ranking | no | re-run on the remaining set |
| MCD (max determinant) | no | multi-band | no | re-run on the remaining set |
| DPC (density peaks) | no | clustering | no | re-run on the remaining set |
| HOGSP (high-order graph) | no | embedded | no | re-run on the remaining set |
| CAE (concrete autoencoder) | no | deep learning | no | re-run on the remaining set |
| mRMR | yes | search | no | re-run on the remaining set |
| LDA (Fisher-LDA SFS) | yes | search | no | re-run on the remaining set |
| LASSO | yes | sparsity | no | re-run on the remaining set |
| LR (sparse logistic) | yes | sparsity | no | re-run on the remaining set |
| NHMC | yes | ranking | yes | slice from the full ranking |

**independent=True** (single-band criteria): each band is scored on its own
and the score does not depend on the selected set, so the ranking of the
remaining set is just the full ranking without its first k entries. S2 is then
sliced from the cached ranking, with no re-run.

**independent=False** (multi-band / search criteria): scores depend on the band
set, so S2 must be refitted on the remaining set with the same criterion (the
mRMR redundancy term, the MVPCA covariance and the Fisher-LDA scatter matrices
all change with the set).

## Caching (four layers)

1. **Shared statistics** `cache/stats/<ds>_stats.npz`: covariance Sigma and
   correlation corr over all pixels, shared by MCD and DPC and computed once
   per dataset.
2. **Full rankings** `cache/rankings/<ds>_<method>.npy`: single-band
   independent criteria compute the full ranking once; any k is a prefix slice.
3. **Selections** `cache/selections/<ds>_<method>_k<k>_S1|S2.npy`: S1 and S2
   are cached separately and reused unless `--force` is given.
4. **Accuracy table** `cache/accuracy/<ds>_acc.pkl`: band subset (frozenset) ->
   accuracy. Every subset (method selections, random baseline subsets) is looked
   up first, computed on a miss, then written back, shared across methods, k
   values, and runs.

The log prints `CACHE-HIT` / `CACHE-WRITE` for every cache event.

## Output columns (results/csv/<ds>_k<k>.csv)

| Column | Meaning |
|---|---|
| Method | method name |
| Q(S1) | accuracy of the first-choice subset (the basis of traditional accuracy ranking) |
| Q(S2) | accuracy of the re-selected subset |
| ΔQ | Q(S1) − Q(S2) |
| SE_ind / t_ind | independent-approximation standard error and test statistic (conservative, primary verdict) |
| SE_mc / t_mc | paired McNemar convention (NaN without per-sample predictions, falls back to the independent convention) |
| Verdict | positive / noise / negative (threshold ±2) |
| Q1_rand_med / Q1_rand_p975 | median and 97.5th percentile of the random baseline (reference only) |

Criterion values are written to `results/csv/<tag>_criterion_values.csv`
(columns: Dataset, Method, k, J1, J2, ΔJ, relΔJ(%)) and cached as JSON under
`cache/criterion_values/`. Both S1 and S2 are always scored with full-set
statistics, so J(S1) − J(S2) is comparable.

## Configuration (config.yaml)

- `project`: random seed, worker count, parallel backend, cache override flags;
- `classifier`: kNN parameters (k, metric, test fraction, split seed);
- `datasets`: datasets to run (directory name plus short tag);
- `methods`: methods to run (names must match the registry);
- `ks`: band counts to evaluate (default 5, 10, 15, 20, 25, 30);
- `random_baseline`: number of random subsets B;
- `stats.max_pixels`: optional pixel subsampling cap for pixel-level statistics.

## Adding a method

1. Create a module under `src/algorithms/` subclassing `Criterion`:
   - set `name / supervised / sample / independent`;
   - implement `_fit(X, y)` returning the full ranking (best to worst, length
     = L);
   - if it reuses shared statistics, accept sigma / corr as constructor
     arguments like `MaxDet` does.
2. Register it in `METHODS` in `src/algorithms/__init__.py`.
3. Add an entry with the same name to `methods` in `config.yaml`.

## Interpreting the results (exploratory evidence)

- Δ significantly positive (t > +2): the first-choice subset is significantly
  better than the re-selected one, so the ordering induced by the criterion is
  consistent with the discriminative gradient; internal validity is supported.
- |t| ≤ 2 (noise): Δ is indistinguishable from zero. A slightly negative value
  inside the noise band is not evidence against the method (search local
  optima and spectral redundancy can push an effective method's Δ towards
  zero), so it should be read together with the absolute level of Q(S1).
- Δ significantly negative (t < −2): the selection function does better on the
  remaining set than on the first choice, a hard signal that the selection
  mechanism is at odds with discriminative ability.
- Random baseline is reference only: if most methods fail to exceed the random
  p97.5 on Q(S1), the traditional accuracy ranking lacks an uninformative
  baseline anchor.
- Criterion-level ΔJ: for pure ranking methods ΔJ ≥ 0 holds by construction, so
  only the relative magnitude ΔJ/J1 carries information; for search and greedy
  methods (LDA-SFS, MCD, mRMR) the sign itself is diagnostic, and a negative
  value means the first pass was not the criterion optimum.

## License

Released for academic use. Cite the related publication when reusing this
code.
