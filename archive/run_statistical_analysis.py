from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy import stats


BASE = Path('/home/illionar/Projects/ml_research')
IN_PATH = BASE / 'data' / 'preprocessed_cleaned' / 'patient_multiomic_cleaned.parquet'
OUT_DIR = BASE / 'data' / 'analysis_reports'


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    p = pvals.fillna(1.0).astype(float).values
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return pd.Series(out, index=pvals.index)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    tab = pd.crosstab(x, y)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.values.sum()
    r, k = tab.shape
    return math.sqrt((chi2 / n) / max(min(k - 1, r - 1), 1))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PATH)

    # Keep analyzable survival rows
    work = df.copy()
    work['OS_MONTHS'] = pd.to_numeric(work.get('OS_MONTHS'), errors='coerce')
    work['OS_EVENT'] = pd.to_numeric(work.get('OS_EVENT'), errors='coerce')
    work = work.dropna(subset=['OS_MONTHS', 'OS_EVENT'])
    work = work[work['OS_MONTHS'] > 0].copy()
    work['OS_EVENT'] = work['OS_EVENT'].astype(int)

    # 1) Dataset summary
    summary = {
        'rows_total': int(df.shape[0]),
        'cols_total': int(df.shape[1]),
        'rows_analyzed': int(work.shape[0]),
        'events': int(work['OS_EVENT'].sum()),
        'censored': int((1 - work['OS_EVENT']).sum()),
        'event_rate': float(work['OS_EVENT'].mean()),
    }

    # 2) Missingness per column
    miss = pd.DataFrame({
        'column': work.columns,
        'missing_count': [int(work[c].isna().sum()) for c in work.columns],
    })
    miss['missing_frac'] = miss['missing_count'] / len(work)
    miss = miss.sort_values('missing_frac', ascending=False)
    miss.to_csv(OUT_DIR / 'missingness_by_column.csv', index=False)

    # 3) Numeric and categorical overviews
    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in work.columns if c not in num_cols]

    num_overview = work[num_cols].describe().T
    num_overview.to_csv(OUT_DIR / 'numeric_overview.csv')

    cat_rows = []
    for c in cat_cols:
        s = work[c].astype('string')
        vc = s.value_counts(dropna=False)
        cat_rows.append({
            'column': c,
            'n_unique': int(s.nunique(dropna=True)),
            'top_value': str(vc.index[0]) if len(vc) else '',
            'top_count': int(vc.iloc[0]) if len(vc) else 0,
            'top_frac': float(vc.iloc[0] / len(s)) if len(vc) else np.nan,
        })
    pd.DataFrame(cat_rows).sort_values('top_frac', ascending=False).to_csv(
        OUT_DIR / 'categorical_overview.csv', index=False
    )

    # 4) Outcome by cancer type
    if 'CANCER_TYPE' in work.columns:
        cancer_outcome = (
            work.groupby('CANCER_TYPE', observed=False)
            .agg(
                n=('PATIENT_ID', 'count'),
                events=('OS_EVENT', 'sum'),
                event_rate=('OS_EVENT', 'mean'),
                median_os_months=('OS_MONTHS', 'median'),
            )
            .reset_index()
            .sort_values('event_rate', ascending=False)
        )
        cancer_outcome.to_csv(OUT_DIR / 'outcome_by_cancer_type.csv', index=False)

    # 5) Univariate tests vs OS_EVENT
    test_rows = []

    # Numeric: Mann-Whitney U + rank-biserial effect size
    for c in num_cols:
        if c in ['OS_EVENT']:
            continue
        x0 = work.loc[work['OS_EVENT'] == 0, c].dropna()
        x1 = work.loc[work['OS_EVENT'] == 1, c].dropna()
        if len(x0) < 20 or len(x1) < 20:
            continue
        if x0.nunique() < 2 and x1.nunique() < 2:
            continue
        try:
            u, p = stats.mannwhitneyu(x0, x1, alternative='two-sided')
            effect = (2 * u) / (len(x0) * len(x1)) - 1
            test_rows.append({
                'feature': c,
                'type': 'numeric',
                'n0': int(len(x0)),
                'n1': int(len(x1)),
                'statistic': float(u),
                'p_value': float(p),
                'effect_size': float(effect),
                'median_event0': float(x0.median()),
                'median_event1': float(x1.median()),
            })
        except Exception:
            continue

    # Categorical: chi-square + Cramer's V
    for c in cat_cols:
        s = work[c].astype('string')
        # ignore very high cardinality text-like columns
        if s.nunique(dropna=True) > 30:
            continue
        tab = pd.crosstab(s.fillna('NA'), work['OS_EVENT'])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        if (tab.values < 1).all():
            continue
        try:
            chi2, p, _, _ = stats.chi2_contingency(tab)
            v = cramers_v(s.fillna('NA'), work['OS_EVENT'])
            test_rows.append({
                'feature': c,
                'type': 'categorical',
                'n0': int((work['OS_EVENT'] == 0).sum()),
                'n1': int((work['OS_EVENT'] == 1).sum()),
                'statistic': float(chi2),
                'p_value': float(p),
                'effect_size': float(v),
                'median_event0': np.nan,
                'median_event1': np.nan,
            })
        except Exception:
            continue

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests['q_value_bh'] = benjamini_hochberg(tests['p_value'])
        tests['significant_q_lt_0_05'] = tests['q_value_bh'] < 0.05
        tests = tests.sort_values(['q_value_bh', 'p_value'])
    tests.to_csv(OUT_DIR / 'univariate_tests_vs_os_event.csv', index=False)

    # 6) Expression/CNA quick signal scan via Spearman with OS_MONTHS
    high_dim_rows = []
    for prefix in ['EXPR_', 'CNA_']:
        cols = [c for c in work.columns if c.startswith(prefix)]
        for c in cols:
            x = pd.to_numeric(work[c], errors='coerce')
            y = work['OS_MONTHS']
            ok = x.notna() & y.notna()
            if ok.sum() < 100:
                continue
            if x[ok].nunique() < 3:
                continue
            rho, p = stats.spearmanr(x[ok], y[ok])
            high_dim_rows.append({
                'feature': c,
                'prefix': prefix,
                'n': int(ok.sum()),
                'spearman_rho': float(rho),
                'p_value': float(p),
            })

    hd = pd.DataFrame(high_dim_rows)
    if not hd.empty:
        hd['q_value_bh'] = benjamini_hochberg(hd['p_value'])
        hd = hd.sort_values('q_value_bh')
    hd.to_csv(OUT_DIR / 'omics_spearman_vs_os_months.csv', index=False)

    # 7) Human-readable report
    lines = []
    lines.append('# Statistical Analysis Report')
    lines.append('')
    lines.append(f"Input file: {IN_PATH}")
    lines.append(f"Rows analyzed: {summary['rows_analyzed']} | Columns: {summary['cols_total']}")
    lines.append(f"Events: {summary['events']} | Censored: {summary['censored']} | Event rate: {summary['event_rate']:.3f}")
    lines.append('')

    top_missing = miss.head(15)
    lines.append('## Top Missingness Columns')
    for _, r in top_missing.iterrows():
        lines.append(f"- {r['column']}: {r['missing_count']} ({r['missing_frac']:.1%})")
    lines.append('')

    if not tests.empty:
        lines.append('## Top Univariate Associations with OS_EVENT (BH q-value)')
        for _, r in tests.head(20).iterrows():
            lines.append(
                f"- {r['feature']} [{r['type']}]: p={r['p_value']:.3g}, q={r['q_value_bh']:.3g}, effect={r['effect_size']:.3f}"
            )
        lines.append('')

    if not hd.empty:
        lines.append('## Top Omics Correlates with OS_MONTHS (Spearman, BH q-value)')
        for _, r in hd.head(20).iterrows():
            lines.append(
                f"- {r['feature']}: rho={r['spearman_rho']:.3f}, p={r['p_value']:.3g}, q={r['q_value_bh']:.3g}, n={int(r['n'])}"
            )
        lines.append('')

    lines.append('## Output Files')
    lines.append('- missingness_by_column.csv')
    lines.append('- numeric_overview.csv')
    lines.append('- categorical_overview.csv')
    lines.append('- outcome_by_cancer_type.csv (if CANCER_TYPE exists)')
    lines.append('- univariate_tests_vs_os_event.csv')
    lines.append('- omics_spearman_vs_os_months.csv')

    (OUT_DIR / 'statistical_analysis_report.md').write_text('\n'.join(lines), encoding='utf-8')

    print('Statistical analysis complete.')
    print(f'Report directory: {OUT_DIR}')


if __name__ == '__main__':
    main()
