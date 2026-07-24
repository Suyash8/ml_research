"""
===============================================================================
EXPLAINABILITY MODULE: PCA Back-Projection & Local Patient Waterfall Drivers
===============================================================================
Decomposes latent PCA risk components back into raw gene space via linear
matrix transformation (W_gene = V @ beta_pca) and computes patient-specific
additive risk contributions.
"""

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from src.utils.io import safe_float


class ExplainabilityModule:
    """Explainability module for Cox Elastic-Net model with PCA genomic features."""

    def __init__(
        self,
        clinical_feature_names: Sequence[str],
        expr_cols: Sequence[str],
        coefs: np.ndarray,
        pca_loadings: np.ndarray = None,
    ):
        self.clinical_feature_names = list(clinical_feature_names)
        self.expr_cols = list(expr_cols)
        self.coefs = np.asarray(coefs, dtype=float)
        self.pca_loadings = np.asarray(pca_loadings, dtype=float) if pca_loadings is not None else None

        n_clin = len(self.clinical_feature_names)
        self.clinical_coefs = self.coefs[:n_clin]
        self.pca_coefs = self.coefs[n_clin:]

    def compute_gene_risk_weights(self) -> pd.DataFrame:
        """
        Unrolls 50 PCA component weights back into 300 raw gene contributions:
        W_gene = V @ beta_pca  where V is (300 x 50) loadings matrix.
        """
        if self.pca_loadings is None or self.pca_coefs.size == 0 or not self.expr_cols:
            return pd.DataFrame()

        # V is (300 x 50), pca_coefs is (50,)
        # Loadings matrix shape: (n_components, n_genes) or (n_genes, n_components)
        if self.pca_loadings.shape[0] == len(self.pca_coefs):
            # Shape is (50, 300) -> transpose to (300, 50)
            V = self.pca_loadings.T
        else:
            V = self.pca_loadings

        gene_weights = V @ self.pca_coefs
        df_genes = pd.DataFrame({
            "gene_name": self.expr_cols,
            "global_gene_weight": [safe_float(w) for w in gene_weights],
            "abs_weight": [safe_float(abs(w)) for w in gene_weights],
            "effect_type": ["risk_increasing" if w > 0 else "protective" if w < 0 else "neutral" for w in gene_weights]
        }).sort_values("abs_weight", ascending=False).reset_index(drop=True)

        return df_genes

    def explain_patient(
        self,
        patient_id: str,
        X_clin_patient: np.ndarray,
        raw_expr_patient: np.ndarray,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Computes patient-specific waterfall risk decomposition:
        eta_i = sum(X_clin * beta_clin) + sum(Z_gene * W_gene)
        """
        clin_contribs = X_clin_patient * self.clinical_coefs
        clin_drivers = []
        for name, val, coef, contrib in zip(self.clinical_feature_names, X_clin_patient, self.clinical_coefs, clin_contribs):
            clin_drivers.append({
                "feature": name,
                "type": "clinical",
                "value": safe_float(val),
                "coefficient": safe_float(coef),
                "contribution": safe_float(contrib),
                "direction": "risk_increasing" if contrib > 0 else "protective" if contrib < 0 else "neutral"
            })

        df_genes = self.compute_gene_risk_weights()
        gene_drivers = []
        if not df_genes.empty:
            gene_weights = df_genes.set_index("gene_name")["global_gene_weight"].to_dict()
            for gene_name, expr_val in zip(self.expr_cols, raw_expr_patient):
                w = gene_weights.get(gene_name, 0.0)
                # Log-transformed & standardized representation impact
                contrib = expr_val * w
                gene_drivers.append({
                    "feature": gene_name,
                    "type": "gene",
                    "value": safe_float(expr_val),
                    "coefficient": safe_float(w),
                    "contribution": safe_float(contrib),
                    "direction": "risk_increasing" if contrib > 0 else "protective" if contrib < 0 else "neutral"
                })

        # Combine & sort
        all_drivers = sorted(clin_drivers + gene_drivers, key=lambda x: abs(x["contribution"]), reverse=True)
        top_risk = [d for d in all_drivers if d["contribution"] > 0][:top_n]
        top_protective = [d for d in all_drivers if d["contribution"] < 0][:top_n]

        return {
            "patient_id": patient_id,
            "clinical_contribution_sum": safe_float(np.sum(clin_contribs)),
            "gene_contribution_sum": safe_float(sum(g["contribution"] for g in gene_drivers)),
            "total_risk_score": safe_float(np.sum(clin_contribs) + sum(g["contribution"] for g in gene_drivers)),
            "top_risk_drivers": top_risk,
            "top_protective_drivers": top_protective,
            "all_drivers_ranked": all_drivers[:top_n*2]
        }
