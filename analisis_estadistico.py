#!/usr/bin/env python3
"""Analisis estadistico comparativo de ext4, XFS y ZFS.

Genera pruebas de Shapiro-Wilk, Levene, Kruskal-Wallis, comparaciones
Mann-Whitney con correccion Holm, tamanos del efecto e informe textual.
"""
import argparse
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError as exc:
    print(f"Falta una dependencia: {exc}", file=sys.stderr)
    print("Instale con: python3 -m pip install numpy pandas scipy", file=sys.stderr)
    sys.exit(2)

SYSTEMS = ["ext4", "xfs", "zfs"]
METRICS = [
    "total_iops",
    "total_bw_mib_s",
    "latencia_media_efectiva_ms",
    "latencia_p95_efectiva_ms",
    "latencia_p99_efectiva_ms",
]
ALPHA = 0.05


def parse_args():
    p = argparse.ArgumentParser(description="Analisis estadistico de sistemas de archivos")
    p.add_argument("--input", required=True, help="CSV con metricas efectivas")
    p.add_argument("--output", default="analisis_estadistico_corregido")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def holm_adjust(p_values):
    """Correccion Holm step-down, conservando el orden original."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * float(p_values[idx]))
        running = max(running, value)
        adjusted_sorted[rank] = running
    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted.tolist()


def rank_biserial_from_u(u, n1, n2):
    # Positivo significa que el grupo 1 tiende a valores mayores.
    return (2.0 * u) / (n1 * n2) - 1.0


def effect_label(value):
    a = abs(value)
    if a < 0.10: return "despreciable"
    if a < 0.30: return "pequeno"
    if a < 0.50: return "moderado"
    return "grande"


def epsilon_squared(h, k, n):
    if n <= k: return np.nan
    return max(0.0, (h - k + 1) / (n - k))


def epsilon_label(value):
    if pd.isna(value): return "no_calculable"
    if value < 0.01: return "despreciable"
    if value < 0.06: return "pequeno"
    if value < 0.14: return "moderado"
    return "grande"


def fmt(v):
    if pd.isna(v): return "NA"
    if abs(v) < 0.001 and v != 0: return f"{v:.3e}"
    return f"{v:.6f}"


def main():
    args = parse_args()
    alpha = args.alpha
    source = Path(args.input).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source)
    required = {"sistema_archivos", "escenario", "repeticion", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Faltan columnas obligatorias: {missing}")

    df = df[df["sistema_archivos"].isin(SYSTEMS)].copy()
    scenarios = sorted(df["escenario"].dropna().unique())
    errors, warnings = [], []

    if len(df) != 75:
        warnings.append(f"Se esperaban 75 filas y se encontraron {len(df)}")

    counts = df.groupby(["sistema_archivos", "escenario"]).size()
    bad_counts = counts[counts != 5]
    for key, value in bad_counts.items():
        warnings.append(f"{key} contiene {value} observaciones; se esperaban 5")

    # Descriptivos usados por el informe estadistico.
    descriptive_rows = []
    normality_rows = []
    levene_rows = []
    kruskal_rows = []
    posthoc_rows = []

    for scenario in scenarios:
        sub = df[df["escenario"] == scenario]
        for metric in METRICS:
            groups = {}
            for fs in SYSTEMS:
                vals = pd.to_numeric(
                    sub.loc[sub["sistema_archivos"] == fs, metric], errors="coerce"
                ).dropna().to_numpy(dtype=float)
                groups[fs] = vals

                if len(vals):
                    descriptive_rows.append({
                        "escenario": scenario,
                        "metrica": metric,
                        "sistema_archivos": fs,
                        "n": len(vals),
                        "media": np.mean(vals),
                        "mediana": np.median(vals),
                        "desviacion_estandar": np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                        "minimo": np.min(vals),
                        "maximo": np.max(vals),
                        "q1": np.quantile(vals, 0.25),
                        "q3": np.quantile(vals, 0.75),
                    })

                if len(vals) >= 3:
                    try:
                        stat, p = stats.shapiro(vals)
                    except Exception as exc:
                        stat, p = np.nan, np.nan
                        warnings.append(f"Shapiro fallo {scenario}/{metric}/{fs}: {exc}")
                else:
                    stat, p = np.nan, np.nan
                normality_rows.append({
                    "escenario": scenario, "metrica": metric,
                    "sistema_archivos": fs, "n": len(vals),
                    "shapiro_w": stat, "p_value": p,
                    "normal_alpha_0_05": bool(p >= alpha) if not pd.isna(p) else None,
                })

            valid = [groups[fs] for fs in SYSTEMS if len(groups[fs]) > 0]
            if len(valid) != 3:
                errors.append(f"Grupos incompletos: {scenario}/{metric}")
                continue

            try:
                lev_stat, lev_p = stats.levene(*valid, center="median")
            except Exception as exc:
                lev_stat, lev_p = np.nan, np.nan
                warnings.append(f"Levene fallo {scenario}/{metric}: {exc}")
            levene_rows.append({
                "escenario": scenario, "metrica": metric,
                "levene_stat": lev_stat, "p_value": lev_p,
                "varianzas_homogeneas_alpha_0_05": bool(lev_p >= alpha) if not pd.isna(lev_p) else None,
            })

            try:
                h, p_kw = stats.kruskal(*valid, nan_policy="omit")
                eps = epsilon_squared(h, 3, sum(len(x) for x in valid))
            except Exception as exc:
                h, p_kw, eps = np.nan, np.nan, np.nan
                errors.append(f"Kruskal fallo {scenario}/{metric}: {exc}")
            kruskal_rows.append({
                "escenario": scenario, "metrica": metric,
                "h_stat": h, "p_value": p_kw,
                "significativo_alpha_0_05": bool(p_kw < alpha) if not pd.isna(p_kw) else None,
                "epsilon_cuadrado": eps,
                "magnitud_efecto": epsilon_label(eps),
                "n_total": sum(len(x) for x in valid),
            })

            # Se incluyen todas las comparaciones para transparencia, incluso si KW no es significativo.
            raw_pairwise = []
            pair_meta = []
            for a, b in itertools.combinations(SYSTEMS, 2):
                x, y = groups[a], groups[b]
                try:
                    u, p_u = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
                    rbc = rank_biserial_from_u(u, len(x), len(y))
                except Exception as exc:
                    u, p_u, rbc = np.nan, np.nan, np.nan
                    warnings.append(f"Mann-Whitney fallo {scenario}/{metric}/{a}-{b}: {exc}")
                raw_pairwise.append(p_u)
                pair_meta.append((a, b, u, rbc, np.mean(x), np.mean(y), np.median(x), np.median(y)))

            adjusted = holm_adjust([1.0 if pd.isna(p) else p for p in raw_pairwise])
            for raw_p, adj_p, meta in zip(raw_pairwise, adjusted, pair_meta):
                a, b, u, rbc, mean_a, mean_b, med_a, med_b = meta
                posthoc_rows.append({
                    "escenario": scenario, "metrica": metric,
                    "grupo_1": a, "grupo_2": b,
                    "u_stat": u, "p_value_sin_ajuste": raw_p,
                    "p_value_holm": adj_p,
                    "significativo_holm_alpha_0_05": bool(adj_p < alpha),
                    "rank_biserial": rbc,
                    "magnitud_efecto": effect_label(rbc) if not pd.isna(rbc) else "no_calculable",
                    "direccion": f"{a}_mayor" if rbc > 0 else (f"{b}_mayor" if rbc < 0 else "sin_direccion"),
                    "media_grupo_1": mean_a, "media_grupo_2": mean_b,
                    "mediana_grupo_1": med_a, "mediana_grupo_2": med_b,
                    "kruskal_significativo": bool(p_kw < alpha) if not pd.isna(p_kw) else None,
                })

    descriptive = pd.DataFrame(descriptive_rows)
    normality = pd.DataFrame(normality_rows)
    levene = pd.DataFrame(levene_rows)
    kruskal = pd.DataFrame(kruskal_rows)
    posthoc = pd.DataFrame(posthoc_rows)

    descriptive.to_csv(out / "descriptivos_estadisticos.csv", index=False)
    normality.to_csv(out / "normalidad_shapiro.csv", index=False)
    levene.to_csv(out / "homogeneidad_levene.csv", index=False)
    kruskal.to_csv(out / "kruskal_wallis.csv", index=False)
    posthoc.to_csv(out / "comparaciones_posthoc_holm.csv", index=False)

    significant_kw = kruskal[kruskal["significativo_alpha_0_05"] == True].copy()
    significant_pairs = posthoc[
        (posthoc["kruskal_significativo"] == True)
        & (posthoc["significativo_holm_alpha_0_05"] == True)
    ].copy()
    significant_kw.to_csv(out / "resultados_significativos_kruskal.csv", index=False)
    significant_pairs.to_csv(out / "resultados_significativos_posthoc.csv", index=False)

    # Informe interpretable.
    lines = [
        "INFORME DE ANALISIS ESTADISTICO",
        f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Archivo: {source}",
        f"SHA-256 entrada: {sha256(source)}",
        f"Filas: {len(df)}",
        f"Escenarios: {len(scenarios)}",
        f"Metricas: {len(METRICS)}",
        f"Nivel alfa: {alpha}",
        "",
        "METODO:",
        "- Shapiro-Wilk por sistema, escenario y metrica.",
        "- Levene centrado en la mediana por escenario y metrica.",
        "- Kruskal-Wallis como comparacion principal entre ext4, XFS y ZFS.",
        "- Mann-Whitney bilateral por pares, con correccion Holm.",
        "- Epsilon cuadrado para Kruskal-Wallis y correlacion biserial de rangos para pares.",
        "",
        f"Comparaciones Kruskal-Wallis ejecutadas: {len(kruskal)}",
        f"Comparaciones Kruskal-Wallis significativas: {len(significant_kw)}",
        f"Comparaciones post hoc significativas tras Holm: {len(significant_pairs)}",
        f"Errores: {len(errors)}",
        f"Advertencias: {len(warnings)}",
        "",
        "RESULTADOS KRUSKAL-WALLIS SIGNIFICATIVOS:",
    ]
    if significant_kw.empty:
        lines.append("- Ninguno con alpha=0.05.")
    else:
        for _, r in significant_kw.sort_values(["escenario", "metrica"]).iterrows():
            lines.append(
                f"- {r['escenario']} | {r['metrica']} | H={fmt(r['h_stat'])} | "
                f"p={fmt(r['p_value'])} | epsilon2={fmt(r['epsilon_cuadrado'])} "
                f"({r['magnitud_efecto']})"
            )

    lines += ["", "COMPARACIONES POST HOC SIGNIFICATIVAS:"]
    if significant_pairs.empty:
        lines.append("- Ninguna tras correccion Holm.")
    else:
        for _, r in significant_pairs.sort_values(["escenario", "metrica", "p_value_holm"]).iterrows():
            lines.append(
                f"- {r['escenario']} | {r['metrica']} | {r['grupo_1']} vs {r['grupo_2']} | "
                f"p_Holm={fmt(r['p_value_holm'])} | r_rb={fmt(r['rank_biserial'])} "
                f"({r['magnitud_efecto']}, {r['direccion']})"
            )

    lines += ["", "ERRORES:"] + [f"- {x}" for x in errors]
    lines += ["", "ADVERTENCIAS:"] + [f"- {x}" for x in warnings]
    (out / "resumen_estadistico.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "archivo_entrada": str(source), "sha256_entrada": sha256(source),
        "filas": len(df), "alpha": alpha, "escenarios": scenarios,
        "metricas": METRICS, "kruskal_total": len(kruskal),
        "kruskal_significativos": len(significant_kw),
        "posthoc_significativos": len(significant_pairs),
        "errores": errors, "advertencias": warnings,
    }
    (out / "resumen_estadistico.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with open(out / "manifiesto_sha256.txt", "w", encoding="utf-8") as f:
        for path in sorted(p for p in out.iterdir() if p.is_file() and p.name != "manifiesto_sha256.txt"):
            f.write(f"{sha256(path)}  {path.name}\n")

    print(f"Analisis estadistico finalizado: {out}")
    print(f"Kruskal-Wallis: {len(kruskal)} pruebas; significativas={len(significant_kw)}")
    print(f"Post hoc Holm significativas={len(significant_pairs)}")
    print(f"Errores={len(errors)}; advertencias={len(warnings)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
