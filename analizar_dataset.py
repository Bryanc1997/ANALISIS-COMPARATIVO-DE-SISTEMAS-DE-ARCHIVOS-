#!/usr/bin/env python3
"""Analisis descriptivo y control de calidad del dataset maestro FIO/iostat."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as exc:
    print("Falta una dependencia:", exc, file=sys.stderr)
    print("Instale con: python3 -m pip install pandas numpy matplotlib", file=sys.stderr)
    sys.exit(2)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="dataset_maestro.csv")
    p.add_argument("--output", default="analisis_descriptivo")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def effective_metrics(df):
    read_only = df["read_percentage"].eq(100)
    write_only = df["write_percentage"].eq(100)
    mixed = ~(read_only | write_only)

    df["latencia_media_efectiva_ms"] = np.nan
    df["latencia_p95_efectiva_ms"] = np.nan
    df["latencia_p99_efectiva_ms"] = np.nan

    df.loc[read_only, "latencia_media_efectiva_ms"] = df.loc[read_only, "read_lat_mean_ms"]
    df.loc[read_only, "latencia_p95_efectiva_ms"] = df.loc[read_only, "read_p95_ms"]
    df.loc[read_only, "latencia_p99_efectiva_ms"] = df.loc[read_only, "read_p99_ms"]

    df.loc[write_only, "latencia_media_efectiva_ms"] = df.loc[write_only, "write_lat_mean_ms"]
    df.loc[write_only, "latencia_p95_efectiva_ms"] = df.loc[write_only, "write_p95_ms"]
    df.loc[write_only, "latencia_p99_efectiva_ms"] = df.loc[write_only, "write_p99_ms"]

    rp = df.loc[mixed, "read_percentage"].fillna(0) / 100
    wp = df.loc[mixed, "write_percentage"].fillna(0) / 100
    df.loc[mixed, "latencia_media_efectiva_ms"] = (
        df.loc[mixed, "read_lat_mean_ms"].fillna(0) * rp
        + df.loc[mixed, "write_lat_mean_ms"].fillna(0) * wp
    )
    df.loc[mixed, "latencia_p95_efectiva_ms"] = (
        df.loc[mixed, "read_p95_ms"].fillna(0) * rp
        + df.loc[mixed, "write_p95_ms"].fillna(0) * wp
    )
    df.loc[mixed, "latencia_p99_efectiva_ms"] = (
        df.loc[mixed, "read_p99_ms"].fillna(0) * rp
        + df.loc[mixed, "write_p99_ms"].fillna(0) * wp
    )
    return df


def main():
    a = args()
    source = Path(a.input).resolve()
    out = Path(a.output).resolve()
    charts = out / "graficos"
    out.mkdir(parents=True, exist_ok=True)
    charts.mkdir(exist_ok=True)

    df = pd.read_csv(source)
    key = ["sistema_archivos", "escenario", "repeticion"]
    required = set(key + ["total_iops", "total_bw_mib_s", "read_percentage", "write_percentage"])
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise SystemExit(f"Faltan columnas obligatorias: {missing_required}")

    df = effective_metrics(df)
    df.to_csv(out / "dataset_maestro_con_metricas_efectivas.csv", index=False)

    # Calidad y faltantes
    faltantes = pd.DataFrame({
        "columna": df.columns,
        "faltantes": [int(df[c].isna().sum()) for c in df.columns],
        "porcentaje": [float(df[c].isna().mean() * 100) for c in df.columns],
        "tipo": [str(df[c].dtype) for c in df.columns],
    }).sort_values(["faltantes", "columna"], ascending=[False, True])
    faltantes.to_csv(out / "valores_faltantes.csv", index=False)

    duplicados = df.duplicated(key, keep=False)
    df.loc[duplicados, key].to_csv(out / "claves_duplicadas.csv", index=False)

    numeric = df.select_dtypes(include="number").columns.tolist()
    metricas = [m for m in [
        "total_iops", "total_bw_mib_s", "latencia_media_efectiva_ms",
        "latencia_p95_efectiva_ms", "latencia_p99_efectiva_ms",
        "cpu_user_pct", "cpu_system_pct", "iostat_util_pct_mean",
        "iostat_aqu_sz_mean", "iostat_r_await_ms_mean", "iostat_w_await_ms_mean"
    ] if m in df.columns]

    # Resumen por grupo
    group = df.groupby(["sistema_archivos", "escenario"], dropna=False)
    parts = []
    for metrica in metricas:
        g = group[metrica]
        block = g.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
        block["metrica"] = metrica
        block["cv_pct"] = np.where(block["mean"].abs() > 0, block["std"] / block["mean"].abs() * 100, np.nan)
        block["se"] = block["std"] / np.sqrt(block["count"])
        block["ic95_inf"] = block["mean"] - 1.96 * block["se"]
        block["ic95_sup"] = block["mean"] + 1.96 * block["se"]
        parts.append(block)
    resumen = pd.concat(parts, ignore_index=True)
    resumen.to_csv(out / "estadisticas_por_sistema_escenario.csv", index=False)

    # Atipicos IQR dentro de cada sistema/escenario
    outliers = []
    for (fs, esc), sub in group:
        for metrica in metricas:
            s = sub[metrica].dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            hits = sub[sub[metrica].notna() & ((sub[metrica] < low) | (sub[metrica] > high))]
            for _, r in hits.iterrows():
                outliers.append({
                    "sistema_archivos": fs, "escenario": esc,
                    "repeticion": int(r["repeticion"]), "metrica": metrica,
                    "valor": float(r[metrica]), "limite_inferior": float(low),
                    "limite_superior": float(high)
                })
    pd.DataFrame(outliers, columns=["sistema_archivos","escenario","repeticion","metrica","valor","limite_inferior","limite_superior"]).to_csv(out / "valores_atipicos_iqr.csv", index=False)

    # Rankings: mayor es mejor para IOPS/BW; menor es mejor para latencias
    ranks = []
    for esc, sub in df.groupby("escenario"):
        means = sub.groupby("sistema_archivos")[metricas].mean(numeric_only=True)
        for metrica in metricas:
            ascending = "latencia" in metrica or "await" in metrica or "aqu_sz" in metrica
            ranked = means[metrica].rank(method="min", ascending=ascending)
            for fs in means.index:
                ranks.append({"escenario": esc, "metrica": metrica, "sistema_archivos": fs,
                              "media": means.loc[fs, metrica], "posicion": int(ranked.loc[fs]),
                              "criterio": "menor_mejor" if ascending else "mayor_mejor"})
    ranking = pd.DataFrame(ranks)
    ranking.to_csv(out / "ranking_por_escenario.csv", index=False)

    # Graficos de medias con DE y boxplots.
    core = [m for m in ["total_iops", "total_bw_mib_s", "latencia_media_efectiva_ms", "latencia_p99_efectiva_ms", "iostat_util_pct_mean"] if m in df.columns]
    for metrica in core:
        pivot = df.groupby(["escenario", "sistema_archivos"])[metrica].agg(["mean", "std"]).reset_index()
        scenarios = sorted(df["escenario"].unique())
        systems = [x for x in ["ext4", "xfs", "zfs"] if x in df["sistema_archivos"].unique()]
        x = np.arange(len(scenarios)); width = 0.8 / len(systems)
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, fs in enumerate(systems):
            sub = pivot[pivot["sistema_archivos"] == fs].set_index("escenario").reindex(scenarios)
            ax.bar(x + (i-(len(systems)-1)/2)*width, sub["mean"], width, yerr=sub["std"], capsize=3, label=fs)
        ax.set_title(f"{metrica} por sistema y escenario")
        ax.set_ylabel(metrica); ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=25, ha="right")
        ax.legend(title="Sistema"); fig.tight_layout()
        fig.savefig(charts / f"barras_{metrica}.png", dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        labels, data = [], []
        for esc in scenarios:
            for fs in systems:
                vals = df[(df["escenario"] == esc) & (df["sistema_archivos"] == fs)][metrica].dropna()
                labels.append(f"{esc}\n{fs}"); data.append(vals.to_numpy())
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(f"Distribucion de {metrica}")
        ax.set_ylabel(metrica); ax.tick_params(axis="x", rotation=75); fig.tight_layout()
        fig.savefig(charts / f"boxplot_{metrica}.png", dpi=180); plt.close(fig)

    report = {
        "archivo_entrada": str(source), "sha256_entrada": sha256(source),
        "filas": int(len(df)), "columnas": int(len(df.columns)),
        "combinaciones": int(df.groupby(["sistema_archivos", "escenario"]).ngroups),
        "duplicados_clave": int(duplicados.sum()),
        "metricas_analizadas": metricas,
        "atipicos_iqr": len(outliers),
    }
    (out / "informe_analisis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "informe_analisis.txt").write_text(
        "INFORME DE ANALISIS DESCRIPTIVO\n"
        f"Archivo: {source}\nFilas: {len(df)}\nColumnas: {len(df.columns)}\n"
        f"Combinaciones sistema/escenario: {report['combinaciones']}\n"
        f"Duplicados de clave: {report['duplicados_clave']}\n"
        f"Atipicos IQR detectados: {len(outliers)}\n"
        f"Metricas: {', '.join(metricas)}\n", encoding="utf-8")

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifiesto_sha256.txt")
    with open(out / "manifiesto_sha256.txt", "w", encoding="utf-8") as f:
        for p in files:
            f.write(f"{sha256(p)}  {p.relative_to(out)}\n")

    print(f"Analisis terminado en: {out}")
    print(f"Filas: {len(df)}; combinaciones: {report['combinaciones']}; duplicados: {report['duplicados_clave']}")
    print(f"Atipicos IQR: {len(outliers)}; graficos: {len(list(charts.glob('*.png')))}")

if __name__ == "__main__":
    main()
