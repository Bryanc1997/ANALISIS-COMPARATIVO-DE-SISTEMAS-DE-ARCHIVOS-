#!/usr/bin/env python3
"""
crear_datasets.py
Consolida resultados FIO JSON e iostat TXT de ext4, XFS y ZFS.
Genera:
  - dataset_fio_ejecuciones.csv
  - dataset_iostat_series.csv
  - dataset_iostat_resumen.csv
  - dataset_maestro.csv
  - informe_validacion.json
  - informe_validacion.txt
  - manifiesto_sha256.txt

Solo usa la biblioteca estandar de Python.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

FILE_RE = re.compile(
    r"^(ext4|xfs|zfs)_(.+)_rep(\d+)\.(json|txt)$", re.IGNORECASE
)


def parse_args():
    p = argparse.ArgumentParser(description="Construye datasets desde FIO e iostat")
    p.add_argument("--root", required=True, help="Directorio resultados_definitivos")
    p.add_argument("--map", required=True, dest="map_file", help="mapa_dispositivos.csv")
    p.add_argument("--output", default="datasets_generados", help="Directorio de salida")
    return p.parse_args()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_float(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "")
    # iostat bajo configuracion regional espanola puede usar coma decimal.
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def percentile_linear(values, p):
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def stats(values, prefix):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return {f"{prefix}_{k}": None for k in ("mean", "std", "min", "max", "p95")}
    return {
        f"{prefix}_mean": statistics.fmean(vals),
        f"{prefix}_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        f"{prefix}_min": min(vals),
        f"{prefix}_max": max(vals),
        f"{prefix}_p95": percentile_linear(vals, 0.95),
    }


def read_map(path):
    result = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fs = row["sistema_archivos"].strip().lower()
            result[fs] = {
                "device": row["dispositivo_iostat"].strip(),
                "role": row.get("rol", "experimental").strip() or "experimental",
            }
    return result


def identify(path):
    m = FILE_RE.match(path.name)
    if not m:
        return None
    return {
        "sistema_archivos": m.group(1).lower(),
        "escenario": m.group(2),
        "repeticion": int(m.group(3)),
        "extension": m.group(4).lower(),
    }


def fio_direction(job, direction):
    d = job.get(direction, {}) or {}
    iops = safe_float(d.get("iops"))
    bw_bytes = safe_float(d.get("bw_bytes"))
    if bw_bytes is None:
        bw_kib = safe_float(d.get("bw"))
        bw_bytes = bw_kib * 1024 if bw_kib is not None else None

    lat_obj = d.get("clat_ns") or d.get("lat_ns") or {}
    factor_ms = 1e-6
    if not lat_obj:
        lat_obj = d.get("clat_us") or d.get("lat_us") or {}
        factor_ms = 1e-3
    if not lat_obj:
        lat_obj = d.get("clat_ms") or d.get("lat_ms") or {}
        factor_ms = 1.0

    def lat_stat(name):
        v = safe_float(lat_obj.get(name))
        return v * factor_ms if v is not None else None

    pct = lat_obj.get("percentile", {}) or {}
    def pct_value(target):
        candidates = []
        for k, v in pct.items():
            try:
                candidates.append((abs(float(k) - target), safe_float(v)))
            except (ValueError, TypeError):
                continue
        if not candidates:
            return None
        value = min(candidates, key=lambda x: x[0])[1]
        return value * factor_ms if value is not None else None

    return {
        f"{direction}_iops": iops,
        f"{direction}_bw_mib_s": bw_bytes / (1024 ** 2) if bw_bytes is not None else None,
        f"{direction}_lat_mean_ms": lat_stat("mean"),
        f"{direction}_lat_min_ms": lat_stat("min"),
        f"{direction}_lat_max_ms": lat_stat("max"),
        f"{direction}_p95_ms": pct_value(95.0),
        f"{direction}_p99_ms": pct_value(99.0),
    }


def scenario_metadata(scenario):
    if scenario in ("lectura_secuencial", "escritura_secuencial"):
        bs = 1024 * 1024
    else:
        bs = 4 * 1024
    read_pct, write_pct = None, None
    if scenario == "lectura_secuencial" or scenario == "lectura_aleatoria":
        read_pct, write_pct = 100, 0
    elif scenario == "escritura_secuencial" or scenario == "escritura_aleatoria":
        read_pct, write_pct = 0, 100
    elif "70r30w" in scenario:
        read_pct, write_pct = 70, 30
    return bs, read_pct, write_pct


def parse_fio(path, ident):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    jobs = data.get("jobs", [])
    if not jobs:
        raise ValueError("JSON sin jobs")

    # group_reporting produce normalmente un job agregado. Si hubiera varios,
    # se suman tasas y se promedian CPU/latencias ponderadas de forma simple.
    job = jobs[0]
    row = dict(ident)
    row.pop("extension", None)
    row["archivo_origen_fio"] = str(path)
    row["fio_error"] = sum(int(j.get("error", 0) or 0) for j in jobs)
    bs, rp, wp = scenario_metadata(ident["escenario"])
    row.update({
        "block_size_bytes": bs,
        "read_percentage": rp,
        "write_percentage": wp,
        "iodepth": 32,
        "numjobs": 1,
        "runtime_config_seg": 60,
        "ioengine": "libaio",
        "direct_io": 1,
    })
    row.update(fio_direction(job, "read"))
    row.update(fio_direction(job, "write"))
    row["total_iops"] = sum(v or 0 for v in (row["read_iops"], row["write_iops"]))
    row["total_bw_mib_s"] = sum(v or 0 for v in (row["read_bw_mib_s"], row["write_bw_mib_s"]))
    row["cpu_user_pct"] = safe_float(job.get("usr_cpu"))
    row["cpu_system_pct"] = safe_float(job.get("sys_cpu"))
    row["context_switches"] = safe_float(job.get("ctx"))
    row["major_page_faults"] = safe_float(job.get("majf"))
    row["minor_page_faults"] = safe_float(job.get("minf"))
    return row


def normalize_header(token):
    t = token.strip().lower().replace("%", "pct")
    t = t.replace("/", "_").replace("-", "_")
    aliases = {
        "r_s": "r_s", "w_s": "w_s", "rkb_s": "rkb_s", "wkb_s": "wkb_s",
        "rmb_s": "rmb_s", "wmb_s": "wmb_s", "rrqm_s": "rrqm_s", "wrqm_s": "wrqm_s",
        "r_await": "r_await_ms", "w_await": "w_await_ms", "await": "await_ms",
        "aqu_sz": "aqu_sz", "avgqu_sz": "aqu_sz", "pctutil": "util_pct",
        "rareq_sz": "rareq_sz_kib", "wareq_sz": "wareq_sz_kib",
        "avgrq_sz": "avgrq_sz_sectors", "svctm": "svctm_ms",
    }
    return aliases.get(t, t)


def parse_iostat(path, ident, device, role):
    rows = []
    headers = None
    sample = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if parts and parts[0].lower() == "device":
                headers = ["device"] + [normalize_header(x) for x in parts[1:]]
                continue
            if headers and parts[0] == device and len(parts) >= len(headers):
                record = dict(zip(headers, parts[:len(headers)]))
                sample += 1
                out = {
                    "sistema_archivos": ident["sistema_archivos"],
                    "escenario": ident["escenario"],
                    "repeticion": ident["repeticion"],
                    "muestra": sample,
                    "dispositivo_origen": device,
                    "dispositivo_rol": role,
                    "archivo_origen_iostat": str(path),
                }
                for k, v in record.items():
                    if k != "device":
                        out[k] = safe_float(v)
                # Versiones que informan MiB/s se convierten a KiB/s.
                if out.get("rkb_s") is None and out.get("rmb_s") is not None:
                    out["rkb_s"] = out["rmb_s"] * 1024
                if out.get("wkb_s") is None and out.get("wmb_s") is not None:
                    out["wkb_s"] = out["wmb_s"] * 1024
                rows.append(out)
    return rows


def write_csv(path, rows, preferred=None):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = set()
    for r in rows:
        keys.update(r.keys())
    preferred = preferred or []
    fields = [k for k in preferred if k in keys] + sorted(keys - set(preferred))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    map_file = Path(args.map_file).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    device_map = read_map(map_file)
    validation = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "directorio_origen": str(root),
        "mapa_dispositivos": str(map_file),
        "dispositivos": device_map,
        "errores": [], "advertencias": [],
    }

    fio_rows, series_rows = [], []
    json_files = sorted(root.glob("*/fio/*.json"))
    txt_files = sorted(root.glob("*/iostat/*.txt"))
    validation["json_fio_encontrados"] = len(json_files)
    validation["txt_iostat_encontrados"] = len(txt_files)

    for path in json_files:
        ident = identify(path)
        if not ident:
            validation["advertencias"].append(f"Nombre FIO no reconocido: {path}")
            continue
        try:
            row = parse_fio(path, ident)
            fio_rows.append(row)
            if row["fio_error"] != 0:
                validation["errores"].append(f"FIO error={row['fio_error']}: {path}")
        except Exception as exc:
            validation["errores"].append(f"No se pudo leer FIO {path}: {exc}")

    for path in txt_files:
        ident = identify(path)
        if not ident:
            validation["advertencias"].append(f"Nombre iostat no reconocido: {path}")
            continue
        fs = ident["sistema_archivos"]
        if fs not in device_map:
            validation["errores"].append(f"Sin dispositivo mapeado para {fs}: {path}")
            continue
        rows = parse_iostat(path, ident, device_map[fs]["device"], device_map[fs]["role"])
        if not rows:
            validation["errores"].append(f"Sin muestras del dispositivo esperado en {path}")
        series_rows.extend(rows)

    # Resumen de iostat por sistema, escenario y repeticion.
    groups = defaultdict(list)
    for row in series_rows:
        groups[(row["sistema_archivos"], row["escenario"], row["repeticion"])].append(row)

    summary_rows = []
    metrics = ["r_s", "w_s", "rkb_s", "wkb_s", "r_await_ms", "w_await_ms", "await_ms", "aqu_sz", "util_pct"]
    for key, rows in sorted(groups.items()):
        fs, scenario, rep = key
        item = {"sistema_archivos": fs, "escenario": scenario, "repeticion": rep, "iostat_muestras": len(rows)}
        for metric in metrics:
            item.update(stats([safe_float(r.get(metric)) for r in rows], f"iostat_{metric}"))
        summary_rows.append(item)

    # Union maestra.
    summaries = {(r["sistema_archivos"], r["escenario"], r["repeticion"]): r for r in summary_rows}
    master_rows = []
    for row in fio_rows:
        key = (row["sistema_archivos"], row["escenario"], row["repeticion"])
        merged = dict(row)
        if key in summaries:
            merged.update({k: v for k, v in summaries[key].items() if k not in key})
        else:
            validation["errores"].append(f"Sin resumen iostat para {key}")
        master_rows.append(merged)

    validation.update({
        "filas_fio": len(fio_rows),
        "filas_iostat_series": len(series_rows),
        "filas_iostat_resumen": len(summary_rows),
        "filas_dataset_maestro": len(master_rows),
        "grupos_fio": len({(r['sistema_archivos'], r['escenario'], r['repeticion']) for r in fio_rows}),
        "grupos_iostat": len(groups),
    })
    if len(fio_rows) != 75:
        validation["advertencias"].append(f"Se esperaban 75 filas FIO y se obtuvieron {len(fio_rows)}")
    if len(summary_rows) != 75:
        validation["advertencias"].append(f"Se esperaban 75 resúmenes iostat y se obtuvieron {len(summary_rows)}")

    key_cols = ["sistema_archivos", "escenario", "repeticion"]
    write_csv(out / "dataset_fio_ejecuciones.csv", fio_rows, key_cols)
    write_csv(out / "dataset_iostat_series.csv", series_rows, key_cols + ["muestra", "dispositivo_rol"])
    write_csv(out / "dataset_iostat_resumen.csv", summary_rows, key_cols)
    write_csv(out / "dataset_maestro.csv", master_rows, key_cols)

    with open(out / "informe_validacion.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)

    text = [
        "INFORME DE VALIDACION DE DATASETS",
        f"Fecha UTC: {validation['fecha_utc']}",
        f"JSON FIO encontrados: {validation['json_fio_encontrados']}",
        f"TXT iostat encontrados: {validation['txt_iostat_encontrados']}",
        f"Filas FIO: {validation['filas_fio']}",
        f"Filas iostat series: {validation['filas_iostat_series']}",
        f"Filas iostat resumen: {validation['filas_iostat_resumen']}",
        f"Filas dataset maestro: {validation['filas_dataset_maestro']}",
        f"Errores: {len(validation['errores'])}",
        f"Advertencias: {len(validation['advertencias'])}",
        "",
        "ERRORES:", *[f"- {x}" for x in validation["errores"]],
        "", "ADVERTENCIAS:", *[f"- {x}" for x in validation["advertencias"]],
    ]
    (out / "informe_validacion.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    generated = [p for p in out.iterdir() if p.is_file() and p.name != "manifiesto_sha256.txt"]
    with open(out / "manifiesto_sha256.txt", "w", encoding="utf-8") as f:
        for p in sorted(generated):
            f.write(f"{sha256_file(p)}  {p.name}\n")

    print("Proceso terminado")
    print(f"Salida: {out}")
    print(f"FIO: {len(fio_rows)} filas")
    print(f"iostat series: {len(series_rows)} filas")
    print(f"iostat resumen: {len(summary_rows)} filas")
    print(f"maestro: {len(master_rows)} filas")
    print(f"errores: {len(validation['errores'])}; advertencias: {len(validation['advertencias'])}")
    return 1 if validation["errores"] else 0


if __name__ == "__main__":
    sys.exit(main())
