#!/usr/bin/env python3
"""Modelado predictivo reproducible para rendimiento de sistemas de archivos."""
import argparse, hashlib, json, math, sys, warnings
from datetime import datetime, timezone
from pathlib import Path

try:
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.svm import SVR
except ImportError as exc:
    print(f"Dependencia ausente: {exc}", file=sys.stderr)
    print("Instale: python3 -m pip install pandas numpy matplotlib scipy scikit-learn joblib", file=sys.stderr)
    sys.exit(2)

RANDOM_STATE = 42
TARGETS = [
    "total_iops",
    "total_bw_mib_s",
    "latencia_media_efectiva_ms",
    "latencia_p99_efectiva_ms",
]
CAT = ["sistema_archivos", "escenario"]
NUM = ["block_size_log2", "read_percentage", "write_percentage"]


def arguments():
    p = argparse.ArgumentParser(description="Modelos ML con validacion GroupKFold")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="modelado_ml_corregido")
    p.add_argument("--splits", type=int, default=5)
    return p.parse_args()


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocessor():
    return ColumnTransformer([
        ("categoricas", encoder(), CAT),
        ("numericas", Pipeline([("escala", StandardScaler())]), NUM),
    ], remainder="drop", sparse_threshold=0)


def models():
    return {
        "baseline_mediana": DummyRegressor(strategy="median"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=2,
            min_samples_leaf=2, random_state=RANDOM_STATE
        ),
        "svr_rbf": SVR(C=10.0, epsilon=0.05, gamma="scale"),
    }


def pipeline(model):
    return Pipeline([("preprocesamiento", preprocessor()), ("modelo", model)])


def metrics(y, pred):
    return {
        "mae": mean_absolute_error(y,pred),
        "rmse": math.sqrt(mean_squared_error(y,pred)),
        "r2": r2_score(y,pred),
        "mape_pct": float(np.mean(np.abs((y-pred)/np.maximum(np.abs(y),1e-12)))*100),
    }


def main():
    a=arguments(); src=Path(a.input).resolve(); out=Path(a.output).resolve()
    plots=out/"graficos"; saved=out/"modelos_entrenados"
    out.mkdir(parents=True,exist_ok=True); plots.mkdir(exist_ok=True); saved.mkdir(exist_ok=True)
    df=pd.read_csv(src)
    required=set(CAT+["repeticion","block_size_bytes","read_percentage","write_percentage"]+TARGETS)
    missing=sorted(required-set(df.columns))
    if missing: raise SystemExit(f"Faltan columnas: {missing}")
    if len(df)!=75: warnings.warn(f"Se esperaban 75 filas; se encontraron {len(df)}")
    df["block_size_log2"]=np.log2(pd.to_numeric(df["block_size_bytes"],errors="raise"))
    groups=df["sistema_archivos"].astype(str)+"|"+df["escenario"].astype(str)
    if groups.nunique()!=15: warnings.warn(f"Se esperaban 15 grupos; hay {groups.nunique()}")
    if a.splits>groups.nunique(): raise SystemExit("Numero de folds mayor al numero de grupos")
    X=df[CAT+NUM].copy()

    prediction_rows=[]; fold_rows=[]; metric_rows=[]; best_rows=[]
    all_models=models(); gkf=GroupKFold(n_splits=a.splits)

    for target in TARGETS:
        y=pd.to_numeric(df[target],errors="coerce")
        if y.isna().any(): raise SystemExit(f"El objetivo {target} contiene faltantes")
        # Entrenamiento en log1p estabiliza escalas marcadamente diferentes entre cargas.
        y_log=np.log1p(y.to_numpy(dtype=float))
        target_results=[]
        for model_name, estimator in all_models.items():
            oof=np.full(len(df),np.nan)
            for fold,(train_idx,test_idx) in enumerate(gkf.split(X,y_log,groups),start=1):
                pipe=pipeline(clone(estimator))
                pipe.fit(X.iloc[train_idx],y_log[train_idx])
                pred=np.expm1(pipe.predict(X.iloc[test_idx])); pred=np.maximum(pred,0)
                oof[test_idx]=pred
                fm=metrics(y.iloc[test_idx].to_numpy(),pred)
                fold_rows.append({"objetivo":target,"modelo":model_name,"fold":fold,
                                  "grupos_prueba":";".join(sorted(groups.iloc[test_idx].unique())),
                                  "n_train":len(train_idx),"n_test":len(test_idx),**fm})
            overall=metrics(y.to_numpy(),oof)
            metric_rows.append({"objetivo":target,"modelo":model_name,"evaluacion":"OOF_GroupKFold",
                                "n":len(df),"n_grupos":groups.nunique(),"n_splits":a.splits,**overall})
            target_results.append((model_name,overall))
            for idx,(actual,pred) in enumerate(zip(y,oof)):
                prediction_rows.append({"objetivo":target,"modelo":model_name,
                    "sistema_archivos":df.loc[idx,"sistema_archivos"],"escenario":df.loc[idx,"escenario"],
                    "repeticion":int(df.loc[idx,"repeticion"]),"grupo":groups.iloc[idx],
                    "real":actual,"prediccion_oof":pred,"error":pred-actual,"error_absoluto":abs(pred-actual)})

        # Seleccion principal por MAE, con RMSE como desempate.
        best_name,best_metrics=min(target_results,key=lambda z:(z[1]["mae"],z[1]["rmse"]))
        baseline=next(m for n,m in target_results if n=="baseline_mediana")
        improvement=(baseline["mae"]-best_metrics["mae"])/baseline["mae"]*100 if baseline["mae"] else np.nan
        best_rows.append({"objetivo":target,"mejor_modelo":best_name,
                          "mae":best_metrics["mae"],"rmse":best_metrics["rmse"],"r2":best_metrics["r2"],
                          "mape_pct":best_metrics["mape_pct"],"mae_baseline":baseline["mae"],
                          "mejora_mae_vs_baseline_pct":improvement})
        final_pipe=pipeline(clone(all_models[best_name])); final_pipe.fit(X,y_log)
        joblib.dump({"pipeline_log_target":final_pipe,"objetivo":target,"features":CAT+NUM,
                     "transformacion_objetivo":"log1p; inversion expm1","grupos_validacion":"sistema_archivos|escenario"},
                    saved/f"modelo_{target}_{best_name}.joblib")

    preds=pd.DataFrame(prediction_rows); folds=pd.DataFrame(fold_rows)
    results=pd.DataFrame(metric_rows); best=pd.DataFrame(best_rows)
    preds.to_csv(out/"predicciones_oof.csv",index=False)
    folds.to_csv(out/"metricas_por_fold.csv",index=False)
    results.to_csv(out/"comparacion_modelos.csv",index=False)
    best.to_csv(out/"mejores_modelos.csv",index=False)

    # Graficos solo para el mejor modelo de cada objetivo.
    for _,br in best.iterrows():
        target=br["objetivo"]; model=br["mejor_modelo"]
        p=preds[(preds.objetivo==target)&(preds.modelo==model)]
        fig,ax=plt.subplots(figsize=(7,6));
        for fs,sub in p.groupby("sistema_archivos"):
            ax.scatter(sub.real,sub.prediccion_oof,label=fs,alpha=.8)
        lo=min(p.real.min(),p.prediccion_oof.min()); hi=max(p.real.max(),p.prediccion_oof.max())
        ax.plot([lo,hi],[lo,hi],linestyle="--"); ax.set_xlabel("Valor real"); ax.set_ylabel("Prediccion fuera de muestra")
        ax.set_title(f"{target}: {model}"); ax.legend(title="Sistema"); fig.tight_layout()
        fig.savefig(plots/f"predicho_vs_real_{target}.png",dpi=180); plt.close(fig)

    lines=["INFORME DE MODELADO PREDICTIVO",f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}",
           f"Archivo: {src}",f"SHA-256 entrada: {digest(src)}",f"Filas: {len(df)}",
           f"Grupos: {groups.nunique()}",f"Validacion: GroupKFold ({a.splits} folds)",
           "Grupo: sistema_archivos|escenario","Transformacion del objetivo: log1p, predicciones invertidas con expm1",
           "", "MEJORES MODELOS:"]
    for _,r in best.iterrows():
        lines.append(f"- {r.objetivo}: {r.mejor_modelo} | MAE={r.mae:.6f} | RMSE={r.rmse:.6f} | "
                     f"R2={r.r2:.6f} | MAPE={r.mape_pct:.3f}% | mejora MAE vs baseline={r.mejora_mae_vs_baseline_pct:.3f}%")
    lines += ["", "LIMITACIONES:",
              "- Existen 75 observaciones, pero solo 15 configuraciones unicas.",
              "- El modelo es una prueba de concepto valida para las condiciones experimentales estudiadas.",
              "- No se utilizaron repeticion, rutas, fecha ni dispositivo sda/sdb como predictores.",
              "- Un R2 negativo indica peor generalizacion que predecir la media en observaciones no vistas."]
    (out/"informe_modelado.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    report={"fecha_utc":datetime.now(timezone.utc).isoformat(),"archivo":str(src),"sha256":digest(src),
            "filas":len(df),"grupos":groups.nunique(),"splits":a.splits,"objetivos":TARGETS,
            "features":CAT+NUM,"modelos":list(all_models),"sklearn_version":sklearn.__version__,
            "mejores_modelos":best.to_dict(orient="records")}
    (out/"informe_modelado.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with open(out/"manifiesto_sha256.txt","w",encoding="utf-8") as f:
        for p in sorted(x for x in out.rglob("*") if x.is_file() and x.name!="manifiesto_sha256.txt"):
            f.write(f"{digest(p)}  {p.relative_to(out)}\n")
    print(f"Modelado finalizado: {out}")
    print(best.to_string(index=False))
    return 0

if __name__=="__main__": sys.exit(main())
