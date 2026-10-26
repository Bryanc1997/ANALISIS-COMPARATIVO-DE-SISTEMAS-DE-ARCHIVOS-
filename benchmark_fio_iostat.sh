#!/usr/bin/env bash
set -Eeuo pipefail

# benchmark_fio_iostat.sh
# Ejecuta cargas FIO por separado y captura iostat simultaneamente.
# No crea, elimina ni formatea sistemas de archivos.

usage() {
  cat <<'EOF'
Uso:
  sudo ./benchmark_fio_iostat.sh -f SISTEMA -d DIRECTORIO [opciones]

Obligatorios:
  -f SISTEMA       Etiqueta: ext4, xfs o zfs
  -d DIRECTORIO    Punto montado dedicado a pruebas, por ejemplo /mnt/fio-test

Opciones:
  -o DIRECTORIO    Directorio de resultados (predeterminado: ./resultados_fio)
  -s TAMANO        Tamano del archivo FIO (predeterminado: 4G)
  -t SEGUNDOS      Duracion medida por escenario (predeterminado: 60)
  -r NUMERO        Repeticiones por escenario (predeterminado: 5)
  -q NUMERO        Profundidad de cola (predeterminado: 32)
  -j NUMERO        Numero de jobs (predeterminado: 1)
  -e MOTOR         Motor FIO (predeterminado: libaio)
  -i SEGUNDOS      Intervalo de iostat (predeterminado: 1)
  -w SEGUNDOS      Espera entre pruebas (predeterminado: 30)
  -h                Mostrar ayuda

Ejemplo:
  sudo ./benchmark_fio_iostat.sh -f zfs -d /mnt/fio-test -s 4G -t 60 -r 5
EOF
}

FS_LABEL=""
TEST_DIR=""
OUTPUT_ROOT="./resultados_fio"
FILE_SIZE="4G"
RUNTIME=60
REPETITIONS=5
IODEPTH=32
NUMJOBS=1
IOENGINE="libaio"
IOSTAT_INTERVAL=1
WAIT_TIME=30
RAMP_TIME=10

while getopts ":f:d:o:s:t:r:q:j:e:i:w:h" opt; do
  case "$opt" in
    f) FS_LABEL="${OPTARG,,}" ;;
    d) TEST_DIR="$OPTARG" ;;
    o) OUTPUT_ROOT="$OPTARG" ;;
    s) FILE_SIZE="$OPTARG" ;;
    t) RUNTIME="$OPTARG" ;;
    r) REPETITIONS="$OPTARG" ;;
    q) IODEPTH="$OPTARG" ;;
    j) NUMJOBS="$OPTARG" ;;
    e) IOENGINE="$OPTARG" ;;
    i) IOSTAT_INTERVAL="$OPTARG" ;;
    w) WAIT_TIME="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Error: -$OPTARG requiere un valor." >&2; usage; exit 2 ;;
    \?) echo "Error: opcion desconocida -$OPTARG." >&2; usage; exit 2 ;;
  esac
done

[[ -n "$FS_LABEL" && -n "$TEST_DIR" ]] || { usage; exit 2; }
[[ "$FS_LABEL" =~ ^(ext4|xfs|zfs)$ ]] || { echo "Error: sistema permitido: ext4, xfs o zfs." >&2; exit 2; }
[[ "$RUNTIME" =~ ^[1-9][0-9]*$ && "$REPETITIONS" =~ ^[1-9][0-9]*$ ]] || { echo "Error: tiempo y repeticiones deben ser enteros positivos." >&2; exit 2; }
[[ "$IODEPTH" =~ ^[1-9][0-9]*$ && "$NUMJOBS" =~ ^[1-9][0-9]*$ ]] || { echo "Error: iodepth y numjobs deben ser enteros positivos." >&2; exit 2; }

for cmd in fio iostat findmnt df lsblk awk grep sed date hostname sync; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Error: falta el comando '$cmd'." >&2; exit 3; }
done

[[ -d "$TEST_DIR" ]] || { echo "Error: no existe $TEST_DIR." >&2; exit 4; }
findmnt -T "$TEST_DIR" >/dev/null 2>&1 || { echo "Error: $TEST_DIR no pertenece a un sistema montado." >&2; exit 4; }
[[ -w "$TEST_DIR" ]] || { echo "Error: $TEST_DIR no permite escritura." >&2; exit 4; }

ACTUAL_FS=$(findmnt -n -o FSTYPE -T "$TEST_DIR")
if [[ "$ACTUAL_FS" != "$FS_LABEL" && !( "$FS_LABEL" == "zfs" && "$ACTUAL_FS" == "zfs" ) ]]; then
  echo "Error: la etiqueta indicada es '$FS_LABEL', pero findmnt detecta '$ACTUAL_FS'." >&2
  exit 4
fi

AVAILABLE_KB=$(df -Pk "$TEST_DIR" | awk 'NR==2 {print $4}')
# FIO valida tambien el espacio. Esta comprobacion solo evita errores obvios.
[[ "$AVAILABLE_KB" =~ ^[0-9]+$ ]] || { echo "Error: no se pudo determinar el espacio libre." >&2; exit 4; }

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$(realpath -m "$OUTPUT_ROOT/${FS_LABEL}_${TIMESTAMP}")
mkdir -p "$RUN_DIR" "$RUN_DIR/fio" "$RUN_DIR/iostat" "$RUN_DIR/logs"

TEST_FILE="$TEST_DIR/fio_benchmark_${FS_LABEL}.dat"
MARKER="$RUN_DIR/.benchmark_in_progress"
touch "$MARKER"
IOSTAT_PID=""

cleanup() {
  if [[ -n "${IOSTAT_PID:-}" ]] && kill -0 "$IOSTAT_PID" 2>/dev/null; then
    kill -INT "$IOSTAT_PID" 2>/dev/null || true
    wait "$IOSTAT_PID" 2>/dev/null || true
  fi
  rm -f "$MARKER"
}
trap cleanup EXIT INT TERM

{
  echo "fecha_inicio=$(date --iso-8601=seconds)"
  echo "etiqueta_fs=$FS_LABEL"
  echo "fs_detectado=$ACTUAL_FS"
  echo "directorio_prueba=$TEST_DIR"
  echo "archivo_prueba=$TEST_FILE"
  echo "tamano=$FILE_SIZE"
  echo "runtime=$RUNTIME"
  echo "ramp_time=$RAMP_TIME"
  echo "repeticiones=$REPETITIONS"
  echo "iodepth=$IODEPTH"
  echo "numjobs=$NUMJOBS"
  echo "ioengine=$IOENGINE"
  echo "fio_version=$(fio --version)"
  echo "kernel=$(uname -r)"
  echo "host=$(hostname)"
  echo
  echo "--- findmnt ---"
  findmnt -T "$TEST_DIR"
  echo
  echo "--- df ---"
  df -Th "$TEST_DIR"
  echo
  echo "--- lsblk ---"
  lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,ROTA,MODEL
  echo
  echo "--- memoria ---"
  free -h 2>/dev/null || true
  echo
  echo "--- cpu ---"
  lscpu 2>/dev/null || true
  if [[ "$FS_LABEL" == "zfs" ]]; then
    echo
    echo "--- propiedades ZFS ---"
    command -v zfs >/dev/null 2>&1 && zfs get -H -o name,property,value mountpoint,compression,atime,dedup,primarycache,recordsize,sync 2>/dev/null || true
    command -v zpool >/dev/null 2>&1 && zpool status 2>/dev/null || true
  fi
} > "$RUN_DIR/entorno.txt" 2>&1

cat > "$RUN_DIR/configuracion.csv" <<EOF
sistema_archivos,directorio,tamano,runtime_seg,ramp_time_seg,repeticiones,iodepth,numjobs,ioengine,intervalo_iostat_seg
$FS_LABEL,$TEST_DIR,$FILE_SIZE,$RUNTIME,$RAMP_TIME,$REPETITIONS,$IODEPTH,$NUMJOBS,$IOENGINE,$IOSTAT_INTERVAL
EOF

prepare_file() {
  echo "Preparando archivo de prueba de $FILE_SIZE..."
  fio --name=preparacion \
      --filename="$TEST_FILE" \
      --rw=write --bs=1M --size="$FILE_SIZE" \
      --ioengine="$IOENGINE" --direct=1 \
      --iodepth="$IODEPTH" --numjobs=1 \
      --group_reporting=1 \
      --output-format=json \
      --output="$RUN_DIR/logs/preparacion.json"
  sync
  sleep "$WAIT_TIME"
}

run_case() {
  local case_name="$1" rw="$2" bs="$3" mixread="${4:-}"
  local rep fio_out io_out log_out fio_rc

  for ((rep=1; rep<=REPETITIONS; rep++)); do
    fio_out="$RUN_DIR/fio/${FS_LABEL}_${case_name}_rep${rep}.json"
    io_out="$RUN_DIR/iostat/${FS_LABEL}_${case_name}_rep${rep}.txt"
    log_out="$RUN_DIR/logs/${FS_LABEL}_${case_name}_rep${rep}.log"

    echo "[$(date +%T)] $FS_LABEL | $case_name | repeticion $rep/$REPETITIONS"

    # -y omite la estadistica acumulada desde el arranque; -x produce detalle extendido.
    iostat -x -y -d "$IOSTAT_INTERVAL" > "$io_out" 2>&1 &
    IOSTAT_PID=$!
    sleep 2

    set +e
    fio --name="$case_name" \
        --filename="$TEST_FILE" \
        --rw="$rw" --bs="$bs" --size="$FILE_SIZE" \
        --ioengine="$IOENGINE" --direct=1 \
        --time_based=1 --runtime="$RUNTIME" --ramp_time="$RAMP_TIME" \
        --iodepth="$IODEPTH" --numjobs="$NUMJOBS" \
        --group_reporting=1 --invalidate=1 \
        --randrepeat=1 --random_generator=tausworthe64 \
        ${mixread:+--rwmixread="$mixread"} \
        --output-format=json --output="$fio_out" \
        > "$log_out" 2>&1
    fio_rc=$?
    set -e

    kill -INT "$IOSTAT_PID" 2>/dev/null || true
    wait "$IOSTAT_PID" 2>/dev/null || true
    IOSTAT_PID=""

    if (( fio_rc != 0 )); then
      echo "Error: FIO fallo en $case_name, repeticion $rep. Revisa $log_out" >&2
      exit "$fio_rc"
    fi

    sync
    sleep "$WAIT_TIME"
  done
}

prepare_file
run_case "lectura_secuencial"  "read"     "1M"
run_case "escritura_secuencial" "write"   "1M"
run_case "lectura_aleatoria"    "randread" "4k"
run_case "escritura_aleatoria"  "randwrite" "4k"
run_case "mixta_aleatoria_70r30w" "randrw" "4k" "70"

rm -f "$TEST_FILE"
sync

echo "fecha_fin=$(date --iso-8601=seconds)" >> "$RUN_DIR/entorno.txt"
echo "Pruebas finalizadas correctamente."
echo "Resultados: $RUN_DIR"
