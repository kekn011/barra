#!/system/bin/sh
# sweep-run.sh <fmtname> <fmtnum> <xmode> [M K]  -> alle gemv_<fmt>_w*_t*_r*.spv, RPW aus dem Namen
cd /data/local/tmp/gemv
FMT=$1; N=$2; XM=$3; M=${4:-8960}; K=${5:-1536}
for f in gemv_${FMT}_w*.spv; do
  b=${f%.spv}; rest=${b#gemv_${FMT}_w}; WG=${rest%%_*}; rest=${rest#*_t}; TPR=${rest%%_*}; R=${rest#*_r}
  RPW=$(( (WG/TPR)*R ))
  printf "%-26s " "$b"; ./gpugemv $f $N $M $K 50 $RPW $XM | tail -1
done
