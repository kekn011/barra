#!/bin/bash
# sweep-build.sh FMT [CONFIGS="WG TPR R;..."]
FMT=$1; shift
cd "$(dirname "$0")"
CONFIGS=${CONFIGS:-"64 16 1;64 16 2;64 16 4;64 8 2;64 8 4;128 16 2"}
echo "$CONFIGS" | tr ';' '\n' | while read WG TPR R; do bash build.sh $FMT $WG $TPR $R | tail -1; done
