grep -aiE 'exccause|epc1|fatal|illegal|ExceptionOnDsp|Core Fatal|runtime error|EPC' /data/local/tmp/tie-gxpd.log | tail -25
echo "=== last 25 lines ==="
tail -25 /data/local/tmp/tie-gxpd.log
