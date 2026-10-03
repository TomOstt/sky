#!/bin/bash
# Passive cross-validation: SMC PSTR (via macmon, 1 s) vs AppleSmartBattery PowerTelemetryData accumulators (60 s cadence)
W=$(cd "$(dirname "$0")" && pwd)
$W/macmon pipe -i 1000 -s 200 > $W/xval_macmon.jsonl 2>/dev/null &
MM=$!
for i in $(seq 1 100); do
  t=$(date +%s)
  p=$(ioreg -rn AppleSmartBattery -w0 | grep -o '"PowerTelemetryData" = {[^}]*}' | grep -oE '"(SystemPowerIn|SystemLoad|AccumulatedSystemPowerIn|SystemPowerInAccumulatorCount|AccumulatedSystemLoad|SystemLoadAccumulatorCount|BatteryPower|WallEnergyEstimate|AdapterEfficiencyLoss)"=[0-9]+' | tr '\n' ' ')
  u=$(ioreg -rn AppleSmartBattery -w0 | grep -oE '"UpdateTime" = [0-9]+' | grep -oE '[0-9]+$')
  echo "$t upd=$u $p"
  sleep 2
done > $W/xval_ioreg.txt
wait $MM
