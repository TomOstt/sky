#!/bin/bash
# Poll AppleSmartBattery PowerTelemetryData (sudoless) every ~0.5 s
for i in $(seq 1 ${1:-20}); do
  t=$(python3 -c 'import time;print(f"{time.time():.3f}")')
  line=$(ioreg -rn AppleSmartBattery -w0 | grep -o '"PowerTelemetryData" = {[^}]*}')
  pin=$(echo "$line" | grep -oE '"SystemPowerIn"=[0-9]+' | cut -d= -f2)
  load=$(echo "$line" | grep -oE '"SystemLoad"=[0-9]+' | cut -d= -f2)
  acc=$(echo "$line" | grep -oE '"AccumulatedSystemPowerIn"=[0-9]+' | cut -d= -f2)
  cnt=$(echo "$line" | grep -oE '"SystemPowerInAccumulatorCount"=[0-9]+' | cut -d= -f2)
  eacc=$(echo "$line" | grep -oE '"AccumulatedSystemEnergyConsumed"=[0-9]+' | cut -d= -f2)
  econs=$(echo "$line" | grep -oE '"SystemEnergyConsumed"=[0-9]+' | cut -d= -f2)
  lacc=$(echo "$line" | grep -oE '"AccumulatedSystemLoad"=[0-9]+' | cut -d= -f2)
  lcnt=$(echo "$line" | grep -oE '"SystemLoadAccumulatorCount"=[0-9]+' | cut -d= -f2)
  echo "$t pin=$pin load=$load accPin=$acc cntPin=$cnt accLoad=$lacc cntLoad=$lcnt sysE=$econs accSysE=$eacc"
  sleep 0.5
done
