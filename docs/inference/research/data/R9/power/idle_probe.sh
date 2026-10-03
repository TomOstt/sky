#!/bin/bash
# Idle probe: macmon (IOReport + SMC PSTR) and AppleSmartBattery PowerTelemetryData, run under benchlock
W=$(dirname "$0")
date +%s.%N > $W/idle_probe.start
$W/macmon pipe -s 15 -i 200 --soc-info > $W/macmon_idle_200ms.jsonl 2> $W/macmon_idle.err
$W/macmon pipe -s 5 -i 1000 > $W/macmon_idle_1000ms.jsonl 2>> $W/macmon_idle.err
ioreg -rn AppleSmartBattery -w0 | grep -o '"PowerTelemetryData" = {[^}]*}' > $W/ioreg_after_macmon.txt
date +%s.%N > $W/idle_probe.end
