#!/usr/bin/env bash
set -euo pipefail

echo "Starting Muon Search Sweep..."
out_csv="muon_search_results.csv"
echo "amplitude,final_mass,mass_density" > "$out_csv"

# amplitudes to test
AMPLITUDES="0.1 0.2 0.3 0.4 0.5 0.7 1.0"

SCRIPT="reports/yang_mills/run_excitation_chain.py"
CFG="reports/yang_mills/configs/step2_muon.yml"

for AMP in $AMPLITUDES; do
  echo "--- RUNNING WITH AMPLITUDE = $AMP ---"

  # run and capture output
  LOG_OUTPUT=$(python "$SCRIPT" --config "$CFG" --step 2 --amplitude "$AMP" 2>&1 || true)

  # pull the single machine-readable line emitted by the runner
  SUMMARY_LINE=$(echo "$LOG_OUTPUT" | grep '^SUMMARY ' || true)
  if [ -z "$SUMMARY_LINE" ]; then
    echo "WARNING: No SUMMARY line found; marking as NaN"
    echo "$AMP,NaN,NaN" >> "$out_csv"
    continue
  fi

  # parse mass/density fields
  FINAL_MASS=$(echo "$SUMMARY_LINE"   | awk '{for(i=1;i<=NF;i++){if($i ~ /^mass=/){split($i,a,"=");print a[2]}}}')
  MASS_DENSITY=$(echo "$SUMMARY_LINE" | awk '{for(i=1;i<=NF;i++){if($i ~ /^density=/){split($i,a,"=");print a[2]}}}')

  # fallbacks
  if [ -z "$FINAL_MASS" ]; then FINAL_MASS="NaN"; fi
  if [ -z "$MASS_DENSITY" ]; then MASS_DENSITY="NaN"; fi

  echo "$AMP,$FINAL_MASS,$MASS_DENSITY" >> "$out_csv"
  echo "--- COMPLETED: Amplitude=$AMP, MassDensity=$MASS_DENSITY ---"
done

echo "Muon Search Sweep complete. Results saved to $out_csv"