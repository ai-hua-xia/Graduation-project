#!/usr/bin/env bash
set -euo pipefail

# Run 10 CARLA servers + 10 collectors in parallel.
# This script collects in two phases to balance global action ratios:
#   Phase A (straight-heavy) and Phase B (turn-heavy).
# Edit variables below as needed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARLA_SH="${ROOT_DIR}/CARLA_0.9.16/CarlaUE4.sh"
LOG_DIR="${ROOT_DIR}/logs"

MAP_NAME="${MAP_NAME:-Town04}"
DATA_DIR="${DATA_DIR:-data/raw_action_corr_v3}"
EPISODES_PER_WORKER="${EPISODES_PER_WORKER:-150}"
PHASE_A_EPISODES_PER_WORKER="${PHASE_A_EPISODES_PER_WORKER:-$EPISODES_PER_WORKER}"
PHASE_B_EPISODES_PER_WORKER="${PHASE_B_EPISODES_PER_WORKER:-$EPISODES_PER_WORKER}"
START_EPISODE="${START_EPISODE:-1}"
FRAMES_PER_EPISODE="${FRAMES_PER_EPISODE:-80}"

# Collector settings (relaxed but safe)
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-6}"
MIN_CORR="${MIN_CORR:-0.15}"
MIN_DELTA="${MIN_DELTA:-0.008}"
MIN_SPEED="${MIN_SPEED:-0.35}"
MIN_STRAIGHT_RATIO="${MIN_STRAIGHT_RATIO:-0.00}"
MIN_TURN_RATIO="${MIN_TURN_RATIO:-0.00}"
MIN_MID_RATIO="${MIN_MID_RATIO:-0.00}"
MIN_HARD_RATIO="${MIN_HARD_RATIO:-0.00}"
MAX_HARD_RATIO="${MAX_HARD_RATIO:-1.00}"
MAX_COLLISIONS="${MAX_COLLISIONS:-0}"
MAX_LANE_INVASIONS="${MAX_LANE_INVASIONS:-0}"
MAX_STUCK_FRAMES="${MAX_STUCK_FRAMES:-6}"
LOW_THROTTLE="${LOW_THROTTLE:-0.30}"
HIGH_THROTTLE="${HIGH_THROTTLE:-0.50}"
MID_STEER="${MID_STEER:-0.12}"
HARD_STEER="${HARD_STEER:-0.20}"
SEGMENT_LEN="${SEGMENT_LEN:-6}"
SEGMENT_JITTER="${SEGMENT_JITTER:-2}"
WARMUP_TICKS="${WARMUP_TICKS:-15}"
STEER_NOISE="${STEER_NOISE:-0.02}"
THROTTLE_NOISE="${THROTTLE_NOISE:-0.02}"
PREVIEW_EVERY="${PREVIEW_EVERY:-50}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-120}"
CONNECT_RETRIES="${CONNECT_RETRIES:-30}"
CONNECT_RETRY_WAIT="${CONNECT_RETRY_WAIT:-3}"
PHASE_A_STRAIGHT_RATIO="${PHASE_A_STRAIGHT_RATIO:-0.70}"
PHASE_A_MID_RATIO="${PHASE_A_MID_RATIO:-0.20}"
PHASE_A_HARD_RATIO="${PHASE_A_HARD_RATIO:-0.10}"

PHASE_B_STRAIGHT_RATIO="${PHASE_B_STRAIGHT_RATIO:-0.30}"
PHASE_B_MID_RATIO="${PHASE_B_MID_RATIO:-0.40}"
PHASE_B_HARD_RATIO="${PHASE_B_HARD_RATIO:-0.30}"

PHASE_GAP="${PHASE_GAP:-0}"

PORTS=(2000 2010 2020 2030 2040 2050 2060 2070 2080 2090)
CARLA_BOOT_WAIT="${CARLA_BOOT_WAIT:-90}"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${CARLA_SH}" ]]; then
  echo "ERROR: CARLA launcher not found or not executable: ${CARLA_SH}"
  exit 1
fi

echo "Starting 10 CARLA servers..."
for port in "${PORTS[@]}"; do
  SDL_AUDIODRIVER=dummy "${CARLA_SH}" \
    -RenderOffScreen -nosound -quality-level=Low -windowed -ResX=800 -ResY=600 \
    -carla-port="${port}" \
    > "${LOG_DIR}/carla_${port}.log" 2>&1 &
  sleep 2
done

echo "Waiting for CARLA servers to boot (${CARLA_BOOT_WAIT}s)..."
sleep "${CARLA_BOOT_WAIT}"

launch_collectors() {
  local phase_name="$1"
  local phase_offset="$2"
  local episodes_per_worker="$3"
  local straight_ratio="$4"
  local mid_ratio="$5"
  local hard_ratio="$6"
  local log_tag="$7"

  echo "Launching collectors (${phase_name})..."
  for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    start=$((START_EPISODE + phase_offset + i * episodes_per_worker))
    end=$((start + episodes_per_worker - 1))

    python collect/collect_data_action_correlated.py \
      --host localhost --port "${port}" \
      --episode-start "${start}" --episode-end "${end}" \
      --map "${MAP_NAME}" \
      --data-dir "${DATA_DIR}" \
      --sample-interval "${SAMPLE_INTERVAL}" \
      --frames-per-episode "${FRAMES_PER_EPISODE}" \
      --min-corr "${MIN_CORR}" --min-delta "${MIN_DELTA}" --min-speed "${MIN_SPEED}" \
      --min-straight-ratio "${MIN_STRAIGHT_RATIO}" \
      --min-turn-ratio "${MIN_TURN_RATIO}" \
      --min-mid-ratio "${MIN_MID_RATIO}" \
      --min-hard-ratio "${MIN_HARD_RATIO}" \
      --max-hard-ratio "${MAX_HARD_RATIO}" \
      --max-collisions "${MAX_COLLISIONS}" \
      --max-lane-invasions "${MAX_LANE_INVASIONS}" \
      --max-stuck-frames "${MAX_STUCK_FRAMES}" \
      --low-throttle "${LOW_THROTTLE}" --high-throttle "${HIGH_THROTTLE}" \
      --mid-steer "${MID_STEER}" --hard-steer "${HARD_STEER}" \
      --straight-ratio "${straight_ratio}" \
      --mid-ratio "${mid_ratio}" \
      --hard-ratio "${hard_ratio}" \
      --steer-noise "${STEER_NOISE}" --throttle-noise "${THROTTLE_NOISE}" \
      --segment-len "${SEGMENT_LEN}" --segment-jitter "${SEGMENT_JITTER}" \
      --warmup-ticks "${WARMUP_TICKS}" \
      --preview-every "${PREVIEW_EVERY}" \
      --client-timeout "${CLIENT_TIMEOUT}" \
      --connect-retries "${CONNECT_RETRIES}" \
      --connect-retry-wait "${CONNECT_RETRY_WAIT}" \
      > "${LOG_DIR}/collect_${port}_${log_tag}.log" 2>&1 &
  done
}

if [[ "${PHASE_GAP}" -eq 0 ]]; then
  PHASE_GAP=$((PHASE_A_EPISODES_PER_WORKER * 10))
fi

launch_collectors "Phase A (straight-heavy)" 0 \
  "${PHASE_A_EPISODES_PER_WORKER}" \
  "${PHASE_A_STRAIGHT_RATIO}" "${PHASE_A_MID_RATIO}" "${PHASE_A_HARD_RATIO}" "phaseA"

echo "Phase A collectors started. Logs in ${LOG_DIR}/collect_*_phaseA.log"
echo "Tip: tail -f ${LOG_DIR}/collect_2000_phaseA.log"
echo ""
echo "Waiting for Phase A to finish..."
wait

launch_collectors "Phase B (turn-heavy)" "${PHASE_GAP}" \
  "${PHASE_B_EPISODES_PER_WORKER}" \
  "${PHASE_B_STRAIGHT_RATIO}" "${PHASE_B_MID_RATIO}" "${PHASE_B_HARD_RATIO}" "phaseB"

echo "Phase B collectors started. Logs in ${LOG_DIR}/collect_*_phaseB.log"
echo "Tip: tail -f ${LOG_DIR}/collect_2000_phaseB.log"
wait

echo "All collectors finished."
