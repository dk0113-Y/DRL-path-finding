@echo off
setlocal

echo ============================================================
echo [phase3] round A: adaptation / performance-priority
echo ============================================================

python train_q_agent.py ^
  --run-name phase3_adapt_perf_priority ^
  --device cuda ^
  --budget-mode env_steps ^
  --total-env-steps 500000 ^
  --warmup-steps 4000 ^
  --collect-steps-per-iter 16 ^
  --learner-updates-per-iter 2 ^
  --train-every-env-steps 16 ^
  --batch-size 128 ^
  --min-replay-size 4000 ^
  --replay-capacity 100000 ^
  --gamma 0.99 ^
  --n-step 3 ^
  --learning-rate 1e-4 ^
  --target-update-interval 1000 ^
  --epsilon-start 1.0 ^
  --epsilon-end 0.03 ^
  --epsilon-decay-steps 400000 ^
  --rows 40 ^
  --cols 60 ^
  --obs-size 6 ^
  --scan-radius 10 ^
  --max-accessible-blocks 16 ^
  --max-entries-per-block 8 ^
  --obstacle-ratio 0.20 ^
  --reward-info-scale 3.0 ^
  --reward-obstacle-weight 0.25 ^
  --reward-step-penalty 0.015 ^
  --reward-terminal-bonus 20.0 ^
  --reward-revisit-penalty 0.07 ^
  --reward-turn-penalty-scale 0.03 ^
  --reward-turn-weight-45 0.0 ^
  --reward-turn-weight-90 0.3333333333 ^
  --reward-turn-weight-135 0.6666666667 ^
  --reward-turn-weight-180 1.0 ^
  --reward-timeout-penalty 8.0 ^
  --use-fixed-train-episode-seeds ^
  --fixed-train-episode-seed-base 20259323 ^
  --use-fixed-eval-seeds ^
  --fixed-final-probe-seed-base 20261323

if errorlevel 1 (
  echo [phase3] round A failed, stop here.
  exit /b 1
)

echo ============================================================
echo [phase3] round B: adaptation / smoothness-priority
echo ============================================================

python train_q_agent.py ^
  --run-name phase3_adapt_smooth_priority ^
  --device cuda ^
  --budget-mode env_steps ^
  --total-env-steps 500000 ^
  --warmup-steps 4000 ^
  --collect-steps-per-iter 16 ^
  --learner-updates-per-iter 2 ^
  --train-every-env-steps 16 ^
  --batch-size 128 ^
  --min-replay-size 4000 ^
  --replay-capacity 100000 ^
  --gamma 0.99 ^
  --n-step 3 ^
  --learning-rate 1e-4 ^
  --target-update-interval 1000 ^
  --epsilon-start 1.0 ^
  --epsilon-end 0.03 ^
  --epsilon-decay-steps 400000 ^
  --rows 40 ^
  --cols 60 ^
  --obs-size 6 ^
  --scan-radius 10 ^
  --max-accessible-blocks 16 ^
  --max-entries-per-block 8 ^
  --obstacle-ratio 0.20 ^
  --reward-info-scale 3.0 ^
  --reward-obstacle-weight 0.25 ^
  --reward-step-penalty 0.018 ^
  --reward-terminal-bonus 20.0 ^
  --reward-revisit-penalty 0.10 ^
  --reward-turn-penalty-scale 0.06 ^
  --reward-turn-weight-45 0.0 ^
  --reward-turn-weight-90 0.3333333333 ^
  --reward-turn-weight-135 0.85 ^
  --reward-turn-weight-180 1.30 ^
  --reward-timeout-penalty 8.0 ^
  --use-fixed-train-episode-seeds ^
  --fixed-train-episode-seed-base 20259323 ^
  --use-fixed-eval-seeds ^
  --fixed-final-probe-seed-base 20261323

if errorlevel 1 (
  echo [phase3] round B failed.
  exit /b 1
)

echo ============================================================
echo [phase3] both rounds finished successfully.
echo ============================================================
exit /b 0
