#!/usr/bin/env bash
set -euo pipefail

REPEATS=3
START_SEED=2
LOGDIR="./logs"
OUTFILE="aggregated_results_table7.txt"

# -----------------------------------------------
# Fill this array with experiments
# Format: "name:::command:::results_dir"
# -----------------------------------------------
COMMANDS=(
    # ============================================
    # Table 1: Fully-supervised 10-way, 10-shot Omniglot alphabet recognition
    # Following sbatch script: nsuper=10 (alphabets), nsub=10 (chars/alphabet), nshot=1 (support), num-test=5 (query)
    # ============================================
    "table1_alphabet_10way_10shot:::python run_eval.py --dataset omniglot --model imp --label-ratio 1.0 --nclasses-train 10 --super-classes --nsuperclassestrain 10 --nsuperclasseseval 10 --disable-distractor --num-unlabel 0 --num-unlabel-test 0 --use-test --results ./results/table1/alphabet_10way_10shot"

    # ============================================
    # Table 3: Alphabet and character recognition accuracy
    # ============================================
    # Row 1: ALPHABET -> ALPHABET (10-way 10-shot) - Train on alphabets, test on alphabets - Same as Table 1
    # "table3_alphabet_alphabet:::python run_eval.py --dataset omniglot --model imp --label-ratio 1.0 --nshot 1 --nclasses-train 10 --super-classes --nsuperclassestrain 10 --nsuperclasseseval 10 --disable-distractor --num-unlabel 0 --num-unlabel-test 0 --use-test --results ./results/table1/alphabet_10way_10shot"

    # Row 2: ALPHABET -> CHARS (20-way 1-shot) - Train on alphabets (10-way 10-shot), test on characters (20-way 1-shot)
    "table3_alphabet_chars:::python run_eval.py --dataset omniglot --model imp --label-ratio 1.0 --nclasses-train 10 --super-classes --nsuperclassestrain 10 --nsuperclasseseval 10 --disable-distractor --num-unlabel 0 --num-unlabel-test 0 --results $PWD/results/table3/alphabet_chars_train_seed_{seed} && TRAIN_DIR=$(ls -td $PWD/results/table3/alphabet_chars_train_seed_{seed}/*/ | head -1) && python run_eval.py --dataset omniglot --model imp --label-ratio 1.0 --nclasses-train 20 --nclasses-eval 20 --disable-distractor --use-test --num-unlabel-test 0 --eval --pretrain ${TRAIN_DIR}modelbest.pt --results $PWD/results/table3/alphabet_chars_test_seed_{seed}"

    # # Row 3: CHARS -> CHARS (20-way 1-shot) - Train and test on characters (no superclasses)
    "table3_chars_chars_train:::python run_eval.py --dataset omniglot --model imp --label-ratio 1.0 --nclasses-train 20 --nclasses-eval 20 --disable-distractor --num-unlabel 0 --num-unlabel-test 0 --use-test --results ./results/table3/chars_chars_train"

    # # ============================================
    # # Table 4: Generalization to held-out characters on 10-way, 5-shot alphabet recognition
    # # 10-way 5-shot = 10 alphabets, 5 characters per alphabet, 40% of characters for training, 60% held out
    # # ============================================
    "table4:::python run_eval.py --dataset omniglot --model imp --label-ratio 0.4 --nclasses-train 5 --super-classes --nsuperclassestrain 10 --nsuperclasseseval 10 --disable-distractor --num-unlabel 0 --num-unlabel-test 0 --results $PWD/results/table4/train_seed_{seed} && TRAIN_DIR=$(ls -td $PWD/results/table4/train_seed_{seed}/*/ | head -1) && python run_eval.py --dataset omniglot --model imp --label-ratio 0.6 --nclasses-train 5 --super-classes --nsuperclassestrain 10 --nsuperclasseseval 10 --disable-distractor --use-test --num-unlabel 0 --num-unlabel-test 0 --eval --pretrain ${TRAIN_DIR}modelbest.pt --results $PWD/results/table4/test_seed_{seed}"

    # ============================================
    # Table 7: Semi-supervised few-shot with 40% labeled + 5 unlabeled + 5 distractors
    # "shot" here refers to support images per character (standard few-shot), using 5 chars per alphabet
    # ============================================
    "table7_5way_1shot:::python run_eval.py --dataset omniglot --model imp --mode-ratio 1.0 --label-ratio 0.4 --nclasses-train 5 --nclasses-episode 5 --nclasses-eval 5 --nsuperclassestrain 5 --nsuperclasseseval 5 --super-classes --nshot 1 --num-test 5 --num-unlabel 5 --num-unlabel-test 5 --use-test --results ./results/table7/5way_1shot"
    "table7_5way_5shot:::python run_eval.py --dataset omniglot --model imp --mode-ratio 1.0 --label-ratio 0.4 --nclasses-train 5 --nclasses-episode 5 --nclasses-eval 5 --nsuperclassestrain 5 --nsuperclasseseval 5 --super-classes --nshot 5 --num-test 5 --num-unlabel 5 --num-unlabel-test 5 --use-test --results ./results/table7/5way_5shot"
    "table7_20way_1shot:::python run_eval.py --dataset omniglot --model imp --mode-ratio 1.0 --label-ratio 0.4 --nclasses-train 5 --nclasses-episode 5 --nclasses-eval 5 --nsuperclassestrain 20 --nsuperclasseseval 20 --super-classes --nshot 1 --num-test 5 --num-unlabel 5 --num-unlabel-test 5 --use-test --results ./results/table7/20way_1shot"
    "table7_20way_5shot:::python run_eval.py --dataset omniglot --model imp --mode-ratio 1.0 --label-ratio 0.4 --nclasses-train 5 --nclasses-episode 5 --nclasses-eval 5 --nsuperclassestrain 20 --nsuperclasseseval 20 --super-classes --nshot 5 --num-test 5 --num-unlabel 5 --num-unlabel-test 5 --use-test --results ./results/table7/20way_5shot"
)

mkdir -p "$LOGDIR"
: > "$OUTFILE"

extract_results_dir() {
    # extract the argument after "--results"
    echo "$1" | sed -n 's/.*--results[[:space:]]\+\([^[:space:]]\+\).*/\1/p'
}

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh to activate conda environment." >&2
    exit 1
fi

conda activate r4rr || {
    echo "ERROR: Failed to activate conda environment 'r4rr'" >&2
    exit 1
}

# Count total experiments
TOTAL_EXPERIMENTS=0
for entry in "${COMMANDS[@]}"; do
  entry="$(echo "$entry" | xargs)"
  [[ -z "$entry" || "$entry" =~ ^# ]] && continue
  if [[ "$entry" == *:::* ]]; then
    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
  fi
done

CURRENT_EXPERIMENT=0
START_TIME=$(date +%s)

echo ""
echo "========================================"
echo "STARTING EXPERIMENT SUITE"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Repeats per experiment: $REPEATS"
echo "Total runs: $((TOTAL_EXPERIMENTS * REPEATS))"
echo "========================================"

for entry in "${COMMANDS[@]}"; do
  entry="$(echo "$entry" | xargs)"
  [[ -z "$entry" || "$entry" =~ ^# ]] && continue

  if [[ "$entry" != *:::* ]]; then
    echo "Skipping malformed entry (missing :::): $entry" >&2
    continue
  fi

  name="${entry%%:::*}"
  cmd="${entry#*:::}"

  name="$(echo "$name" | xargs)"
  cmd="$(echo "$cmd" | xargs)"

  # pull results dir from the command itself
  results_dir="$(extract_results_dir "$cmd")"

  if [[ -z "$results_dir" ]]; then
      echo "ERROR: command for $name does not contain a --results flag" >&2
      continue
  fi

  CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
  PERCENT=$((CURRENT_EXPERIMENT * 100 / TOTAL_EXPERIMENTS))

  # Calculate elapsed and estimated time
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))
  if [ $CURRENT_EXPERIMENT -gt 1 ]; then
    AVG_TIME=$((ELAPSED / (CURRENT_EXPERIMENT - 1)))
    REMAINING=$((AVG_TIME * (TOTAL_EXPERIMENTS - CURRENT_EXPERIMENT + 1)))
    EST_REMAINING=$(printf '%02d:%02d:%02d' $((REMAINING/3600)) $((REMAINING%3600/60)) $((REMAINING%60)))
  else
    EST_REMAINING="calculating..."
  fi

  mkdir -p "$results_dir"
  echo ""
  echo "========================================"
  echo "EXPERIMENT $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS ($PERCENT%)"
  echo "Name: $name"
  echo "Results → $results_dir"
  echo "Est. time remaining: $EST_REMAINING"
  echo "========================================"

  for ((i=0;i<REPEATS;i++)); do
    seed=$(((START_SEED)**i))

    if [[ "$cmd" == *"{seed}"* ]]; then
      full_cmd="${cmd//\{seed\}/$seed}"
    else
      full_cmd="$cmd --seed $seed"
    fi

    TOTAL_RUN=$((((CURRENT_EXPERIMENT - 1) * REPEATS) + i + 1))
    OVERALL_TOTAL=$((TOTAL_EXPERIMENTS * REPEATS))
    RUN_PERCENT=$((TOTAL_RUN * 100 / OVERALL_TOTAL))

    logfile="$LOGDIR/${name}_seed_${seed}.log"
    echo ""
    echo "──────────────────────────────────────"
    echo "  Run $((i+1))/$REPEATS (seed=$seed) | Overall: $TOTAL_RUN/$OVERALL_TOTAL ($RUN_PERCENT%)"
    echo "  Log: $logfile"
    echo "──────────────────────────────────────"

    # Run command in subshell with conda environment activated
    # Show progress on single overwriting line, save all output to log file
    (
        eval "$(conda shell.bash hook)"
        conda activate r4rr
        eval "$full_cmd"
    ) 2>&1 | tee "$logfile" | while IFS= read -r line; do
        if echo "$line" | grep -qE "(Episode|Epoch|iteration|Loss|accuracy)"; then
            printf "\r\033[K  %s" "$line"
        fi
    done
    echo ""  # New line after completion

    exit_code=${PIPESTATUS[0]}

    acc_file="$results_dir/accuracies.txt"
    if [ $exit_code -ne 0 ]; then
        echo "  ⚠️  Command failed with exit code $exit_code" >&2
        echo "$name seed_$seed FAILED" >> "$OUTFILE"
    else
        if [[ -f "$acc_file" ]]; then
            tail -n 1 "$acc_file" | awk -v n="$name" -v s="$seed" '{print n, s, $0}' >> "$OUTFILE"
            echo "  ✓ Run completed successfully"
        else
            echo "$name seed_$seed EXTRACTION_FAILED" >> "$OUTFILE"
            echo "  ⚠️  Run completed but accuracy extraction failed"
        fi
    fi

  done
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_TIME=$(printf '%02d:%02d:%02d' $((TOTAL_ELAPSED/3600)) $((TOTAL_ELAPSED%3600/60)) $((TOTAL_ELAPSED%60)))

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETED ✓"
echo "========================================"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Total runs: $((TOTAL_EXPERIMENTS * REPEATS))"
echo "Total time: $TOTAL_TIME"
echo ""
echo "Aggregated results: $OUTFILE"
echo "Logs directory: $LOGDIR"
echo ""
echo "Results summary:"
cat "$OUTFILE"
echo "========================================"