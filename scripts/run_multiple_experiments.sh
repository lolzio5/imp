#!/usr/bin/env bash
set -euo pipefail

REPEATS=3
START_SEED=0
LOGDIR="./logs"
OUTFILE="aggregated_results.txt"

# -----------------------------------------------
# Fill this array with experiments
# Format: "name:::command:::results_dir"
# -----------------------------------------------
COMMANDS=(

    # Table 1: fully supervised 10-way, 10-shot
    "table1_10way_10shot:::python run_eval.py --dataset omniglot --model kmeans-refine --nshot 10 --nclasses-episode 10 --label-ratio 1 --results ./results/table1:::./results/table1"

    # Table 3: ALPHABET->CHARS (train vinyals, test lake, 20-way 1-shot)
    "table3_alphabet_chars:::python run_eval.py --dataset omniglot --model kmeans-refine --split-def vinyals --split train --train-ratio 1 --nclasses-episode 20 --nshot 1 --results ./results/table3/alphabet_chars:::./results/table3/alphabet_chars"
    "table3_chars_chars:::python run_eval.py --dataset omniglot --model kmeans-refine --split-def lake --split train --train-ratio 1 --nclasses-episode 20 --nshot 1 --results ./results/table3/chars_chars:::./results/table3/chars_chars"

    # Table 4: 40% train / 60% test generalization, 10-way, 5-shot
    "table4_train:::python run_eval.py --dataset omniglot --model kmeans-refine --split-def lake --split train --train-ratio 0.4 --nclasses-episode 10 --nshot 5 --results ./results/table4/train:::./results/table4/train"
    "table4_test:::python run_eval.py --dataset omniglot --model kmeans-refine --split-def lake --split test --train-ratio 0.4 --nclasses-episode 10 --nshot 5 --results ./results/table4/test:::./results/table4/test"

    # Table 7: semi-supervised few-shot, 40% labeled + 5 unlabeled + 5 distractors, 5-way 1-shot
    "table7_semi_supervised:::python run_eval.py --dataset omniglot --model kmeans-refine --split-def lake --split train --train-ratio 0.4 --nclasses-episode 5 --nshot 1 --num-unlabel 5 --results ./results/table7:::./results/table7"

)

# -----------------------------------------------
# Script scaffold (as in your original)
# -----------------------------------------------
usage(){
  cat <<EOF
Usage: $0

Edit the COMMANDS array in this script to add experiments. Each element must be:
  "name:::command:::results_dir"

The script will run each command REPEATS times, varying the seed from START_SEED..,
saving per-run logs to $LOGDIR and appending the last line of results_dir/accuracies.txt
to $OUTFILE.
EOF
}

if [ ${#COMMANDS[@]} -eq 0 ]; then
  echo "No commands found in the COMMANDS array. Open the file and add experiments." >&2
  usage
  exit 1
fi

mkdir -p "$LOGDIR"
: > "$OUTFILE"

for entry in "${COMMANDS[@]}"; do
  IFS=':::' read -r name cmd results_dir <<< "$entry"
  if [ -z "$name" ] || [ -z "$cmd" ] || [ -z "$results_dir" ]; then
    echo "Skipping malformed entry: $entry" >&2
    continue
  fi
  mkdir -p "$results_dir"
  echo "Starting experiment: $name"

  for ((i=0;i<REPEATS;i++)); do
    seed=$((START_SEED + 2**i))

    if [[ "$cmd" == *"{seed}"* ]]; then
      full_cmd="${cmd//\{seed\}/$seed}"
    else
      full_cmd="$cmd --seed $seed"
    fi

    logfile="$LOGDIR/${name}_seed_${seed}.log"
    echo "  Run $((i+1))/$REPEATS: seed=$seed -> $full_cmd"
    bash -c "$full_cmd" > "$logfile" 2>&1 || echo "Command failed (see $logfile)" >&2

    acc_file="$results_dir/accuracies.txt"
    if [ -f "$acc_file" ]; then
      tail -n 1 "$acc_file" | awk -v n="$name" -v s="$seed" '{print n, s, $0}' >> "$OUTFILE"
    else
      echo "$name seed_$seed EXTRACTION_FAILED" >> "$OUTFILE"
    fi
  done
done

echo "All experiments done. Aggregated results in $OUTFILE. Logs in $LOGDIR."
