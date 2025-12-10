## Table 1: Multi-modal Clustering on Omniglot

### Paper Results
Table 1. Multi-modal clustering and learning cluster variances on fully-supervised 10-way, 10-shot Omniglot alphabet recognition.

| Method | Omniglot Accuracy |
|--------|----------------|
| PROTOTYPES | 65.2 ± 0.6 | 66.1 ± 0.6 |
| PROTOTYPES | 65.2 ± 0.6 | 67.2 ± 0.5 |
| **IMP** | **92.0 ± 0.1** | **68.1 ± 0.8** |

### My Results
| Seed | Accuracy | Std Dev |
|------|----------|---------|
| 1 | 47.25% | 0.39% |
| 2 | 48.13% | 0.40% |
| 4 | 47.33% | 0.38% |

**Average: 47.57% ± 0.39%** vs. **Paper: 92.0% ± 0.1%**

### Command I ran
```bash
python run_eval.py --dataset omniglot --model imp --label-ratio 1.0 \
  --nclasses-train 10 --nclasses-episode 10 --nclasses-eval 10 \
  --nshot 1 --num-test 5 --super-classes \
  --nsuperclassestrain 10 --nsuperclasseseval 10 \
  --disable-distractor --num-unlabel 0 --num-unlabel-test 0 \
  --use-test --results ./results/table1/alphabet_10way_10shot
```

This tests 10-way, 10-shot alphabet recognition where:
- `--nclasses-train 10`: 10 training classes (characters)
- `--super-classes`: Use alphabet-level superclasses
- `--nsuperclassestrain 10`: 10 alphabets for training
- `--nshot 1`: 1 support image per character
- `--num-test 5`: 5 query images per character
- `--use-test`: Evaluate on test set

---

## Table 3: Alphabet and Character Recognition

### Paper Results
Table 3. Alphabet and character recognition accuracy on Omniglot.

| Training | Testing | Prototypes | IMP | Neighbors |
|----------|---------|------------|-----|-----------|
| ALPHABET | ALPHABET (10-WAY 10-SHOT) | 65.6±0.4 | **92.0±0.1** | 92.4±0.2 |
| ALPHABET | CHARS (20-WAY 1-SHOT) | 82.1±0.4 | **95.4±0.2** | 95.4±0.2 |
| CHARS | CHARS (20-WAY 1-SHOT) | 94.9±0.2 | **95.1±0.1** | 95.1±0.1 |

### My Results

#### Row 1: Train on ALPHABET, test on ALPHABET (10-way 10-shot)
Same as Table 1 - 47.57% vs 92.0%

#### Row 2: Train on ALPHABET, test on CHARS (20-way 1-shot)
| Seed | Accuracy | Std Dev |
|------|----------|---------|
| 1 | 56.49% | 0.58% |
| 2 | 55.06% | 0.60% |
| 4 | 56.78% | 0.61% |

**Average: 56.11% ± 0.60%** vs. **Paper: 95.4% ± 0.2%**

#### Row 2: Train on alphabets, test on characters

### Command I ran
**Training:**
```bash
python run_eval.py --dataset omniglot --model imp --mode-ratio 1.0 --label-ratio 1.0 \
  --nclasses-train 10 --nclasses-episode 10 --nclasses-eval 10 \
  --nshot 1 --num-test 5 --super-classes \
  --nsuperclassestrain 10 --nsuperclasseseval 10 \
  --disable-distractor --num-unlabel 0 --num-unlabel-test 0 \
  --results ./results/table3/alphabet_chars_train
```

**Testing:**
```bash
python run_eval.py --dataset omniglot --model imp --mode-ratio 1.0 --label-ratio 1.0 \
  --nclasses-train 20 --nclasses-episode 20 --nclasses-eval 20 \
  --nshot 1 --num-test 5 --disable-distractor \
  --use-test --num-unlabel-test 0 --eval \
  --pretrain ./results/table3/alphabet_chars_train/best \
  --results ./results/table3/alphabet_chars_test
```

**Explanation:**
- **Training phase:** Train on 10 alphabets (superclasses) with 10 characters per alphabet
- **Testing phase:** Test on individual characters (20-way 1-shot), without superclass structure

#### Row 3: Train on CHARS, tets on CHARS (20-way 1-shot)
| Seed | Accuracy | Std Dev |
|------|----------|---------|
| 1 | 94.61% | 0.30% |
| 2 | 94.38% | 0.30% |
| 4 | 94.68% | 0.27% |

**Average: 94.56% ± 0.29%** vs. **Paper: 95.1% ± 0.1%**

##### Command I ran

#### Row 3: Train and test on characters
```bash
python run_eval.py --dataset omniglot --model imp --mode-ratio 1.0 --label-ratio 1.0 \
  --nclasses-train 20 --nclasses-episode 20 --nclasses-eval 20 \
  --nshot 1 --num-test 5 --disable-distractor \
  --num-unlabel 0 --num-unlabel-test 0 --use-test \
  --results ./results/table3/chars_chars_train
```

**Explanation:**
- Train and test directly on characters (no alphabet superclass structure)
- 20-way 1-shot: 20 character classes, 1 support image per class
- This is the standard few-shot learning setup without multi-modal clustering

---

## Table 4: Generalisation to Held-out Characters

### Paper Results
**Table 4.** Generalisation to held-out characters on 10-way, 5-shot Omniglot alphabet recognition. 40% of characters kept for training, 60% held out for testing.

| Method | Training Modes | Testing Modes | Both Modes |
|--------|----------------|---------------|------------|
| IMP (OURS) | 99.0 ± 0.1 | 94.9 ± 0.2 | **96.6 ± 0.2** |
| PROTOTYPES | 92.4 ± 0.3 | 77.7 ± 0.4 | 82.9 ± 0.4 |

### My Results
| Seed | Training Accuracy | Testing Accuracy |
|------|-------------------|------------------|
| 1 | 66.44% ± 0.54% | 64.71% ± 0.41% |
| 2 | 66.45% ± 0.54% | 64.73% ± 0.40% |
| 4 | 66.45% ± 0.54% | 64.88% ± 0.42% |

**Average on training: 63.67% ± 0.53%** vs. **Paper: 96.6% ± 0.2%**

**Average on testing: 63.11% ± 0.40%** vs. **Paper: 96.6% ± 0.2%**

### Commands I ran

**Training:**
```bash
python run_eval.py --dataset omniglot --model imp --label-ratio 0.4 \
  --nclasses-train 5 --nclasses-episode 5 --nclasses-eval 5 \
  --nshot 1 --num-test 5 --super-classes \
  --nsuperclassestrain 10 --nsuperclasseseval 10 \
  --disable-distractor --num-unlabel 0 --num-unlabel-test 0 \
  --results ./results/table4/train
```

**Testing:**
```bash
python run_eval.py --dataset omniglot --model imp --label-ratio 0.6  \
  --nclasses-train 5 --nclasses-episode 5 --nclasses-eval 5 \
  --nshot 1 --num-test 5 --super-classes \
  --nsuperclassestrain 10 --nsuperclasseseval 10 \
  --disable-distractor --use-test --num-unlabel-test 0 --eval \
  --pretrain ./results/table4/train/best \
  --results ./results/table4/test
```

**Explanation:**
- `--label-ratio 0.4`: Critical parameter - only 40% of characters within each alphabet are used for training
- The remaining 60% of characters are held out and only appear during testing
