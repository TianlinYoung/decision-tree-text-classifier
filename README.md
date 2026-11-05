# Decision Tree Text Classifier

An in-memory Java binary decision-tree classifier by Tianlin Yang, originally
written for UW coursework. It trains from labeled text, predicts labels, and
saves/loads a text representation of its tree.

## Pipeline and design

1. `CsvReader` skips the CSV header and splits each remaining line on commas
   outside quotes. `DataLoader` selects label/content columns and optionally
   shuffles them together.
2. `TextBlock` counts whitespace-separated tokens and uses each token's count
   divided by total token count as its feature value. Case and punctuation are
   preserved; missing tokens and empty text have feature value zero.
3. `Classifier` starts with a leaf for the first training example. When a new
   example reaches a leaf with a different label, it selects the token with the
   largest frequency difference and splits at the midpoint of the two values.
   Recursive calls build and traverse the tree. This is an incremental learner,
   not a globally optimized decision tree; training order can matter.
4. Inference takes the left child for a value below the threshold and the right
   child otherwise, until reaching a label leaf.
5. Save writes a preorder tree: internal nodes use `Feature:` and `Threshold:`
   lines, followed by left/right subtrees; leaves use a label line. The Scanner
   constructor reloads this format.

## Build and test

Requirements: JDK 17 or newer and Maven 3.6.3 or newer. Maven manages JUnit 5 and
uses `src/main/java` and `src/test/java`. From the repository root:

```sh
mvn package
mvn test
```

The package command also runs tests and creates
`target/decision-tree-text-classifier-1.0-SNAPSHOT.jar`. GitHub Actions runs
`mvn test` on push and pull request with Java 17.

## Run a self-contained demo

The files in `examples/` are synthetic smoke-test fixtures, not a real dataset or
measured results. `sample.csv` contains toy labels/text; `demo-tree.txt` is a
handwritten model that splits on the token `apple`.

From the repository root, run the existing interactive client with scripted
input: load the demo tree, classify the CSV, then quit.

```sh
java -cp target/classes Client <<'EOF_INPUT'
2
examples/demo-tree.txt
1
examples/sample.csv
4
EOF_INPUT
```

The output includes `Results: [fruit, other]`. To interact manually:

```sh
java -cp target/classes Client
```

Choose mode 2 and enter `examples/demo-tree.txt` to load the fixture. Choose
operation 1 and enter `examples/sample.csv` to classify it; operation 3 saves the
model, and operation 4 exits. Training mode 1 and accuracy operation 2 use the
`TRAIN_FILE` / `TEST_FILE` constants in `Client.java` (currently
`data/emails/train.csv` and `data/emails/test.csv`). These real data files are not
included; supply authorized files with a header, label in column 0, and text in
column 1 before using those modes.

## Test coverage

JUnit smoke tests train on tiny synthetic text, predict known labels, save/load
through a temporary file and compare predictions and serialized output, parse a
quoted comma and an empty middle field, and load the example CSV through the
full parsing → feature extraction → training → inference pipeline. These tests
check functionality, not real-world classification quality.

## Dataset & results

- Dataset source/license: **TODO(owner)**.
- Dataset size and label distribution: **TODO(owner)**.
- Feature count: **TODO(owner)**.
- Train/test split and evaluation method: **TODO(owner)**.
- Accuracy and other evaluation metrics: **TODO(owner)**.

No real dataset measurements are claimed by the toy examples or smoke tests.

## Current limits

The existing parser handles single-line quoted commas but retains quotes, drops
trailing empty fields, and does not implement full CSV escaping or multiline
records. Text tokenization does not normalize case or punctuation. Conflicting
labels for identical feature vectors cannot be separated by a frequency split;
large input/tree depth can exceed Java's recursion stack. Model labels must not
collide with the reserved `Feature:` format. This hardening change preserves all
production Java files byte-for-byte and does not tune accuracy or change logic.
