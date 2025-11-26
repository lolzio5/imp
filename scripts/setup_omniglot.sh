#!/usr/bin/env bash
set -e

TARGET_DIR="${1:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$TARGET_DIR" ]; then
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
    TARGET_DIR="$REPO_ROOT/omniglot"
fi

echo "Target Omniglot folder: $TARGET_DIR"

ZIP_URL="https://github.com/brendenlake/omniglot/archive/refs/heads/master.zip"
ZIP_LOCAL="/tmp/omniglot-master.zip"
EXTRACT_DIR="/tmp/omniglot_master"

mkdir -p "$TARGET_DIR/images_all"

echo "Downloading Omniglot archive..."
rm -f "$ZIP_LOCAL"
curl -L "$ZIP_URL" -o "$ZIP_LOCAL"

echo "Extracting archive..."
rm -rf "$EXTRACT_DIR"
unzip -q "$ZIP_LOCAL" -d "$EXTRACT_DIR"

ARCHIVE_ROOT="$EXTRACT_DIR/omniglot-master"
if [ ! -d "$ARCHIVE_ROOT" ]; then
    ARCHIVE_ROOT="$EXTRACT_DIR"
fi

PYTHON_DIR="$ARCHIVE_ROOT/python"
if [ ! -d "$PYTHON_DIR" ]; then
    echo "Error: python directory not found in archive"
    exit 1
fi

echo "Extracting Omniglot zip files from python/ folder..."

for ZIP_FILE in "$PYTHON_DIR"/images_*.zip; do
    [ -f "$ZIP_FILE" ] || continue
    echo "Processing $ZIP_FILE"
    TEMP_DIR=$(mktemp -d)
    unzip -q "$ZIP_FILE" -d "$TEMP_DIR"
    # Move all character folders to images_all
    for CHAR_DIR in "$TEMP_DIR"/*/; do
        DEST_DIR="$TARGET_DIR/images_all/$(basename "$CHAR_DIR")"
        echo "Copying folder: $CHAR_DIR -> $DEST_DIR"
        cp -r "$CHAR_DIR" "$DEST_DIR"
    done
    rm -rf "$TEMP_DIR"
done

echo "Cleaning up temporary files..."
rm -f "$ZIP_LOCAL"
rm -rf "$EXTRACT_DIR"

echo "Omniglot prepared at: $TARGET_DIR/images_all"
echo "Done. You can now run run_eval.py with --dataset omniglot"