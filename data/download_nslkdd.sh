#!/usr/bin/env bash
# ──────────────────────────────────────────────
# CortiX — NSL-KDD Dataset Downloader
#
# Downloads the NSL-KDD dataset from the UNB/CIC archive.
# This is a cleaned version of the original KDD Cup 1999 dataset,
# with redundant records removed and difficulty levels added.
#
# Usage:
#     bash data/download_nslkdd.sh
#
# Output:
#     data/nslkdd/KDDTrain+.txt
#     data/nslkdd/KDDTest+.txt
# ──────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/nslkdd"

echo "════════════════════════════════════════════════"
echo "  CortiX — NSL-KDD Dataset Downloader"
echo "════════════════════════════════════════════════"

mkdir -p "${OUTPUT_DIR}"

# NSL-KDD is hosted on the UNB CIC datasets page
# Alternative mirrors in case UNB is down
BASE_URL="https://raw.githubusercontent.com/defcom17/NSL_KDD/master"

FILES=(
    "KDDTrain+.txt"
    "KDDTest+.txt"
    "KDDTrain+_20Percent.txt"
    "KDDTest-21.txt"
)

for file in "${FILES[@]}"; do
    output_path="${OUTPUT_DIR}/${file}"
    if [ -f "${output_path}" ]; then
        echo "  ✓ Already exists: ${file}"
        continue
    fi

    echo "  ↓ Downloading: ${file}..."
    if curl -fsSL "${BASE_URL}/${file}" -o "${output_path}"; then
        size=$(du -h "${output_path}" | cut -f1)
        echo "    ✓ Downloaded: ${file} (${size})"
    else
        echo "    ✗ Failed to download: ${file}"
        echo "    Trying alternative mirror..."
        ALT_URL="https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/${file}"
        if curl -fsSL "${ALT_URL}" -o "${output_path}"; then
            size=$(du -h "${output_path}" | cut -f1)
            echo "    ✓ Downloaded from mirror: ${file} (${size})"
        else
            echo "    ✗ Both mirrors failed for ${file}. Please download manually."
        fi
    fi
done

echo ""
echo "════════════════════════════════════════════════"
echo "  Download complete. Files saved to: ${OUTPUT_DIR}"
echo ""
echo "  Next step: python data/prepare_nslkdd.py"
echo "════════════════════════════════════════════════"
