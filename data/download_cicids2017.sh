#!/usr/bin/env bash
# ──────────────────────────────────────────────
# CortiX — CICIDS2017 Dataset Download Script
#
# Downloads the CICIDS2017 Intrusion Detection dataset
# from the Canadian Institute for Cybersecurity (CIC).
#
# Usage:
#   bash data/download_cicids2017.sh
#
# The dataset will be placed in data/cicids2017/
# ──────────────────────────────────────────────

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)/cicids2017"
mkdir -p "$DATA_DIR"

echo "═══════════════════════════════════════════════"
echo " CortiX — CICIDS2017 Dataset Download"
echo "═══════════════════════════════════════════════"

# The CICIDS2017 dataset files
FILES=(
    "Monday-WorkingHours.pcap_ISCX.csv"
    "Tuesday-WorkingHours.pcap_ISCX.csv"
    "Wednesday-workingHours.pcap_ISCX.csv"
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    "Friday-WorkingHours-Morning.pcap_ISCX.csv"
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

BASE_URL="https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/CSVs"

echo ""
echo "Note: The CICIDS2017 dataset is hosted by the University of New Brunswick."
echo "Total download size: approximately 1.3 GB (CSV files)."
echo ""
echo "If the download URL is unavailable, you can download the dataset from:"
echo "  https://www.unb.ca/cic/datasets/ids-2017.html"
echo ""
echo "Alternatively, a Kaggle mirror is available at:"
echo "  https://www.kaggle.com/datasets/cicdataset/cicids2017"
echo ""

for file in "${FILES[@]}"; do
    filepath="$DATA_DIR/$file"
    if [ -f "$filepath" ]; then
        echo "  ✓ Already exists: $file"
    else
        echo "  ↓ Downloading: $file"
        if command -v wget &> /dev/null; then
            wget -q --show-progress -O "$filepath" "$BASE_URL/$file" || {
                echo "    ✗ wget failed for $file. Please download manually."
            }
        elif command -v curl &> /dev/null; then
            curl -L -o "$filepath" "$BASE_URL/$file" --progress-bar || {
                echo "    ✗ curl failed for $file. Please download manually."
            }
        else
            echo "    ✗ Neither wget nor curl found. Install one and retry."
            exit 1
        fi
    fi
done

echo ""
echo "═══════════════════════════════════════════════"
echo " Download complete! Files saved to:"
echo "   $DATA_DIR/"
echo ""
echo " Next steps:"
echo "   1. Merge CSVs:  python data/merge_cicids2017.py"
echo "   2. Train model: python -m cortix.classifier.train --dataset data/cicids2017_merged.csv"
echo "═══════════════════════════════════════════════"
