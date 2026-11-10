#!/usr/bin/env bash
# Empaqueta el checkpoint sin crear una segunda copia grande en CloudShell.
set -euo pipefail
: "${MODEL_BUCKET:?Ejecuta primero: source scripts/bootstrap_aws.sh}"
MODEL_VERSION="${MODEL_VERSION:-v2}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="$ROOT/model/resunet_best.pth"
[[ -f "$CHECKPOINT" ]] || { echo "Falta $CHECKPOINT" >&2; exit 1; }
[[ "$(stat -c%s "$CHECKPOINT")" -gt 100000000 ]] || { echo "El checkpoint parece un puntero de Git LFS" >&2; exit 1; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/code"
cp "$ROOT/src/model.py" "$ROOT/src/inference.py" "$ROOT/src/requirements.txt" "$TMP/code/"
DEST="s3://$MODEL_BUCKET/deployment/resunet/$MODEL_VERSION/model.tar.gz"
# El tar se genera en stdout y se transmite a S3 sin guardarlo en CloudShell.
tar -czf - -C "$ROOT/model" resunet_best.pth -C "$TMP" code | aws s3 cp - "$DEST" --region "$AWS_REGION" --expected-size 500000000
aws s3 ls "$DEST" --region "$AWS_REGION"
echo "Modelo cargado en $DEST"
