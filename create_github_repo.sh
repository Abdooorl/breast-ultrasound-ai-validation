#!/usr/bin/env bash
set -euo pipefail

OWNER="Abdooorl"
REPO="breast-ultrasound-ai-validation"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required for automatic remote creation." >&2
  echo "Install on macOS with: brew install gh" >&2
  exit 1
fi

gh auth status || gh auth login

git init
git branch -M main
git add .
git commit -m "chore: initialize breast ultrasound AI validation project" || true

gh repo create "$OWNER/$REPO" \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --description "External validation and uncertainty-aware evaluation of a pretrained Vision Transformer for breast ultrasound classification."

echo "Repository created: https://github.com/$OWNER/$REPO"
