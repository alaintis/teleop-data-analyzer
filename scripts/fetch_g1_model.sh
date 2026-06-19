#!/usr/bin/env bash
# Fetch the Unitree G1 (with Dex3 hands) MuJoCo model + meshes from
# mujoco_menagerie into models/unitree_g1/.
#
# The 43-DOF `g1_with_hands.xml` has the exact same joint names as the
# teleop dataset's `observation.state` / `action.wbc`, so the action viewer
# maps joints by name with no hand-tuned index table.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/models/unitree_g1"

if [[ -f "$DEST/g1_with_hands.xml" ]]; then
    echo "G1 model already present at $DEST"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Sparse-cloning unitree_g1 from mujoco_menagerie ..."
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git "$TMP/menagerie"
git -C "$TMP/menagerie" sparse-checkout set unitree_g1

mkdir -p "$REPO_ROOT/models"
cp -r "$TMP/menagerie/unitree_g1" "$DEST"
echo "G1 model installed at $DEST"
