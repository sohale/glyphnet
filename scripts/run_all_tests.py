# run tests + runs all it can, as test + demo (headless)
set -ex

export REPO_ROOT=$(git rev-parse --show-toplevel)
echo "$REPO_ROOT"
cd "$REPO_ROOT"

python glyphnet/mln_topology_tests.py

mkdir -p ./graph

python glyphnet/glyphnet3.py && \
tensorboard --logdir="./graph"

