#!/usr/bin/env bash
set -euo pipefail

python -m training.train --config configs/vit_b16_cifar10.yaml
