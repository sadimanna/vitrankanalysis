#!/usr/bin/env bash
set -euo pipefail

python -m training.train --config configs/vit_s16_cifar10.yaml
