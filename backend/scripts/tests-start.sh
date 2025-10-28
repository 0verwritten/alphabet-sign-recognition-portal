#! /usr/bin/env bash
set -e
set -x

echo "Starting tests with simple mock API setup."

bash scripts/test.sh "$@"
