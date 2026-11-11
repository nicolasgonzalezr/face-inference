# Type: ./Start-FaceInference.sh

#!/usr/bin/env bash
set -euo pipefail

# 1) go to the folder this script lives in
cd "$(dirname "$0")"

# 2) activate your Python 3.11 venv
# (change 'venv' if your env has a different name)
source ./venv/bin/activate

# 3) OPTIONAL: rein in threads if you want the Mac to stay snappy
# export OPENCV_THREADS=1
# export TORCH_CPU_THREADS=2
# export OMP_NUM_THREADS=2
# export MKL_NUM_THREADS=2
# export NUMEXPR_NUM_THREADS=2

# 4) run the server on localhost:8001
exec uvicorn main:app --host 127.0.0.1 --port 8001 --workers 1 --no-access-log