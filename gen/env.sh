SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/venv/bin/activate"

export PATH="$SCRIPT_DIR/STL-2009/src/sv56:$PATH"
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR"
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR/selec_anon/compute_anon_spk_vector/flow_matching"
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR/selec_anon/compute_anon_spk_vector/WGAN"

# REPLACE THE PATH BELOW WITH YOUR OWN CONFIGURATION
export PYTHONPATH="$PYTHONPATH:/scratch/work/nguyent166/fairseq"

echo "Emotion Compensation environment loaded"