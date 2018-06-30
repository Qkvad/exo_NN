# arguments:
#  $1: model directory
#  $2: tfrecord directory

# set pythonpath relative to this location
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
export PYTHONPATH=${SCRIPTPATH}/../astronet

HOME=${SCRIPTPATH}/..
# model and tfrecord files locations.
MODEL_DIR=${HOME}/astronet/astronet/$1
TFRECORD_DIR=${MODEL_DIR}/../$2

# run the evaluation script.
python astronet/astronet/evaluate.py \
  --model=AstroCNNModel \
  --config_name=local \
  --eval_files=${TFRECORD_DIR}/test* \
  --model_dir=${MODEL_DIR}


