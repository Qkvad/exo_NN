# arguments:
#  $1: tfrecord directory
#  $2: model directory
#  $3: train steps

# set pythonpath relative to this location
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
export PYTHONPATH=${SCRIPTPATH}/../astronet

HOME=${SCRIPTPATH}/..
# directory to save model checkpoints into.
MODEL_DIR=${HOME}/astronet/astronet/$2
TFRECORD_DIR=${MODEL_DIR}/../$1

# run the training script.
python astronet/astronet/train.py \
  --model=AstroCNNModel \
  --config_name=local \
  --train_files=${TFRECORD_DIR}/train* \
  --eval_files=${TFRECORD_DIR}/val* \
  --model_dir=${MODEL_DIR} \
  --train_steps=$3


