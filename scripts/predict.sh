# arguments:
#  $1: model directory
#  $2: kepid
#  $3: tce_period
#  $4: tce_time0bk
#  $5: tce_duration (h)

# set pythonpath relative to this location
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
export PYTHONPATH=${SCRIPTPATH}/../astronet

HOME=${SCRIPTPATH}/..
# model and raw lightcurves locations
MODEL_DIR="${HOME}/astronet/astronet/$1"
KEPLER_DATA_DIR="${HOME}/data/kepler"

# generate a prediction for a new TCE.
python astronet/astronet/predict.py \
  --model=AstroCNNModel \
  --config_name=local \
  --model_dir=${MODEL_DIR} \
  --kepler_data_dir=${KEPLER_DATA_DIR} \
  --kepler_id=$2 \ #11442793 \
  --period=$3 \ #14.44912 \
  --t0=$4 \ #2.2 \
  --duration=$5 \ #0.11267 \
  --output_image_file="${HOME}/astronet/current_candidate.png"


