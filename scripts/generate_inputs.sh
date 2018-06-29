# arguments:
#  $1: csv file name
#  $2: tfrecord directory
#  $3: number of processes

# set pythonpath relative to this location
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
export PYTHONPATH=${SCRIPTPATH}/../astronet

HOME=${SCRIPTPATH}/..
# tce table, kepler data and tfrecord desired location.
TCE_CSV_FILE=${HOME}/data/csv/$1.csv
TFRECORD_DIR=${HOME}/astronet/astronet/$2
KEPLER_DATA_DIR=${HOME}/data/kepler/

# Preprocess light curves into sharded TFRecord files using $3 worker processes.
python ../astronet/astronet/data/generate_input_records.py \
  --input_tce_csv_file=${TCE_CSV_FILE} \
  --kepler_data_dir=${KEPLER_DATA_DIR} \
  --output_dir=${TFRECORD_DIR} \
  --num_worker_processes=$3
