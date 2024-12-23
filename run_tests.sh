DATA=ag_ckpt_vocab
MELIAD_PATH=meliad_lib/meliad
export PYTHONPATH=$PYTHONPATH:$MELIAD_PATH

python problem_test.py
python geometry_test.py
python graph_utils_test.py
python numericals_test.py
python graph_test.py
python dd_test.py
python ar_test.py
python ddar_test.py
python trace_back_test.py
python alphageometry_test.py
python lm_inference_test.py --meliad_path=$MELIAD_PATH --data_path=$DATA
