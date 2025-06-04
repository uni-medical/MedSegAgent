python ./evaluate.py --test_file_path "model_selection_bench.jsonl" --test_pattern "ReAct" --model "Qwen/Qwen2.5-7B-Instruct" --log_to_file
python ./evaluate.py --test_file_path "model_selection_bench.jsonl" --test_pattern "C2F" --model "Qwen/Qwen2.5-7B-Instruct" --log_to_file
python ./evaluate.py --test_file_path "model_selection_bench.jsonl" --test_pattern "ReAct" --model "meta-llama/Meta-Llama-3.1-8B-Instruct" --log_to_file
python ./evaluate.py --test_file_path "model_selection_bench.jsonl" --test_pattern "C2F" --model "meta-llama/Meta-Llama-3.1-8B-Instruct" --log_to_file