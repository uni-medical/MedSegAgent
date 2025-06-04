import argparse
import json
import logging
import colorlog
import os
import os.path as osp
from datetime import datetime

from utils.chat_utils import chat_with_llm
from utils.metric import dataset_hit_rate, all_dataset_found_rate, any_dataset_found_rate
from utils.reply_utils import *

import time


def score_the_answer(gt_answer, llm_answer):
    return {
        "hit_rate": dataset_hit_rate(gt_answer, llm_answer),
        "strict_hit_rate": all_dataset_found_rate(gt_answer, llm_answer),
        "soft_hit_rate": any_dataset_found_rate(gt_answer, llm_answer),
    }


def run_tests_from_jsonl(file_path,
                         llm_func,
                         result_file,
                         resume=False,
                         state_filename='task_state.json'):
    final_score, total_score = 0, 0
    final_strict_score, final_soft_score = 0, 0
    start_line_id = 0
    evaluation_results = dict()

    # try to resume state
    if resume:
        logging.info("[Results] resume from state")
        state = load_state(state_filename=state_filename)
        if state:
            start_line_id = state['current_line']
            final_score = state['final_score']
            final_strict_score = state['final_strict_score']
            final_soft_score = state['final_soft_score']
            total_score = state['total_score']
            evaluation_results = state['evaluation_results']

    with open(file_path, 'r') as file:
        for line_id, line in enumerate(file):
            line_id += 1
            if line_id <= start_line_id:
                continue

            test_case = json.loads(line)
            query = test_case['query']
            gt_answer = extract_ground_truth(test_case)

            res_dict = llm_func(query)
            evaluation_results[line_id] = res_dict
            try:
                res = res_dict['reply']
            except KeyError:
                res = res_dict
            try:
                chat_history = res_dict['chat_history']
            except KeyError:
                chat_history = ""
            formatted_history = '\n'.join([
                chat['role'] + '(' + chat.get('name', "unamed") + "): " + chat['content']
                for chat in chat_history
            ])
            logging.info(f"[{line_id}] Chat History:\n{formatted_history}")
            # llm_answer = extract_llm_answer(res)
            llm_answer = res
            case_score = score_the_answer(gt_answer, llm_answer)

            hit_rate = case_score['hit_rate']
            strict_hit_rate = case_score['strict_hit_rate']
            soft_hit_rate = case_score['soft_hit_rate']

            final_score += hit_rate
            final_strict_score += strict_hit_rate
            final_soft_score += soft_hit_rate
            total_score += 1.0

            if hit_rate == 1.0:  # answer correctly
                logging.info(
                    f"[{line_id}] Correct! (hit_rate: {hit_rate}, final: {final_score})\nExpected answer:\n{gt_answer}\nLLM:\n{llm_answer}"
                )
                print(
                    f"[{line_id}] Correct! (hit_rate: {hit_rate}, final: {final_score})\nExpected answer:\n{gt_answer}\nLLM:\n{llm_answer}"
                )
            else:
                logging.error(
                    f"[{line_id}] Wrong! (hit_rate: {hit_rate}, final: {final_score})\nExpected answer:\n{gt_answer}\nLLM:\n{llm_answer}"
                )
                print(
                    f"[{line_id}] Wrong! (hit_rate: {hit_rate}, final: {final_score})\nExpected answer:\n{gt_answer}\nLLM:\n{llm_answer}"
                )

            # time.sleep(5)

            # save the state
            state = {
                'current_line': line_id,
                'final_score': final_score,
                'final_strict_score': final_strict_score,
                'final_soft_score': final_soft_score,
                'total_score': total_score,
                'evaluation_results': evaluation_results,
            }
            save_state(state, state_filename=state_filename)

    logging.info(f"[Summary] [Datasets Hit Rate] final score {final_score:.2f} / {total_score}")
    logging.info(
        f"[Summary] [All Dataset Found] final score {final_strict_score:.2f} / {total_score}")
    logging.info(
        f"[Summary] [Any Dataset Found] final score {final_soft_score:.2f} / {total_score}")

    if not args.skip_results_saving:
        logging.info(f"[Results] result saved to {result_file}")
        with open(result_file, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
    else:
        logging.info("[Results] skip results saving")

    if os.path.exists(state_filename):
        logging.info("[Results] clean state")
        os.remove(state_filename)

    return {
        'final_score': final_score,
        'final_strict_score': final_strict_score,
        'final_soft_score': final_soft_score,
        'total_score': total_score,
    }


def recompute_metrics_from_file(question_file, result_file):
    final_score, total_score = 0, 0
    final_strict_score, final_soft_score = 0, 0
    evaluation_results = json.load(open(result_file, "r"))

    question_numbers = 0
    with open(question_file, 'r') as file:
        for line_id, line in enumerate(file):
            question_numbers += 1

    if (len(evaluation_results) != question_numbers):
        logging.error("")
        return -1

    with open(question_file, 'r') as file:
        for line_id, line in enumerate(file):
            line_id += 1

            test_case = json.loads(line)
            gt_answer = extract_ground_truth(test_case)

            res_dict = evaluation_results[str(line_id)]
            res = res_dict['reply']
            chat_history = res_dict['chat_history']
            formatted_history = '\n'.join(
                [chat['role'] + ": " + chat['content'] for chat in chat_history])
            logging.info(f"[{line_id}] Chat History:\n{formatted_history}")
            llm_answer = extract_llm_answer(res)

            case_score = score_the_answer(gt_answer, llm_answer)

            hit_rate = case_score['hit_rate']
            strict_hit_rate = case_score['strict_hit_rate']
            soft_hit_rate = case_score['soft_hit_rate']

            final_score += hit_rate
            final_strict_score += strict_hit_rate
            final_soft_score += soft_hit_rate
            total_score += 1.0

            if hit_rate == 1.0:  # answer correctly
                logging.info(
                    f"[{line_id}] Correct! (hit_rate: {hit_rate}, final: {final_score})\nExpected answer:\n{gt_answer}\nLLM:\n{llm_answer}"
                )
            else:
                logging.error(
                    f"[{line_id}] Wrong! (hit_rate: {hit_rate}, final: {final_score})\nExpected answer:\n{gt_answer}\nLLM:\n{llm_answer}"
                )

    logging.warning(f"[Summary] [Dataset  Hit Rate] final score {final_score:.2f} / {total_score}")
    logging.warning(
        f"[Summary] [Dataset All Found] final score {final_strict_score:.2f} / {total_score}")
    logging.warning(
        f"[Summary] [Dataset Any Found] final score {final_soft_score:.2f} / {total_score}")

    return {
        'final_score': final_score,
        'final_strict_score': final_strict_score,
        'final_soft_score': final_soft_score,
        'total_score': total_score,
    }


def get_timestamped_log_file(pattern, file_path, tag):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = osp.basename(file_path).replace('.jsonl', '')
    os.makedirs("results", exist_ok=True)
    return osp.join("results", f'{pattern}_{base_name}_{tag}_{timestamp}.log')


def main(test_file_path, test_pattern, model, tag, resume, log_to_file):
    tag = osp.basename(model)
    log_outfile = get_timestamped_log_file(test_pattern, test_file_path, tag)
    if (args.recompute_metrics_from_file):
        log_outfile = args.recompute_metrics_from_file.replace(".json", "_recompute.log")
    if resume:
        state = load_state()
        if state:
            log_outfile = log_outfile.replace(".log", "_resume.log")

    if log_to_file:
        print(f"saving file to {log_outfile}")
        logging.basicConfig(filename=log_outfile,
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        handler = colorlog.StreamHandler()
        log_colors = {
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(levelname)s - %(message)s', log_colors=log_colors)
        handler.setFormatter(formatter)
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.handlers.clear()  # Clear existing handlers
        logger.addHandler(handler)

    if (args.recompute_metrics_from_file):
        pass
        final_score_dict = recompute_metrics_from_file(
            question_file=test_file_path,
            result_file=args.recompute_metrics_from_file,
        )
        print(
            f"recomputed  hit rate: {final_score_dict['final_score']:.2f} / {final_score_dict['total_score']}"
        )
        print(
            f"recomputed all found: {final_score_dict['final_strict_score']:.2f} / {final_score_dict['total_score']}"
        )
        print(
            f"recomputed any found: {final_score_dict['final_soft_score']:.2f} / {final_score_dict['total_score']}"
        )
    else:
        final_score_dict = run_tests_from_jsonl(
            test_file_path,
            llm_func=lambda x: chat_with_llm(test_pattern, x, model=model),
            result_file=log_outfile.replace(".log", ".json"),
            resume=resume)
        print(
            f"hit rate: {final_score_dict['final_score']:.2f} / {final_score_dict['total_score']}")
        print(
            f"all found: {final_score_dict['final_strict_score']:.2f} / {final_score_dict['total_score']}"
        )
        print(
            f"any found: {final_score_dict['final_soft_score']:.2f} / {final_score_dict['total_score']}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--recompute_metrics_from_file',
                        type=str,
                        default=None,
                        help='Recompute the metrics without query')
    parser.add_argument('--test_file_path', type=str, required=True, help='Path to the test file')
    parser.add_argument('--test_pattern',
                        type=str,
                        required=True,
                        default="Single_Turn",
                        help='Test pattern')
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        default='gpt-4o-mini-2024-07-18',
                        help='Model name')
    parser.add_argument('--tag', type=str, default=None, required=False, help='Tag name')
    parser.add_argument('--resume', action='store_true', help='Resume flag')
    parser.add_argument('--log_to_file', action='store_true', help='Log to file flag')
    parser.add_argument('--skip_results_saving', action='store_true', help='Log to file flag')
    args = parser.parse_args()

    main(args.test_file_path, args.test_pattern, args.model, args.tag, args.resume,
         args.log_to_file)
