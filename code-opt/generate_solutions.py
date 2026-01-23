import argparse
from client.models import LLM4PP_Problem, LLM4PP_Submission
from client.pareval_client import ParEvalDriver
from client.polybench_client import PolyBenchDriver
from client.chatapi import ChatAPI, MessageHistory
# from vllm import LLM, SamplingParams
from openai import OpenAI
import json
# import hydra
import jsonlines
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# from omegaconf import DictConfig, OmegaConf
from agents.utils import *
from strategies.utils import *
from agents.prompts.base_prompts import *
from agents.utils import *

from transformers import AutoTokenizer, AutoModel

from typing import Optional, List

def build_prompt(original_code: str, language: str) -> str:
    """Build an optimization prompt using a simple on-disk template with safe token replacement.

    The template must contain tokens {{LANGUAGE_NAME}}, {{SOURCE_FILENAME}}, {{BASELINE_SECONDS}},
    and markers :::HEADER::: and :::CODE::: which will be replaced verbatim.
    Falls back to an inline template if the file is missing.
    """

    # Fallback inline prompt (kept simple by design)
    env_lines = (
        # "Environment: modern x86-64; 8 physical cores, 2-way hyperthreading (16 threads).\n"
        "Environment: modern x86-64; 8 threads\n"
        "Compiler: gcc -O2 with OpenMP support (-fopenmp).\n"
        # f"Baseline runtime on benchmark input: ~{baseline_seconds:.2f}s.\n"
    )
    return (
        f"You are an expert {language} performance engineer.\n\n"
        f"{env_lines}\n"
        f"Task: Optimize the provided code for speed while preserving the exact behavior as the original code.\n"
        f"- Do not change the function signature expected by the harness.\n"
        f"- Provide a full replacement for this code.\n"
        f"- Return only the code in a single fenced block.\n\n"
        f"--- current source ---\n{original_code}\n"
    )

def extract_code_from_text(text: str) -> Optional[str]:
    fence_pattern = re.compile(r"```(cpp|c\+\+|c|rust)?\n([\s\S]*?)```", re.IGNORECASE)
    matches = list(fence_pattern.finditer(text or ""))
    if matches:
        chosen = max(matches, key=lambda m: len(m.group(2) or ""))
        return (chosen.group(2) or "").strip()
    s = (text or "").strip()
    if not s:
        return None
    # Heuristic fallback
    if "#include" in s or "int main" in s or "template<" in s or s.startswith("//"):
        return s
    if s.startswith("fn ") or "use " in s:
        return s
    return None

def do_inference(model: str, output_path: str, benchmark: str):
    benchmark = benchmark.lower().strip()
    if benchmark == "pareval":
        driver = ParEvalDriver(mode="OpenMP")
    elif benchmark == "polybench":
        driver = ParEvalDriver(mode="OpenMP", data_path="ParEval-PolyBench-Code-Opt/prompts/polybench_code_opt.json")
    else:
        assert False, f"benchmark: {benchmark} not supported"

    print(f"start to evaluate {model}, output: {output_path}")

    problems = list(driver)
    total = len(problems)
    print(f"loaded {total} problems")

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="ryan123",
    )

    def infer_one(problem: LLM4PP_Problem):
        prompt = build_prompt(problem.source_code, "C++")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="low",
            temperature=1.0,
            top_p=1.0,
        )
        text = completion.choices[0].message.content
        return problem.problem_id, text

    problem_id_to_inference_results = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(infer_one, problem): problem
            for problem in problems
        }
        for future in as_completed(futures):
            problem_id, text = future.result()
            problem_id_to_inference_results[problem_id] = text

    with open(output_path, mode="w") as f:
        json.dump(problem_id_to_inference_results, f, indent=2)

"""
def run(cfg: DictConfig) -> None:
    benchmark = cfg.benchmark
    mode = cfg.mode

    temperature = cfg.temperature
    # Use benchmark drivers passed from main.py
    driver = cfg.driver
    # evaldriver = cfg.evaldriver
    trial = cfg.trial

    if mode != "serial": # assume mode is a parallel package can be integrated in c++ only
        additional_package = f"You should use {mode} to parallelize the code."
    else: #serial
        additional_package = ""

    #model = "gpt-4o"
    #model = "o3"
    #model = "gpt-4o-mini"

    model = cfg.model

    if model == "o3":
        reason = True
    else:
        reason = False

    #save_destination = f"./logs/{model}-{benchmark}-{mode}-lessons.jsonl"
    savename = f"openai_{model}_{benchmark}_{model}_{mode}_trial_{trial}.jsonl"
    os.makedirs("evaluator_results", exist_ok=True)
    evaluator_save_path = f"evaluator_results/{savename}.jsonl"
    print(f"OpenAI {model} Evaluator Results save path: ", evaluator_save_path)

    # chatAPI = ChatAPI(temperature=temperature)
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    print("start to evaluate ", model, trial)

    # model = "/workspace/train_sft/llama_code_contests_sft/model"
    # model = "/workspace/train_sft/llama_code_contests_sft_tfk/model"
    # model = "/workspace/train_sft/qwen_coder_code_contests_sft/model"
    model = "/workspace/train_sft/qwen_coder_code_contests_sft_tfk/model"

    for i, problem in enumerate_driver_resume(driver, evaluator_save_path):
        problem : LLM4PP_Problem

        print(problem.problem_id)

        # messages = MessageHistory()
        # messages.add_message("system", optimizer_prompt)
        # prompt = generate_code_opt_prompt_code(src_code=problem.source_code, additional_package=additional_package)
        prompt = build_prompt(problem.source_code, "C++")

        # messages.add_message("user", prompt)
        # response = chatAPI.get_response(model, messages, json_format=False, reason=reason)
        # optimized_code = clean_output(response)

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        print(completion)
        assert False

        text = completion.choices[0].message.content
        optimized_code = extract_code_from_text(text)

        submission = LLM4PP_Submission(problem=problem,
                                    submitted_code=optimized_code)

        try:
            response = driver.submit(submission)
        except Exception as e:
            print(f"skipping problem due to exception: {e}")
            print("--- ParEval driver stdout ---")
            print(response.stdout)

        print(f"problem: {problem.problem_id}, compiled: {response.compiled}, correct: {response.correct}, runtime: {response.runtime}, reference: {response.reference_runtime}")
        # print("--- response code ---")
        # print(response.submission.submitted_code)
        # print("--- stdout ---")
        # print(response.stdout)

        driver.save_one_response_jsonl(evaluator_save_path, [response.model_dump()], append=True)
    driver.evaluate()
    # print(chatAPI.get_cost())
    # out_tokens, input_tokens, total_tokens = chatAPI.get_usage()
    # print("Output tokens: ", out_tokens[model])
    # print("Input tokens: ", input_tokens[model])
    # print("total tokens: ", input_tokens[model] + out_tokens[model])
"""

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize LLM code optimization speedups.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination file for the generated plot (ignored if --show is set).",
    )
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Destination file for the generated plot (ignored if --show is set).",
    )
    args = parser.parse_args()
    do_inference(args.model, str(args.output), args.benchmark)

if __name__ == "__main__":
    main()
