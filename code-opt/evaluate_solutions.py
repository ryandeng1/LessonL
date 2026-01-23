import argparse

from client.models import LLM4PP_Problem, LLM4PP_Submission
from client.pareval_client import ParEvalDriver
from client.polybench_client import PolyBenchDriver
from client.chatapi import ChatAPI, MessageHistory
# from vllm import LLM, SamplingParams
from openai import OpenAI
import json
import time
# import hydra
import jsonlines
import os

# from omegaconf import DictConfig, OmegaConf
from agents.utils import *
from strategies.utils import *
from agents.prompts.base_prompts import *
from agents.utils import *

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
        f"Task: Optimize the provided for speed while preserving exact behavior and I/O.\n"
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
        return s
    # Heuristic fallback
    if "#include" in s or "int main" in s or "template<" in s or s.startswith("//"):
        return s
    if s.startswith("fn ") or "use " in s:
        return s
    return text

def eval_inference(input_path: str, output_path: str, benchmark: str):
    benchmark = benchmark.lower().strip()
    if benchmark == "pareval":
        driver = ParEvalDriver(mode="OpenMP", data_path="ParEval-PolyBench-Code-Opt/prompts/pareval_code_opt.json")
    elif benchmark == "polybench":
        driver = ParEvalDriver(mode="OpenMP", data_path="ParEval-PolyBench-Code-Opt/prompts/polybench_code_opt.json")
    else:
        assert False, f"benchmark: {benchmark} not supported"
    
    if input_path.endswith("json"):
        with open(input_path) as f:
            problem_id_to_inference_results = json.load(f)
    else:
        problem_id_to_inference_results = {}
        with jsonlines.open(input_path) as reader:
            for obj in reader:
                pid = obj["submission"]["problem"]["problem_id"]
                submitted_code = obj["submission"]["submitted_code"]
                problem_id_to_inference_results[pid] = submitted_code

    lst_responses = []
    for i, problem in enumerate(driver):
        problem : LLM4PP_Problem

        start = time.time()
        print(f"{i}: {problem.problem_id}")

        text = problem_id_to_inference_results[problem.problem_id]
        optimized_code = extract_code_from_text(text)

        submission = LLM4PP_Submission(problem=problem,
                                    submitted_code=optimized_code)

        try:
            response = driver.submit(submission)
        except Exception as e:
            print(f"skipping problem due to exception: {e}")
            print("--- ParEval driver stdout ---")
            print(response.stdout)

        end = time.time()

        print(f"problem: {problem.problem_id}, compiled: {response.compiled}, correct: {response.correct}, runtime: {response.runtime}, reference: {response.reference_runtime}, time: {end - start}")

        lst_responses.append(response.model_dump())

        # driver.save_one_response_jsonl(output_path, [response.model_dump()], append=True)

    driver.save_one_response_jsonl(output_path, lst_responses, append=False)
    driver.evaluate()

def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ParEval Inference")
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="input file containing generations"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="output file containing stats"
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        type=str,
        help="output file containing stats"
    )

    args = parser.parse_args()
    start = time.time()
    eval_inference(args.input, args.output, args.benchmark)
    end = time.time()
    print(f"vanilla inference: {end - start} seconds")

if __name__ == "__main__":
    main()

