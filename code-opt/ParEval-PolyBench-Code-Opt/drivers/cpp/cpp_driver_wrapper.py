""" Wrapper for calling c++ drivers
    author: Daniel Nichols
    date: October 2023
"""
# std imports
import copy
import logging
import os
from os import PathLike, environ
import shlex
import subprocess
import sys
import tempfile
from typing import List

# local imports
sys.path.append("..")
from drivers.driver_wrapper import DriverWrapper, BuildOutput, RunOutput, GeneratedTextResult
from util import run_command

import time
import re

""" Map parallelism models to driver files """
DRIVER_MAP = {
    "serial": "serial-driver.o",
    "omp": "omp-driver.o",
    "mpi": "mpi-driver.o",
    "mpi+omp": "mpi-omp-driver.o",
    "kokkos": "kokkos-driver.o",
    "cuda": "cuda-driver.o",
    "hip": "hip-driver.o",
}

""" Compiler settings """
COMPILER_SETTINGS = {
    "serial": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3 -g -march=native "},
    "omp": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3 -fopenmp -g -march=native "},
    # "serial": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3 -g "},
    # "omp": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3 -fopenmp -g "},
    "mpi": {"CXX": "mpicxx", "CXXFLAGS": "-std=c++17 -O3"},
    "mpi+omp": {"CXX": "mpicxx", "CXXFLAGS": "-std=c++17 -O3 -fopenmp"},
    "kokkos": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3 -fopenmp -I../tpl/kokkos/build/include ../tpl/kokkos/build/lib64/libkokkoscore.a ../tpl/kokkos/build/lib64/libkokkoscontainers.a ../tpl/kokkos/build/lib64/libkokkossimd.a"},
    "cuda": {"CXX": "nvcc", "CXXFLAGS": "-std=c++17 --generate-code arch=compute_80,code=sm_80 -O3 -Xcompiler \"-std=c++17 -O3\""},
    "hip": {"CXX": "hipcc", "CXXFLAGS": "-std=c++17 -O3 -Xcompiler \"-std=c++17\" -Xcompiler \"-O3\" -Wno-unused-result"}
}

"""
def wrap_with_namespace_gpt(code: str, namespace="submission") -> str:
    lines = code.splitlines(keepends=True)  # Keep line endings intact

    in_struct = False
    struct_indent = None
    output = []
    namespace_code = []

    for line in lines:
        stripped = line.lstrip()

        # Exclude headers, macros, and using statements from the namespace
        if stripped.startswith(("#include", "#define", "using ", "#pragma once", "#if", "#endif")):
            output.append(line)
            continue

        # Detect struct start (avoid struct keywords in comments)
        if re.match(r'^\s*struct\s+\w+', stripped) and not stripped.startswith("//"):
            in_struct = True
            struct_indent = len(line) - len(stripped)
            output.append(line)
            continue

        # Detect struct end (based on indentation)
        if in_struct and struct_indent is not None:
            if len(line) - len(line.lstrip()) <= struct_indent and stripped != "":
                in_struct = False
            output.append(line)
            continue

        # Everything else goes inside the namespace
        namespace_code.append(line)

    # Wrap non-struct code in the namespace
    if namespace_code:
        output.append(f"namespace {namespace} {{\n")
        output.extend(["    " + line if line.strip() else line for line in namespace_code])
        output.append(f"}} // namespace {namespace}\n")

    return "".join(output)
"""

def wrap_with_namespace_gpt(code: str, namespace="submission") -> str:
    lines = code.splitlines(keepends=True)  # Keep line endings intact
    in_struct = False
    struct_brace_depth = 0
    waiting_for_open_brace = False
    in_block_comment = False
    output = []
    namespace_code = []

    for line in lines:
        stripped = line.lstrip()

        # Track block comment state
        line_starts_in_comment = in_block_comment

        # Update block comment state for this line
        temp_line = stripped
        while '/*' in temp_line or '*/' in temp_line:
            if '/*' in temp_line:
                idx = temp_line.find('/*')
                if '*/' in temp_line[idx:]:
                    # Single-line comment /* ... */
                    end_idx = temp_line.find('*/', idx) + 2
                    temp_line = temp_line[:idx] + temp_line[end_idx:]
                else:
                    # Multi-line comment starts
                    in_block_comment = True
                    temp_line = temp_line[:idx]
                    break
            elif '*/' in temp_line:
                # Multi-line comment ends
                idx = temp_line.find('*/')
                in_block_comment = False
                temp_line = temp_line[idx + 2:]

        # If entire line was inside a block comment, route appropriately
        if line_starts_in_comment and in_block_comment:
            if in_struct:
                output.append(line)
            else:
                namespace_code.append(line)
            continue

        # Use temp_line (comment-stripped) for pattern matching
        stripped_no_comments = temp_line.lstrip()
        code_portion = stripped_no_comments.split("//", 1)[0]

        # Exclude headers, macros, and using statements from the namespace
        if stripped_no_comments.startswith(("#include", "#define", "using ", "#if", "#endif", "#else", "#elif", "#ifndef", "#ifdef")):
            output.append(line)
            continue

        # Only exclude #pragma once, not other pragmas like #pragma omp
        if stripped_no_comments.startswith("#pragma once"):
            output.append(line)
            continue

        # Detect struct start (using comment-stripped version)
        struct_candidate = (
            not in_struct
            and re.match(r'^\s*(?:typedef\s+)?struct\b', code_portion)
            and not code_portion.startswith("//")
        )
        if struct_candidate:
            has_open_brace = "{" in code_portion
            has_semicolon = ";" in code_portion

            # Skip bare declarations like `struct Foo;` or `struct Foo bar;`
            if has_semicolon and not has_open_brace:
                namespace_code.append(line)
                continue

            in_struct = True
            waiting_for_open_brace = not has_open_brace
            struct_brace_depth = code_portion.count("{") - code_portion.count("}")
            output.append(line)

            if not waiting_for_open_brace and struct_brace_depth <= 0:
                in_struct = False
                struct_brace_depth = 0
            continue

        if in_struct:
            output.append(line)
            brace_line = code_portion

            if waiting_for_open_brace and "{" in brace_line:
                waiting_for_open_brace = False

            struct_brace_depth += brace_line.count("{") - brace_line.count("}")

            if not waiting_for_open_brace and struct_brace_depth <= 0:
                in_struct = False
                struct_brace_depth = 0
            continue

        # Everything else goes inside the namespace
        namespace_code.append(line)

    # Wrap non-struct code in the namespace
    if namespace_code:
        output.append(f"namespace {namespace} {{\n")
        output.extend(["    " + line if line.strip() else line for line in namespace_code])
        output.append(f"}} // namespace {namespace}\n")

    return "".join(output)

def add_noinline_to_function(cpp_code, function_name):
    # Regex pattern to match the specific function definition
    function_pattern = re.compile(
        rf'(\b(?:void|int|float|double|char|bool|long|short|unsigned|signed|auto|constexpr|inline|static)[\s*&]+)' # Return type
        rf'({function_name})' # Specific function name
        r'\s*\(' # Opening parenthesis for parameters
    )

    # Function to add NOINLINE macro between the return type and the function definition. This is mirrored after ParEval's implementation below.
    def add_noinline(match):
        return f"{match.group(1)}NO_INLINE {match.group(2)}("

    # Apply the regex substitution
    modified_code = function_pattern.sub(add_noinline, cpp_code)

    return modified_code

def extract_function_names_from_prompt(prompt: str):
    # lines = prompt.split("\n")
    # last_line = [line for line in lines if len(line) > 0][-1]
    # Regex pattern to match function definitions (excluding main and class methods)
    function_pattern = re.compile(
        r'\b(?:void|int|float|double|char|bool|long|short|unsigned|signed|auto|constexpr|inline|static|size_t)[\s*&]+'
        r'([a-zA-Z_][a-zA-Z0-9_]*)'  # Capture function name
        r'\s*\('  # Match opening parenthesis
    )

    # Extract all matching function names
    # function_names = function_pattern.findall(last_line)
    function_names = function_pattern.findall(prompt)
    return function_names[-1]

    # assert len(function_names) == 1, f"prompt: {prompt}, lines: {lines}, last line: {last_line}, function_names: {function_names}"
    # return function_names[0]

def build_kokkos(driver_src: PathLike, output_root: PathLike, problem_size: str = "(1<<20)"):
    """ Custom steps for the Kokkos programs, since they require cmake """
    # cp cmake file into the output directory
    cmake_path = "cpp/KokkosCMakeLists.txt"
    cmake_dest = os.path.join(output_root, "CMakeLists.txt")
    run_command(f"cp {cmake_path} {cmake_dest}", dry=False)

    # run cmake and make
    pwd = os.getcwd()
    cmake_flags = f"-DKokkos_DIR=../tpl/kokkos/build -DDRIVER_PATH={pwd} -DDRIVER_SRC_FILE={driver_src} -DDRIVER_PROBLEM_SIZE=\"{problem_size}\""
    cmake_out = run_command(f"cmake -B{output_root} -S{output_root} {cmake_flags}", dry=False)
    if cmake_out.returncode != 0:
        return cmake_out
    return run_command(f"make -C {output_root}", dry=False)

class CppDriverWrapper(DriverWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_driver_file = os.path.join("cpp", "models", DRIVER_MAP[self.parallelism_model])

    def write_source(self, content: str, fpath: PathLike) -> bool:
        """ Write the given c++ source to the given file. """
        with open(fpath, "w") as fp:
            fp.write(content)
        return True

    def patch_prompt(self, content: str) -> str:
        """ Add NO_INLINE to the given source code. """
        # the last line of content should be: return_type function_name(args) {
        # we want to add NO_INLINE after the return_type
        parts = content.split("\n")[-1].split(" ")
        assert len(parts) > 1, f"Could not parse return type from {parts}"
        parts.insert(1, "NO_INLINE")
        return "\n".join(content.split("\n")[:-1] + [" ".join(parts)])

    def compile(
        self, 
        *binaries: PathLike, 
        output_path: PathLike = "a.out", 
        CXX: str = "g++", 
        CXXFLAGS: str = "-std=c++17 -O3",
        problem_size: str = "(1<<20)"
    ) -> BuildOutput:
        """ Compile the given binaries into a single executable. """
        if self.parallelism_model == "kokkos":
            driver_src = [b for b in binaries if b.endswith(".cc")][0]
            compile_process = build_kokkos(driver_src, os.path.dirname(output_path), problem_size=problem_size)
        else:
            binaries_str = ' '.join(binaries)
            macro = f"-DUSE_{self.parallelism_model.upper()}"
            cmd = f"{CXX} {CXXFLAGS} -Icpp -Icpp/models {macro} {binaries_str} -o {output_path}"
            try:
                compile_process = run_command(cmd, timeout=self.build_timeout, dry=self.dry)
            except subprocess.TimeoutExpired as e:
                return BuildOutput(-1, str(e.stdout), f"[Timeout] {str(e.stderr)}")
        return BuildOutput(compile_process.returncode, compile_process.stdout, compile_process.stderr)

    def run(self, executable: PathLike, **run_config) -> RunOutput:
        """ Run the given executable. """
        launch_format = self.launch_configs["format"]
        launch_cmd = launch_format.format(exec_path=executable, args="", **run_config).strip()
        try:
            run_process = run_command(launch_cmd, timeout=self.run_timeout, dry=self.dry)
        except subprocess.TimeoutExpired as e:
            print("--- RUN TIMEOUT ---")
            return RunOutput(-1, str(e.stdout), f"[Timeout] {str(e.stderr)}", config=run_config)
        except UnicodeDecodeError as e:
            assert False
            logging.warning(f"UnicodeDecodeError: {str(e)}\nRunnning command: {launch_cmd}")
            return RunOutput(-1, "", f"UnicodeDecodeError: {str(e)}", config=run_config)

        return RunOutput(run_process.returncode, run_process.stdout, run_process.stderr, config=run_config)

    def test_single_output(self, prompt: str, output: str, test_driver_file: PathLike, problem_size: str) -> GeneratedTextResult:
        """ Test a single generated output. """
        code_opt = True

        logging.debug(f"Testing output:\n{output}")
        with tempfile.TemporaryDirectory(dir=self.scratch_dir) as tmpdir:
            # write out the prompt + output
            src_ext = "cuh" if self.parallelism_model in ["cuda", "hip"] else "hpp"
            src_path = os.path.join(tmpdir, f"generated-code.{src_ext}")

            # include the C++ standard library as well as header for vectorization
            # include_header = "#pragma once\n#include <bits/stdc++.h>\n#include <immintrin.h>\n"
            if self.code_opt:
                # write_success = self.write_source(output, src_path)
                output_with_namespace = wrap_with_namespace_gpt(output)
                write_success = self.write_source(output_with_namespace, src_path)
                # function_name = extract_function_names_from_prompt(prompt)
                # add_noinline = add_noinline_to_function(output_with_namespace, function_name)
                # output = add_noinline
                # write_success = self.write_source(output, src_path)
            else:
                prompt = self.patch_prompt(prompt)
                write_success = self.write_source(include_header+"\n"+prompt+"\n"+output, src_path)

            logging.debug(f"Wrote source to {src_path}.")

            # compile and run the output
            exec_path = os.path.join(tmpdir, "a.out")
            compiler_kwargs = copy.deepcopy(COMPILER_SETTINGS[self.parallelism_model])
            compiler_kwargs["problem_size"] = problem_size  # for kokkos
            compiler_kwargs["CXXFLAGS"] += f" -I{tmpdir} -DDRIVER_PROBLEM_SIZE=\"{problem_size}\" -I{os.path.dirname(test_driver_file)}"
            build_result = self.compile(self.model_driver_file, test_driver_file, output_path=exec_path, **compiler_kwargs)

            if build_result.exit_code != 0:
                print(f"----- DID NOT BUILD ---- build result stderr: {build_result.stderr}")
                print("--- code without namespace ---")
                print(output)
                print("--- CODE ---")
                print(output_with_namespace)

            logging.debug(f"Build result: {build_result}")
            if self.display_build_errors and build_result.stderr and not build_result.did_build:
                logging.debug(build_result.stderr)

            # run the code
            configs = self.launch_configs["params"]
            if build_result.did_build:
                run_results = []
                for c in configs:
                    start = time.time()
                    run_result = self.run(exec_path, **c)
                    end = time.time()
                    print(f"one run time: {end - start}")
                    run_results.append(run_result)

                    print("RUN RESULT: ", run_result)
                    print("STDOUT: ", run_result.stdout)
                    print("STDERR: ", run_result.stderr)
                    # print("--- OUTPUT ---")
                    # print(output_with_namespace)

                    # exit code 0 means no runtime errors
                    if run_result.exit_code == 0 and run_result.is_valid:
                        speedup = run_result.best_sequential_runtime / run_result.runtime
                        print(f"valid run runtime: {run_result.runtime}, best sequential runtime: {run_result.best_sequential_runtime}, speedup: {run_result.best_sequential_runtime / run_result.runtime}")
                    else:
                        print("--- INCORRECT ---")

                    if self.display_runs:
                        logging.debug(run_result.stderr)
                        logging.debug(run_result.stdout)

                    if self.early_exit_runs and (run_result.exit_code != 0 or not run_result.is_valid):
                        break
            else:
                run_results = None
            logging.debug(f"Run result: {run_results}")
            if run_results:
                for run_result in run_results:
                    if run_result.exit_code != 0:
                        logging.debug(f"Ouputs:\n\tstdout: {run_result.stdout}\n\tstderr: {run_result.stderr}")
        
        return GeneratedTextResult(write_success, build_result, run_results)
