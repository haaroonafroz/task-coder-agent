# Save as debug_task.py and run: python debug_task.py
import json, sys, textwrap
sys.path.insert(0, '/home/haaroon/coding-agent')

from pipeline.agents import run_planner, run_generator
from pipeline.test_runner import run_tests, extract_code

# Load first non-warmup task (task 14)
with open('datasets/mbpp_subset.json') as f:
    tasks = json.load(f)

task = tasks[3]  # index 3 = task_id 14 (first 3 are warmup)
print("=" * 60)
print(f"TASK {task['task_id']}: {task['text']}")
print(f"TESTS: {task['test_list']}")
print("=" * 60)

# Step 1: Planner
print("\n--- PLANNER OUTPUT ---")
plan = run_planner(task['text'], mode='baseline')
print(plan.text)

# Step 2: Generator
print("\n--- GENERATOR RAW OUTPUT (repr) ---")
gen = run_generator(task['text'], plan.text, mode='baseline')
print(repr(gen.text))

# Step 3: What extract_code produces
print("\n--- AFTER extract_code ---")
extracted = extract_code(gen.text)
print(repr(extracted))

# Step 4: Build the test script exactly as test_runner would
print("\n--- TEST SCRIPT THAT GETS EXECUTED ---")
test_block = "\n".join(task['test_list'])
full_script = (
    "import resource as _resource\n"
    "_resource.setrlimit(_resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))\n"
    "\n"
    f"{extracted}\n"
    "\n"
    f"{test_block}\n"
    'print("PASS")\n'
)
print(full_script)

# Step 5: Run it
print("\n--- TEST RESULT ---")
result = run_tests(gen.text, task['test_list'])
print(result)