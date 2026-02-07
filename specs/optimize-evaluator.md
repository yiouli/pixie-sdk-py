Your task is to create an async function to optimize the DSPy module inside rate_prompt_llm_call(), using bootstrapFewShot, and data from the llm call record storage; then properly store the optimized program for future use.

The data records are associated with prompts. optimize function should create optimized module specific for each prompt, so the optimize function should take a prompt_id as args.

You can get the labeled records using get_llm_call_records(), filter by prompt_id, rating, and ratedby source. for the train/val data, you should only get records rated good/bad, and rated by user/system.

You'd need to transform the storage record property before feeding them into the optimizer. You should read the code in batch_call_llm (server handler) and recordToTestCase (client code) to understand how a llmcallrecord is converted to the data structure feeding into rate_prompt_llm_call(), and you need to implement the exact same logic to prepare the data for the optimizer.

once optimization is complete, you should store the optimzed dspy program state in .pixie/evaluators/<prompt_id>/v<version_number>.json. e.g.

```
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

...
dspy_program = dspy.ChainOfThought("question -> answer")

optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5)
compiled_dspy_program = optimizer.compile(dspy_program, trainset=...)


compiled_dspy_program.save("./dspy_program/program.json", save_program=False)
```

Once you setup the storage, you should also create the function to return the latest optimized dspy program by promptId.
