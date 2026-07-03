from config import MODEL_NAME , NUM_BASELINE_SAMPLES, RUN_FULL_EVALUATION
from transformers import AutoTokenizer , AutoModelForCausalLM
from load_dataset import get_dataset
import pandas as pd

data = get_dataset()
test_data = data['test']

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if RUN_FULL_EVALUATION :
    num_samples = len(test_data)
else: 
    num_samples = NUM_BASELINE_SAMPLES

results = []

for i in range(num_samples):
    inputs = test_data['instruction'][i]
    gold_answer = test_data['response'][i]

    tokenized_input = tokenizer(inputs , return_tensors = 'pt')
    input_tokens = tokenized_input['input_ids'].shape[1]

    prediction = model.generate(**tokenized_input , max_new_tokens = 150)
    generated_tokens = prediction[0][input_tokens:]

    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    results.append({
        "input": inputs,
        "Prediction" : result,
        "Ground_truth" : gold_answer
    })

    if i % 100 == 0:
        print(f"Completed {i}/{len(test_data)}")

results_df = pd.DataFrame(results)
results_df.to_csv('results/baseline_evaluation.csv',index=False)
