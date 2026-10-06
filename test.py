from lumina import LUMINA
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float32
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf"
)

detector = LUMINA(model, tokenizer)


def read_json(file_path):
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data
dolly_data = read_json("datasets/dolly/merged.jsonl")

correct_passages = dolly_data[0]["source_info"]["passages"].replace("\n", " ")
correct_passages = correct_passages.replace("  ", " ")

random_passages = dolly_data[10]["source_info"]["passages"].replace("\n", " ")
random_passages = random_passages.replace("  ", " ")

prompt = dolly_data[0]["prompt"].replace("\n", " ")
correct_instruction = prompt.replace("  ", " ")
random_instruction = correct_instruction.replace(correct_passages, random_passages)
print(random_instruction)
print(correct_instruction)



prompt_w_context = correct_instruction
prompt_w_random_context = random_instruction
response = dolly_data[0]["response"]

# Returns (hallucination_score, mmd, ipr)
hallucination_score, mmd, ipr = detector.predict(
    prompt_w_context, 
    prompt_w_random_context, 
    response
)

print(hallucination_score.mean())
print(mmd)
print(ipr)