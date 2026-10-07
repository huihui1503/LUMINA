from lumina import LUMINA
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import time
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Run hallucination-detection pipeline")
    parser.add_argument(
        "-model", "--model",
        type=str,
        default=None,
        choices=["llama2-7b", "llama2-13b", "mistral-7b", "llama3-8b"],
        help="Which model to run"
    )
    parser.add_argument(
        "-dataset", "--dataset",
        type=str,
        default=None,
        choices=["ragtruth", "hallurag", "dolly"],
        help="Quantization mode (optional)"
    )

    return parser.parse_args()

def read_json(file_path, data_type):
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    filtered_data = [i for i in data if i["model"] == data_type]
    return filtered_data

if __name__ == "__main__":
    args = parse_args()

    print(args.model, args.dataset)
    if args.model == "llama2-7b":
        llm_model_name = "meta-llama/Llama-2-7b-chat-hf"
        data_type = "llama-2-7b-chat"
    elif args.model == "llama2-13b":
        llm_model_name = "meta-llama/Llama-2-13b-chat-hf"
        data_type="llama-2-13b-chat"
    elif args.model == "llama3-8b":
        llm_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        data_type = "llama-3-8b-instruct"
    elif args.model == "mistral-7b":
        llm_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        data_type = "mistral-7B-instruct"
    else:
        print("model name error")
        exit(-1)

    if args.dataset == "dolly":
        file_path = "datasets/dolly/preprocessed_merged.jsonl"
    else:
        print("dataset name error")
        exit(-1)

    dataset = read_json(
        file_path=file_path,
        data_type=data_type
    )

    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.float32
    ).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    
    detector = LUMINA(model, tokenizer)

    labels = []
    scores = []
    for i in tqdm(dataset):
        prompt_w_context = i["prompt_w_context"]
        prompt_w_random_context = i["prompt_w_random_context"]
        response = i["response"]

        start = time.time()
        # Returns (hallucination_score, mmd, ipr)
        hallucination_score, mmd, ipr = detector.predict(
            prompt_w_context, 
            prompt_w_random_context, 
            response
        )

        labels.append(i["labels"])
        scores.append(hallucination_score.mean())

        end = time.time()
        print(f"Running time: {end - start:.4f} seconds")

    y_true = np.array(labels)
    y_score = torch.stack(scores).cpu().numpy()
    auroc = roc_auc_score(y_true, y_score)

    print(f"Auroc score: {auroc:.4f}")



    np.save(f"logs/{data_type}_{dataset}.npy", y_score)


