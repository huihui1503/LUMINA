from lumina import LUMINA
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def main():
    model = AutoModelForCausalLM.from_pretrained('...')
    tokenizer = AutoTokenizer.from_pretrained('...')

    detector = LUMINA(model, tokenizer)

    prompt_w_context = "Instruction: [INSTRUCTION] Context: [CONTEXT]"
    prompt_w_random_context = "Instruction: [INSTRUCTION] Context: [RANDOM CONTEXT]"
    response = "[RESPONSE]"

    # Returns (hallucination_score, mmd, ipr)
    hallucination_score, mmd, ipr = detector.predict(
        prompt_w_context, 
        prompt_w_random_context, 
        response
    )


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

        
    main()
