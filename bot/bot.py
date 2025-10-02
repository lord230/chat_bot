import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./fine_tuned_phi3_bot"  
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.float16 if device == "cuda" else torch.float32
)
model.eval()


def generate_reply(user_input,
                   history="",
                   max_new_tokens=400,
                   temperature=0.7,
                   top_p=0.9,
                   repetition_penalty=1.1,
                   no_repeat_ngram_size=3):


    prompt = f"""{history}
User: {user_input}
Assistant:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=None,
        early_stopping=False
    )


    gen_tokens = outputs[0][input_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)


    text = re.sub(r'^\s*(Assistant:|Bot:)\s*', '', text, flags=re.IGNORECASE)

    return text.strip(), prompt + " " + text.strip() + "\n"


if __name__ == "__main__":
    print("Chatbot is ready! Type 'exit' or 'quit' to stop.\n")
    history = ""

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye! 👋")
            break

        reply, history = generate_reply(user_input, history)
        print("Bot:", reply)
