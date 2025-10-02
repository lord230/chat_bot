import re
import json
import os
import emoji
import random

def clean_message(msg: str) -> str:

    msg = emoji.replace_emoji(msg, replace="")

 
    skip_phrases = [
        "this message was deleted",
        "<media omitted>",
        "image omitted",
        "video omitted",
        "sticker omitted",
        "audio omitted",
        "gif omitted",
        "<This message was edited>",
        "document omitted"
    ]
    lower_msg = msg.lower()
    if any(phrase in lower_msg for phrase in skip_phrases):
        return ""

    # Strip extra spaces
    return msg.strip()

def parse_whatsapp_folder(folder_path, output_file="whatsapp_training.jsonl", shuffle=True):

    patterns = [
        re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2} [ap]m) - (.*?): (.*)$"),  # Format A (12h)
        re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - (.*?): (.*)$"),         # Format B (24h)
        re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4}), (.*?)\] (.*?): (.*)$"),                # Format C [..]
        re.compile(r"^(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}\s?[ap]m) - (.*?): (.*)$")   # Format D (yy)
    ]

    conversations = []

    # Loop over all txt files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {file_path}...")

            prev_message = None
            prev_sender = None

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                match = None
                for p in patterns:
                    match = p.match(line)
                    if match:
                        break

                if match:
                    _, _, sender, message = match.groups()
                    message = clean_message(message)

                    if not message:  
                        continue

                    
                    if prev_message and prev_sender and prev_sender != sender:
                        conversations.append({
                            "prompt": prev_message,
                            "response": message
                        })

                    prev_message = message
                    prev_sender = sender
                else:
                    
                    if prev_message:
                        prev_message += " " + line

    
    if shuffle:
        random.shuffle(conversations)


    with open(output_file, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Extracted {len(conversations)} clean samples → {output_file}")


if __name__ == "__main__":
    parse_whatsapp_folder("chat_folder", "whatsapp_training.jsonl", shuffle=True)
