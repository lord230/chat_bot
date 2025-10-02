# chat_bot

A project to help you build your own language model (LM) — especially using WhatsApp data (or your own custom data) — with minimal hassle.  
This repository provides tools, scripts, and guidance to train, fine-tune, and deploy a conversational AI.  

GitHub: https://github.com/lord230/chat_bot.git  

---

##  Features

- Preprocessing pipeline for chat / conversational data  
- Training scripts to fine-tune a causal LM on your own dataset  
- Support for using WhatsApp chats or arbitrary text conversation logs  
- Example configurations / templates you can adapt  
- Simple inference / chat loop for testing the chatbot  

---

##  Repository Structure

Here’s a rough outline of the files and folders:

chat_bot/
├── data_aug/ # Data augmentation scripts or utilities
├── bot # Core chatbot / training / inference code
├── req.txt # Required dependencies
└── README.md # This file



- `data_aug/` — scripts to clean, augment, or transform chat logs  
- `bot/` — main model training, fine-tuning, and chat/run code  
- `req.txt` — Python package dependencies  

---

## 🛠️ Setup & Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/lord230/chat_bot.git
   cd chat_bot

2. Create a virtual environment (optional but recommended):
   ```bash
    python3 -m venv venv
    source venv/bin/activate
3.Install dependencies:
   ```bash
   pip install -r req.txt
   ```
   
4. Prepare your data:

Put your chat logs or conversation data in a designated folder (e.g. data/)

Clean / format them as needed (see data_aug/ for help)

  


