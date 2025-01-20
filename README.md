# AI-Therapist

An empathy-driven therapist chatbot built using the Llama 3.3 language model. This project aims to provide conversational support for mental health and emotional well-being through non-diagnostic dialogue.


## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-therapist.git
cd ai-therapist
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Hugging Face API token:
```env
HF_AUTH_TOKEN=your_token_here
```


## 💻 Usage

1. Start the chatbot:
```bash
python main.py
```

2. Interact with the chatbot through the command line interface:
```
Hello, I'm TherapistBot. I'm here to offer support. Type 'exit' or 'quit' at any time to end our session.

User: I'm feeling anxious about work lately
TherapistBot: I understand that work-related anxiety can be really challenging...
```

## ⚙️ Configuration

Key settings can be modified in `config/settings.py`:

```python
# Model settings
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LOCAL_CACHE_DIR = "/path/to/cache"

# Generation settings
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 0.9

# Conversation settings
MAX_RESPONSE_LENGTH = 1000
MAX_TURNS = 10
```

## ❗ Important Notes

- This chatbot is designed for emotional support only and does not provide medical advice or diagnosis
- All conversations are processed locally and are not stored permanently
- The model requires significant computational resources; performance may vary based on hardware
- Response generation parameters can be adjusted in the configuration file to optimize the conversation flow

## 🤝 Contributing

Contributions are welcome! 

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer:** This chatbot is not a replacement for professional mental health services. If you're experiencing serious mental health issues, please seek help from a qualified mental health professional.