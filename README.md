# 🧠 DeepSeek OCR – Multilingual Image-to-Text Extraction

A lightweight and powerful **Optical Character Recognition (OCR)** web app powered by **[DeepSeek-AI/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)** and deployed on **Hugging Face Spaces**.  
It supports **Persian**, **English**, and other languages with automatic text recognition from images.

🔗 **Live Demo:** [https://huggingface.co/spaces/samyhusy/OCR](https://huggingface.co/spaces/samyhusy/OCR)

---

## 🚀 Features

- 🖼️ Upload or paste any image containing text  
- 🌐 Supports **Persian (Farsi)**, **English**, and **multilingual OCR**  
- ⚙️ Built with **Gradio UI** for an intuitive interface  
- ⚡ Optimized for different GPUs (A100, T4, 1660 Ti) using `attn_implementation` variants  
- 🧩 Runs locally or in Hugging Face Spaces  
- 🔍 Transformer-based architecture using **FlashAttention 2** when available  

---

## 🧩 Architecture

This app uses the **DeepSeek-OCR** model from the Hugging Face Hub.

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = "deepseek-ai/DeepSeek-OCR"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
```

🖥️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/samyvivo/OCR.git
cd OCR
pip install -r requirements.txt
```

If using CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

▶️ Usage

Run the app locally:
```bash
python main.py
```

hen open the URL shown in your terminal (e.g. http://127.0.0.1:7860) in your browser.

Upload an image, and the model will automatically detect and extract text.

⚙️ GPU Compatibility
```bash
| GPU Type    | Supported `attn_implementation` | Note                           |
| ----------- | ------------------------------- | ------------------------------ |
| A100        | ✅ `"flash_attention_2"`         | Best performance               |
| T4          | ⚠️ `"eager"` or `"sdpa"`        | FlashAttention 2 not supported |
| GTX 1660 Ti | ⚠️ `"eager"` only               | Older architecture             |
```

The app automatically detects GPU capability and adjusts configuration.

📁 Project Structure
```bash
OCR/
│
├── main.py              # Main Gradio application
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
└── assets/              # Example images and logs
```

🧑‍💻 Author

Saman Zeitounian
📊 Data Scientist | Machine Learning Engineer

🌐 [LinkedIn](https://www.linkedin.com/in/saman-zeitounian-56a0a5164)

💻 [GitHub](https://github.com/samyvivo)

📈 [Kaggle](https://www.kaggle.com/samanzeitounain)

📝 License

This repository is licensed under the MIT License.
Feel free to use, modify, and share it with attribution.
