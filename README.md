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
    torch_dtype=torch.bfloat16,
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
| GPU                        | Architecture             | FlashAttention 2 Support | Native BF16 Support           | Recommended `torch_dtype`           | Recommended `_attn_implementation` | Relative Speed ⚡ | Notes                                                       |
| -------------------------- | ------------------------ | ------------------------ | ----------------------------- | ----------------------------------- | ---------------------------------- | ---------------- | ----------------------------------------------------------- |
| **H100**                   | **Hopper (SM 90)**       | ✅ **Yes (best)**         | ✅ **Yes (native)**            | `torch.bfloat16`                    | `"flash_attention_2"`              | ⚡⚡⚡⚡             | Fastest GPU for FA2 — supports FP8/BF16, highest efficiency |
| **A100**                   | **Ampere (SM 80)**       | ✅ **Yes**                | ✅ **Yes (native)**            | `torch.bfloat16`                    | `"flash_attention_2"`              | ⚡⚡⚡              | Excellent performance; ideal for FA2 inference/training     |
| **RTX 4090 / 4080 / 4070** | **Ada Lovelace (SM 89)** | ✅ **Yes**                | ⚠️ Limited (driver-dependent) | `torch.float16` or `torch.bfloat16` | `"flash_attention_2"`              | ⚡⚡⚡              | BF16 not always stable; FA2 works perfectly                 |
| **RTX 3090 / 3080 / 3070** | **Ampere (SM 86)**       | ✅ **Yes**                | ⚠️ Partial (no native BF16)   | `torch.float16`                     | `"flash_attention_2"`              | ⚡⚡               | Great speed, less stable in mixed precision                 |
| **A10 / L40 / L4**         | **Ampere / Ada**         | ✅ **Yes**                | ✅                             | `torch.bfloat16`                    | `"flash_attention_2"`              | ⚡⚡               | Datacenter GPUs similar to A100 in FA2 support              |
| **T4**                     | **Turing (SM 75)**       | ❌ **No**                 | ❌                             | `torch.float16`                     | `"sdpa"`                           | ⚡                | Solid FP16 inference, but no FA2 kernels                    |
| **RTX 2080 / 2070 / 2060** | **Turing (SM 75)**       | ❌ **No**                 | ❌                             | `torch.float16`                     | `"sdpa"`                           | ⚡                | Same as T4 — FA2 unsupported                                |
| **GTX 1660 Ti / 1650**     | **Turing (SM 75)**       | ❌ **No**                 | ❌                             | `torch.float16`                     | `"sdpa"`                           | ⚙️               | Consumer GPU, limited memory, good with SDPA                |
| **Older (GTX 10xx)**       | **Pascal (SM 61)**       | ❌ **No**                 | ❌                             | `torch.float32`                     | `"eager"`                          | 🐢               | Only eager attention works reliably                         |

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
