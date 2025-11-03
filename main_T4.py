import torch
from transformers import AutoModel, AutoTokenizer
import os
import tempfile
from PIL import Image
import gradio as gr

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-OCR"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation="eager",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model = model.eval()
    print("✅ مدل با موفقیت بارگذاری شد!")
except Exception as e:
    print(f"❌ خطا در بارگذاری مدل: {e}")
    raise

def process_image_ocr(image, model_size, task_type, is_eval_mode=False, progress=gr.Progress()):
    """
    پردازش تصاویر برای وظایف OCR و Markdown با پشتیبانی از پیشرفت
    """
    if image is None:
        return None, "لطفا ابتدا یک تصویر آپلود کنید.", "لطفا ابتدا یک تصویر آپلود کنید."

    try:
        # به‌روزرسانی پیشرفت
        if progress is not None:
            progress(0.1, desc="در حال آماده‌سازی محیط پردازش...")

        # ایجاد دایرکتوری موقت برای خروجی
        with tempfile.TemporaryDirectory() as output_path:
            # تنظیم prompt بر اساس نوع وظیفه
            if task_type == "OCR":
                prompt = "<image>\nFree OCR. "
            elif task_type == "تبدیل به Markdown":
                prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            else:
                prompt = "<image>\nFree OCR. "

            # ذخیره تصویر آپلود شده به صورت موقت
            temp_image_path = os.path.join(output_path, "temp_image.jpg")
            image.save(temp_image_path, quality=95)

            if progress is not None:
                progress(0.3, desc="در حال بارگذاری و تنظیم تصویر...")

            # پیکربندی پارامترهای اندازه مدل
            size_configs = {
                "کوچک": {"base_size": 640, "image_size": 640, "crop_mode": False},
                "پایه (توصیه شده)": {"base_size": 1024, "image_size": 1024, "crop_mode": True},
                "بزرگ": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
            }

            config = size_configs.get(model_size, size_configs["پایه (توصیه شده)"])

            if progress is not None:
                progress(0.5, desc="در حال اجرای مدل هوشمند...")

            # اجرای استنتاج
            plain_text_result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=output_path,
                base_size=config["base_size"],
                image_size=config["image_size"],
                crop_mode=config["crop_mode"],
                save_results=True,
                test_compress=True,
                eval_mode=is_eval_mode,
            )

            if progress is not None:
                progress(0.8, desc="در حال پردازش نتایج...")

            # تعریف مسیرهای فایل‌های تولید شده
            image_result_path = os.path.join(output_path, "result_with_boxes.jpg")
            markdown_result_path = os.path.join(output_path, "result.mmd")

            # خواندن محتوای فایل markdown در صورت وجود
            markdown_content = ""
            if os.path.exists(markdown_result_path):
                with open(markdown_result_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
            else:
                markdown_content = plain_text_result if plain_text_result else "نتیجه‌ای تولید نشد."

            if progress is not None:
                progress(1.0, desc="پردازش کامل شد!")

            # بازگرداندن نتایج
            text_result = plain_text_result if plain_text_result else markdown_content
            return None, markdown_content, text_result

    except Exception as e:
        error_msg = f"خطا در پردازش تصویر: {str(e)}"
        print(f"❌ {error_msg}")
        return None, error_msg, error_msg