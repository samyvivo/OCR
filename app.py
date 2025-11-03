import gradio as gr
import spaces
from main_T4 import process_image_ocr
import time

# CSS for better Persian styling
custom_css = """
.persian-text { 
    font-family: "Vazirmatn", "Tahoma", "Arial", sans-serif;
    direction: rtl;
}
.rtl-direction {
    direction: rtl;
    text-align: right;
}
.center-content {
    display: flex;
    justify-content: center;
    align-items: center;
}
.progress-text {
    text-align: center;
    font-weight: bold;
    margin: 10px 0;
}
.markdown-output {
    min-height: 400px;
    border: 1px solid #e0e0e0;
    padding: 15px;
    border-radius: 8px;
}
"""

def process_image_with_progress(image, model_size, task_type):
    """
    تابع پردازش تصویر با نوار پیشرفت
    """
    progress = gr.Progress()
    
    # شبیه‌سازی مراحل پیشرفت
    progress(0, desc="در حال آماده‌سازی مدل...")
    time.sleep(0.5)
    
    progress(0.3, desc="در حال پردازش تصویر...")
    time.sleep(0.5)
    
    progress(0.6, desc="در حال استخراج متن...")
    time.sleep(0.5)
    
    progress(0.8, desc="در حال تولید خروجی...")
    
    # پردازش اصلی
    result_image, markdown_content, text_result = process_image_ocr(
        image, model_size, task_type, is_eval_mode=False
    )
    
    progress(1.0, desc="پردازش کامل شد!")
    
    return markdown_content, text_result

# ایجاد رابط Gradio بهبود یافته
with gr.Blocks(
    title=" OCR استخراج متن از تصویر", 
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"),
    css=custom_css
) as demo:
    
    # هدر اصلی
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(
                """
                <div class="persian-text" style="text-align: center;">
                    <h1>🧠 پردازش هوشمند تصویر-OCR</h1>
                    <h3>استخراج هوشمند متن از تصاویر</h3>
                    <p>تصویر خود را آپلود کنید تا متن آن به صورت خودکار استخراج شود</p>
                </div>
                """
            )
    
    with gr.Row():
        # پنل ورودی‌ها
        with gr.Column(scale=1, min_width=400):
            with gr.Group():
                gr.Markdown("### ⚙️ تنظیمات پردازش", elem_classes="persian-text")
                
                image_input = gr.Image(
                    type="pil", 
                    label="📷 تصویر ورودی",
                    sources=["upload", "clipboard"],
                    height=300,
                    elem_classes="rtl-direction"
                )
                
                model_size = gr.Dropdown(
                    choices=["کوچک", "پایه (توصیه شده)", "بزرگ"],
                    value="پایه (توصیه شده)",
                    label="📊 اندازه مدل",
                    info="مدل بزرگتر دقت بهتر اما سرعت کمتر",
                    elem_classes="rtl-direction"
                )
                
                task_type = gr.Dropdown(
                    choices=["OCR", "تبدیل به Markdown"],
                    value="OCR",
                    label="🎯 نوع وظیفه",
                    info="OCR: فقط استخراج متن | Markdown: ساختاردهی پیشرفته",
                    elem_classes="rtl-direction"
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ پاک کردن", size="sm")
                    submit_btn = gr.Button("🚀 شروع پردازش", variant="primary", size="lg")
        
        # پنل خروجی‌ها
        with gr.Column(scale=2, min_width=600):
            with gr.Tabs() as tabs:
                # تب پیش‌نمایش Markdown
                with gr.TabItem("📝 پیش‌ نمایش", id=1):
                    gr.Markdown("**خروجی قالب‌ بندی شده:**", elem_classes="persian-text")
                    output_markdown = gr.Markdown(
                        elem_classes=["persian-text", "markdown-output"],
                        value="خروجی اینجا نمایش داده می‌شود..."
                    )
                
                # تب متن خام
                with gr.TabItem("📄 متن خام", id=2):
                    output_text = gr.Textbox(
                        lines=20,
                        show_copy_button=True,
                        label="متن استخراج شده",
                        elem_classes="rtl-direction",
                        value="متن استخراج شده در اینجا نمایش داده می‌شود..."
                    )
            
            # بخش اطلاعات و راهنما
            with gr.Accordion("ℹ️ راهنمای استفاده", open=False):
                gr.Markdown("""
                **راهنمای سریع:**
                
                - **تصویر با کیفیت بالا** آپلود کنید
                - برای **اسناد متنی** از حالت 'پایه' استفاده کنید
                - برای **تصاویر پیچیده** از حالت 'بزرگ' استفاده کنید
                - حالت **Markdown** برای اسناد ساختاریافته مناسب است
                
                **نکات:**
                - فرمت‌های پشتیبانی شده: JPG, PNG, WebP
                - حداکثر حجم تصویر: 10MB
                - پردازش ممکن است 10-30 ثانیه زمان ببرد
                """, elem_classes="persian-text")
    
    # بخش مثال‌ها
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📁 مثال‌های آماده", elem_classes="persian-text")
            gr.Examples(
                examples=[
                    ["example1.png", "پایه (توصیه شده)", "OCR"],
                    ["example2.png", "پایه (توصیه شده)", "تبدیل به Markdown"],
                ],
                inputs=[image_input, model_size, task_type],
                outputs=[output_markdown, output_text],
                fn=process_image_with_progress,
                cache_examples=False,
                label="برای تست سریع روی یکی از مثال‌ها کلیک کنید",
                examples_per_page=3
            )
    
    # وضعیت سیستم
    with gr.Row():
        gr.HTML("""
        <div class="persian-text" style="text-align: center; color: #666; font-size: 0.9em; margin-top: 20px;">
            <p>ساخته شده توسط *سامان زیتونیان* | OCR | پردازش تصویر هوشمند</p>
        </div>
        """)
    
    # مدیریت رویدادها
    def clear_all():
        return None, "خروجی اینجا نمایش داده می‌شود...", "متن استخراج شده در اینجا نمایش داده می‌شود..."
    
    # اتصال دکمه‌ها
    submit_btn.click(
        fn=process_image_with_progress,
        inputs=[image_input, model_size, task_type],
        outputs=[output_markdown, output_text],
        show_progress="minimal"
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[image_input, output_markdown, output_text]
    )

# راه‌اندازی برنامه
if __name__ == "__main__":
    demo.launch(
        share=True,
        show_error=True
    )