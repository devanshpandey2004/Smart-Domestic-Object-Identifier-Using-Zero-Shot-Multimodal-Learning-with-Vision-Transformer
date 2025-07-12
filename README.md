# Smart-Domestic-Object-Identifier-Using-Zero-Shot-Multimodal-Learning-with-Vision-Transformer
A smart AI-powered object identifier using CLIP (Vision Transformer) for zero-shot classification of household items. Users can upload or capture images, and the system predicts the object label with speech output, all via a simple Streamlit web interface.
# Smart Domestic Object Identifier Using Zero-Shot Multimodal Learning with Vision Transformer

This project uses OpenAI’s CLIP (Contrastive Language–Image Pretraining) model to identify household objects from images captured via webcam or uploaded manually. It leverages zero-shot learning and a Vision Transformer (ViT) backbone to classify images without retraining.

###  What It Does?
- Takes an image input (via upload or camera)
- Uses CLIP to compare the image against a list of predefined household item labels
- Predicts the most probable object
- Displays top-5 predictions with confidence
- Speaks the top prediction using the browser’s voice engine

---

### 🚀 Live Demo (Local)
Run with Streamlit:
```bash
streamlit run your_script_name.py
