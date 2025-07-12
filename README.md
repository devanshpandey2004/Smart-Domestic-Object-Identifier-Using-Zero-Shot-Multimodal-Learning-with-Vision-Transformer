# Smart-Domestic-Object-Identifier-Using-Zero-Shot-Multimodal-Learning-with-Vision-Transformer
A smart AI-powered object identifier using CLIP (Vision Transformer) for zero-shot classification of household items. Users can upload or capture images, and the system predicts the object label with speech output, all via a simple Streamlit web interface.

This project uses OpenAI’s CLIP (Contrastive Language–Image Pretraining) model to identify household objects from images captured via webcam or uploaded manually. It leverages zero-shot learning and a Vision Transformer (ViT) backbone to classify images without retraining.

###  What It Does?
- Takes an image input (via upload or camera)
- Uses CLIP to compare the image against a list of predefined household item labels
- Predicts the most probable object
- Displays top-5 predictions with confidence
- Speaks the top prediction using the browser’s voice engine
###  Zero-Shot Learning in This Project
This project uses zero-shot learning via OpenAI’s CLIP model to identify household objects without any additional training. By comparing the uploaded image with a set of text labels (like "bottle", "laptop", etc.), the model predicts the object directly — even if it has never seen that exact item before.

### Key Features
Zero-shot classification
Vision Transformer-based architecture
Multimodal input (image + text)
Simple web interface using Streamlit
Speech output via browser (JavaScript)

### Tech Stack
Python
PyTorch
Hugging Face Transformers (CLIP)
Streamlit
JavaScript (browser TTS)


### 🚀 Live Demo (Local)
Run with Streamlit:
https://divyadrishti1.streamlit.app





