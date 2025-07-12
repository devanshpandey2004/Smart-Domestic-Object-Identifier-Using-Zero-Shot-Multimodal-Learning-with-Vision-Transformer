#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# ✅ Load CLIP model and processor
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

# ✅ Labels of household objects
labels = [
    "bed", "table", "highlighter pen", "peanut packet", "plate", "spoon", "knife",
    "mug", "glass", "bowl", "pan", "kettle", "gas stove", "mop", "bucket", "brush",
    "detergent packet", "charger", "laptop", "mobile phone", "earphones", "headphones",
    "iron", "extension board", "speaker", "charging cable", "door", "window", "curtain",
    "wall", "mirror", "floor", "ceiling", "switchboard", "light switch", "pen",
    "notebook", "bottle", "bag", "hanger", "basket", "shoe", "slippers", "towel",
    "pillow", "bedsheet", "blanket", "comb", "toothbrush", "toothpaste", "soap",
    "shampoo", "scissors", "keys", "lock", "sketch pen", "hair oil bottle", "box",
    "medicine tablet","Man","women"
]

# ✅ Streamlit UI
st.title("🧠 CLIP Household Object Identifier")
st.markdown("Upload or capture an image, and I will tell you what I see — and say it out loud!")

# Upload or camera input
image_data = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
camera_data = st.camera_input("📸 Or take a photo using your webcam")

# Use camera image if both given
if camera_data:
    image = Image.open(camera_data).convert("RGB")
elif image_data:
    image = Image.open(image_data).convert("RGB")
else:
    image = None

if image:
    st.image(image, caption="Your Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        # Prepare inputs
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

        # Get prediction
        top_idx = torch.argmax(probs).item()
        top_label = labels[top_idx]
        confidence = probs[top_idx].item() * 100

        # Display result
        st.success(f"✅ I think it's a **{top_label}** ({confidence:.2f}%)")

        # Top-5 predictions
        top5_probs, top5_indices = torch.topk(probs, 5)
        st.subheader("🔝 Top 5 Predictions")
        for i in range(5):
            label = labels[top5_indices[i]]
            prob = top5_probs[i].item() * 100
            st.write(f"{i+1}. {label} ({prob:.2f}%)")

        # ✅ Speak result using browser
        st.components.v1.html(f"""
            <script>
                var msg = new SpeechSynthesisUtterance("I think it's a {top_label}");
                window.speechSynthesis.speak(msg);
            </script>
        """, height=0)


# In[ ]:




