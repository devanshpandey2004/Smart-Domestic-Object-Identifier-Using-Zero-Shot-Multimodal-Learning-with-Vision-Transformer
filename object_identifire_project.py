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
    # 🧍 People
    "man", "woman", "boy", "girl", "human",

    # 🛏️ Bedroom & Personal Items
    "bed", "pillow", "bedsheet", "blanket", "towel", "comb", "mirror",
    "notebook", "pen", "sketch pen", "bag", "shoe", "slippers", "hanger",
    "keys", "lock", "basket",

    # 🍽️ Kitchen & Food Items
    "plate", "spoon", "knife", "mug", "glass", "bowl", "pan",
    "kettle", "peanut packet", "bottle", "box", "gas stove",

    # 🛁 Toiletries & Health
    "toothbrush", "toothpaste", "soap", "shampoo", "hair oil bottle", "medicine tablet",

    # 🧹 Cleaning & Utility
    "mop", "bucket", "brush", "detergent packet", "scrubber", "dustbin",

    # 🔌 Electronics & Accessories
    "mobile phone", "charger", "charging cable", "laptop", "earphones",
    "headphones", "iron", "speaker", "extension board", "remote", "television",

    # 🪑 Furniture & Room Fixtures
    "table", "chair", "door", "window", "curtain", "wall", "floor", "ceiling",
    "switchboard", "light switch", "desk", "mirror", "mat", "lamp", "clock",

    # ✂️ Stationery & Tools
    "highlighter pen", "scissors", "paper", "copy", "stapler", "paper clip"
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




