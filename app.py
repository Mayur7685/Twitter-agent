import streamlit as st
from PIL import Image
import moondream as md
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")

if MOONDREAM_API_KEY is None:
    st.error("API Key is missing. Please set it in the .env file.")
else:
    # Initialize Moondream model with API key
    model = md.vl(api_key=MOONDREAM_API_KEY)

# Title of the Web App
st.title("SnapBack Complaint Analyzer 🚀")

# User Input for Complaint and Image
uploaded_image = st.file_uploader("Upload complaint image", type=["jpg", "jpeg", "png"])
complaint_text = st.text_area("Describe your complaint (e.g., 'Expired milk from Blinkit')", height=100)

if uploaded_image and complaint_text:
    image = Image.open(uploaded_image)

    if st.button("Analyze"):
        with st.spinner("Analyzing complaint..."):
            try:
                # Basic Analysis Prompt
                basic_prompt = f"""
Analyze this food-related complaint: "{complaint_text}". Based on the uploaded image, perform the following checks:

1. **Product Condition**: Assess the physical state of the food item.
2. **Expiry Date**: Verify if the product is expired or close to expiration.
3. **Packaging Integrity**: Check for signs of tampering, leaks, or damage.
4. **Food Safety Concerns**: Identify potential health hazards.

Respond in JSON format:
{{
  "product_condition": "...",
  "expiry_status": "...",
  "packaging_integrity": "...",
  "food_safety_concerns": "...",
  "severity": "..."
}}
"""
                encoded_image = model.encode_image(image)
                basic_analysis = model.query(encoded_image, basic_prompt)["answer"]
                basic_analysis_json = json.loads(basic_analysis)

                # Detailed Observations Prompt for VLM
                detailed_prompt = f"""
Given the image, provide a detailed description of the observed issues, expanding on the initial complaint: "{complaint_text}". Focus on:

- **Detailed Physical Observations**: Describe any visible defects, mold, discoloration, etc.
- **Contextual Information**: Note any text, logos, or environmental context in the image that might relate to the complaint.

Respond in a paragraph format:
"""
                detailed_observations = model.query(encoded_image, detailed_prompt)["answer"]

                # AI Analysis Summary
                ai_analysis = f"""
**AI Analysis Summary:**\n
- **Product Condition:** {basic_analysis_json['product_condition']}
- **Expiry Status:** {basic_analysis_json['expiry_status']}
- **Packaging Integrity:** {basic_analysis_json['packaging_integrity']}
- **Food Safety Concerns:** {basic_analysis_json['food_safety_concerns']}
- **Severity:** {basic_analysis_json['severity'].upper()}
"""

                # Twitter Thread Suggestion
                st.subheader("Suggested X Thread")

                # First Tweet - Image and Complaint
                st.markdown(f"""
**SnapBack AI**  🚀 (@SnapBackAI)  
*1h ago*

🚨 Alert! I encountered a problem with {complaint_text}. @SnapBackAI
""")
                st.image(image, caption="Complaint Image",use_container_width=True)

                st.divider()
                # Second Tweet - Observations
                st.markdown(f"""
**SnapBack AI**  🚀 (@SnapBackAI)  
*1h ago*

👁️ Observations: {detailed_observations}
""")


                st.divider()
                # Third Tweet - AI Analysis
                st.markdown(f"""
**SnapBack AI**  🚀 (@SnapBackAI)  
*1h ago*

{ai_analysis}
""")
                st.divider()
                # Fourth Tweet - Potential Compensation
                compensation_suggestion = f"💰 Potential Compensation: Given the severity, a cashback or monetary compensation could be warranted. @Blinkit, please address this issue."
                st.markdown(f"""
**SnapBack AI**  🚀 (@SnapBackAI)  
*1h ago*

{compensation_suggestion}
""")

                # Hash Tags for all posts with blue color
                st.markdown(f"""
**Hashtags:** :blue[#ConsumerAlert] :blue[#FoodSafety] :blue[#SnapBackAI]
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
