import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
from transformers import pipeline
import io
import cv2
import numpy as np
import exifread
from collections import Counter
import json
import base64
import re
try:
    import google.generativeai as genai
except ImportError:
    genai = None
import os
import hashlib
from web3 import Web3
from dotenv import load_dotenv
from blockchain import BlockchainManager
from BlockLens_ai import BlockLensManager

load_dotenv()

gemini_model = None
if genai:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')

st.set_page_config(
    page_title="BlockLens",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def get_blockchain_manager():
    return BlockchainManager()

bc = get_blockchain_manager()

@st.cache_resource
def get_blocklens_manager():
    return BlockLensManager()

blocklens_manager = get_blocklens_manager()

@st.cache_resource
def load_models():
    models = {}
    model_configs = [
        ("umm-maybe/AI-image-detector", "image-classification"),
        ("facebook/dino-vits16", "image-classification"),
        ("google/vit-base-patch16-224", "image-classification"),
        ("prithivMLmods/Deep-Fake-Detector-v2-Model", "image-classification"),
        ("dima806/deepfake_vs_real_image_detection", "image-classification"),
        ("Organika/sdxl-detector", "image-classification")
    ]

    for model_name, task in model_configs:
        try:
            models[model_name] = pipeline(task, model=model_name)
        except Exception as e:
            models[model_name] = None

    return models

pipes = load_models()

def ela_analysis(image):
    try:
        image = image.convert('RGB')
        temp = io.BytesIO()
        image.save(temp, 'JPEG', quality=90)
        temp.seek(0)
        compressed = Image.open(temp)

        ela = ImageChops.difference(image, compressed)

        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        ela = ImageEnhance.Brightness(ela).enhance(scale)

        ela_array = np.array(ela)
        avg_diff = np.mean(ela_array)
        return avg_diff, ela
    except Exception as e:
        st.error(f"ELA Analysis failed: {e}")
        return 0, None

def noise_analysis(image):
    try:
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        noise = cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0))
        noise_var = np.var(noise)
        return noise_var
    except Exception as e:
        st.error(f"Noise Analysis failed: {e}")
        return 0

def metadata_analysis(image_file):
    try:
        image_file.seek(0)
        tags = exifread.process_file(image_file)
        software = str(tags.get('Image Software', ''))
        if 'photoshop' in software.lower() or 'ai' in software.lower() or 'gimp' in software.lower():
            return False, software
        return True, software
    except Exception as e:
        st.error(f"Metadata Analysis failed: {e}")
        return True, "Unknown"

def detect_screenshot_heuristic(image, noise_score, software_tag):
    is_screenshot = False
    confidence = 0

    if noise_score < 5.0:
        is_screenshot = True
        confidence = 90
    elif noise_score < 15.0:
        is_screenshot = True
        confidence = 70

    software_lower = software_tag.lower()
    if 'screenshot' in software_lower or 'snip' in software_lower or 'capture' in software_lower:
        is_screenshot = True
        confidence = 95

    return is_screenshot, confidence

def analyze_image(image_file):
    image_file.seek(0)
    image = Image.open(image_file)

    model_opinions = {}
    votes = []

    for model_name, pipe in pipes.items():
        if pipe:
            try:
                results = pipe(image)
                top_result = results[0]
                label = top_result['label']
                score = top_result['score']

                if 'real' in label.lower() or 'authentic' in label.lower():
                    decision = "real_image"
                else:
                    decision = "ai_generated"

                votes.append(decision)
                reasoning = f"Predicted '{label}' with confidence {score:.2f} based on visual patterns and model training."
                model_opinions[model_name] = {"decision": decision, "reasoning": reasoning}

            except Exception as e:
                model_opinions[model_name] = {"decision": "unknown", "reasoning": "Model failed to analyze"}
        else:
            model_opinions[model_name] = {"decision": "unknown", "reasoning": "Model not loaded"}

    if gemini_model:
        try:
            image_bytes = image_file.getvalue()
            mime_type = image_file.type if hasattr(image_file, 'type') else "image/jpeg"
            GEMINI_IMAGE_PROMPT = """
Analyze this image and classify it as ONE of these 3 categories ONLY:

**real_image** - Authentic camera photo (phone/camera taken)
**ai_generated** - AI-created/synthesized image
**screenshot** - Screen capture/digital composite

LOOK FOR THESE CLUES:
- **Screenshot**: UI elements, perfect edges, compression blocks, browser chrome, low noise variance
- **AI Generated**: Anatomical errors (extra fingers, weird hands), symmetrical artifacts, unnatural lighting/shadows, blurry text/logos
- **Real Photo**: Natural noise/grain, lens distortion, organic lighting, camera sensor artifacts

OUTPUT EXACTLY:
{
  "decision": "real_image" | "ai_generated" | "screenshot",
  "confidence": 85,  // 0-100
  "evidence": "2-3 specific visual clues you saw"
}

NEVER say "uncertain" - pick your best guess with realistic confidence.
"""
            response = gemini_model.generate_content([
                GEMINI_IMAGE_PROMPT,
                {"inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(image_bytes).decode()
                }}
            ])
            text = re.sub(r'```json\s*', '', response.text).strip('`').strip()
            result = json.loads(text)
            model_opinions["Gemini"] = {"decision": result["decision"], "reasoning": result["evidence"]}
            votes.append(result["decision"])
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                model_opinions["Gemini"] = {
                    "decision": "unknown", 
                    "reasoning": "⚠️ Daily AI Quota Exceeded. The Teacher is taking a break. Please wait or check billing."
                }
            else:
                model_opinions["Gemini"] = {
                    "decision": "unknown", 
                    "reasoning": f"Gemini failed: {error_msg}"
                }

    ela_score, ela_image = ela_analysis(image)
    noise_score = noise_analysis(image)
    meta_ok, software = metadata_analysis(image_file)

    is_screenshot, screenshot_conf = detect_screenshot_heuristic(image, noise_score, software)

    ai_probs = []
    for m in model_opinions.values():
        if m["decision"] == "ai_generated":
            ai_probs.append(1.0) 
        elif m["decision"] == "real_image":
            ai_probs.append(0.0)
    
    avg_ai_prob = sum(ai_probs) / len(ai_probs) if ai_probs else 0.5
    
    meta_score = 1.0 if not meta_ok else 0.0
    signals = [ela_score, noise_score, screenshot_conf/100.0, avg_ai_prob, meta_score]

    blocklens_verdict, blocklens_conf = blocklens_manager.predict(image, signals)

    gemini_result = model_opinions.get("Gemini", {})
    gemini_decision = gemini_result.get("decision", "unknown")

    if gemini_decision != "unknown":
        final_decision = gemini_decision
        confidence = 100 
        supporting_reasoning = f"{gemini_result.get('reasoning', '')}"
        
        loss = blocklens_manager.train_step(image, signals, gemini_decision)
        if loss:
            print(f"BlockLens Model trained. Loss: {loss:.4f}")
            
    else:
        if votes:
            vote_counts = Counter(votes)
            most_common = vote_counts.most_common(1)[0]
            final_decision = most_common[0]
            confidence = int((most_common[1] / len(votes)) * 100)
            supporting_reasoning = f"Gemini unavailable. Consensus reached: {final_decision}."
        else:
            final_decision = "unknown"
            confidence = 0
            supporting_reasoning = "Insufficient data."

    return {
        "final_decision": final_decision,
        "confidence": confidence,
        "supporting_reasoning": supporting_reasoning,
        "model_opinions": model_opinions,
        "blocklens_verdict": blocklens_verdict,
        "blocklens_confidence": blocklens_conf,
        "signals": signals,
        "gemini_decision": gemini_decision
    }, ela_image, software

st.title("BlockLens")
st.markdown("### AI Image Detection & Analysis")

if not bc.connected:
    st.warning("⚠️ Blockchain not connected. Check .env configuration.")

uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width='stretch')

    image_bytes = uploaded_file.getvalue()
    image_hash = bc.hash_image(image_bytes)
    st.code(f"Image Hash: {image_hash}", language="text")

    existing_verdict = bc.get_verdict(image_hash)

    if existing_verdict:
        st.info("This image is already registered on the Blockchain!")
        status = existing_verdict['status']
        if status == "Real":
            status_text = "Authentic Photo"
        elif status == "AI-Generated":
            status_text = "AI Generated: Yes"
        elif status == "Screenshot":
            status_text = "Screenshot: Yes"
        else:
            status_text = f"Result: {status}"
        st.write(f"**{status_text}**")
        st.write(f"**Confidence:** {existing_verdict['confidence']}%")
        st.write(f"**Timestamp:** {existing_verdict['timestamp']}")
        st.write("**Registrar:**")
        st.code(existing_verdict['registrar'], language="text")
        st.markdown(f"[Check Registrar Transaction History on Sepolia](https://sepolia.etherscan.io/address/{existing_verdict['registrar']})")
    else:
        st.write("This image has not been registered yet.")

    st.divider()

    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.analysis_results = None
        st.session_state.last_uploaded_file = uploaded_file.name

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            analysis_result, ela_image, software = analyze_image(uploaded_file)

            st.session_state.analysis_results = {
                "analysis": analysis_result,
                "ela_image": ela_image,
                "software": software
            }

    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        analysis = results["analysis"]
        verdict = analysis["final_decision"]
        
        confidence = analysis["confidence"]

        if verdict == "real_image":
            st.success(f"Authentic Photo (Confidence: {confidence}%)")
        elif verdict == "screenshot":
            st.warning(f"Screenshot Detected (Confidence: {confidence}%)")
        else:
            st.error(f"AI-Generated (Confidence: {confidence}%)")

        with st.expander("Analysis Details"):
            st.write(f"**Supporting Reasoning:** {analysis['supporting_reasoning']}")
            st.write("**Model Opinions:**")
            for model, opinion in analysis['model_opinions'].items():
                st.write(f"- **{model}:** {opinion['decision']} - {opinion['reasoning']}")
            st.write(f"**Metadata Software:** {results['software']}")
            if results['ela_image']:
                st.image(results['ela_image'], caption="Error Level Analysis (ELA)", width='stretch')

        st.divider()
        st.subheader("Incorrect Verdict? Provide Feedback")
        st.caption("Help improve the BlockLens Student AI by providing the correct label if the system is wrong.")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        override_verdict = None
        
        if col_f1.button("Mark as Real Photo"):
            override_verdict = "real_image"
        if col_f2.button("Mark as AI Generated"):
            override_verdict = "ai_generated"
        if col_f3.button("Mark as Screenshot"):
            override_verdict = "screenshot"
            
        if override_verdict:
            with st.spinner(f"Retraining Student AI with correction: {override_verdict}..."):
                uploaded_file.seek(0)
                image_for_training = Image.open(uploaded_file)
                loss = blocklens_manager.train_step(image_for_training, analysis['signals'], override_verdict)
                
                st.session_state.analysis_results['analysis']['final_decision'] = override_verdict
                st.session_state.analysis_results['analysis']['confidence'] = 100
                st.session_state.analysis_results['analysis']['supporting_reasoning'] = f"User manually overrode the verdict to '{override_verdict}'."
                
                st.success(f"Feedback recorded! Student AI trained (Loss: {loss:.4f}). Verdict updated.")
                st.rerun()

        if not existing_verdict:
            st.divider()
            st.subheader("Blockchain Registration")
            if bc.connected and bc.account:
                if st.button("Register Result to Blockchain"):
                    with st.spinner("⛓️ Recording to Blockchain..."):
                        blockchain_verdict = {
                            "real_image": "Real",
                            "ai_generated": "AI-Generated",
                            "screenshot": "Screenshot"
                        }.get(verdict, verdict)
                        
                        tx_hash = bc.register_verdict(
                            image_hash, 
                            blockchain_verdict,
                            analysis.get("gemini_decision", "N/A"),
                            analysis.get("blocklens_verdict", "N/A"),
                            analysis.get("signals", []),
                            confidence
                        )

                    if tx_hash:
                        st.success("Successfully Registered!")
                        display_hash = tx_hash if tx_hash.startswith('0x') else f'0x{tx_hash}'
                        st.write("**Transaction Hash:**")
                        st.code(display_hash, language="text")
                        st.markdown(f"**Check Transaction:** [Open Sepolia Etherscan](https://sepolia.etherscan.io/) and paste the hash above")
                        st.info("Transaction successfully recorded on blockchain for permanent verification!")
                    else:
                        st.error("Registration failed. Check console logs for details.")
            else:
                st.warning("Cannot register: Wallet not connected or configured.")