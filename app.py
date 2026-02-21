import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
import io
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
from dotenv import load_dotenv
from blockchain import BlockchainManager
from BlockLens_ai import BlockLensManager

load_dotenv()

LOGO_FILENAME = "blocklens.png"
DIV_CLOSE = "</div>"


def get_secret(key, default=None):
    """Get secret from st.secrets (Streamlit Cloud) or os.getenv (local)."""
    try:
        return st.secrets[key]
    except (KeyError, AttributeError, FileNotFoundError):
        return os.getenv(key, default)


gemini_model = None
if genai:
    gemini_api_key = get_secret("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BlockLens",
    page_icon=LOGO_FILENAME,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS (matches Vercel frontend) ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.85);
    --border-color: rgba(99, 102, 241, 0.2);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.4);
    --green: #10b981;
    --green-bg: rgba(16, 185, 129, 0.15);
    --red: #ef4444;
    --red-bg: rgba(239, 68, 68, 0.15);
    --yellow: #f59e0b;
    --yellow-bg: rgba(245, 158, 11, 0.15);
    --blue: #3b82f6;
    --blue-bg: rgba(59, 130, 246, 0.15);
    --radius: 16px;
    --radius-sm: 10px;
}

/* Global */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Background grid */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(99, 102, 241, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99, 102, 241, 0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}
div[data-testid="stDecoration"] {display: none;}

/* Header */
.app-header {
    text-align: center;
    padding: 2rem 1.5rem 1rem;
    background: linear-gradient(180deg, rgba(99, 102, 241, 0.08) 0%, transparent 100%);
    border-bottom: 1px solid var(--border-color);
    margin: -1rem -1rem 2rem -1rem;
}

.logo-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 8px;
}

.logo-container img {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    filter: drop-shadow(0 0 12px rgba(99, 102, 241, 0.4));
}

.logo-text {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #6366f1, #4f46e5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin: 0;
}

.tagline {
    color: var(--text-secondary);
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 2rem;
    backdrop-filter: blur(20px);
    margin-bottom: 1.5rem;
    transition: border-color 0.3s ease;
    animation: fadeInUp 0.4s ease forwards;
}

.card:hover {
    border-color: rgba(99, 102, 241, 0.35);
}

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Verdict boxes */
.verdict-box {
    padding: 1.5rem;
    border-radius: var(--radius-sm);
    text-align: center;
    margin-bottom: 1rem;
}

.verdict-box.real {
    background: var(--green-bg);
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.verdict-box.ai {
    background: var(--red-bg);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.verdict-box.screenshot {
    background: var(--yellow-bg);
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.verdict-box.unknown {
    background: var(--blue-bg);
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.verdict-label {
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 4px;
}

.verdict-box.real .verdict-label { color: var(--green); }
.verdict-box.ai .verdict-label { color: var(--red); }
.verdict-box.screenshot .verdict-label { color: var(--yellow); }
.verdict-box.unknown .verdict-label { color: var(--blue); }

.verdict-confidence {
    font-size: 0.95rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* Reasoning */
.reasoning-box {
    padding: 1rem 1.25rem;
    background: rgba(99, 102, 241, 0.06);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-bottom: 1.25rem;
    line-height: 1.7;
}

/* Forensics grid */
.forensics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-top: 12px;
}

.forensic-item {
    padding: 14px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
}

.forensic-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-secondary);
    font-weight: 600;
    margin-bottom: 6px;
}

.forensic-value {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* Image hash */
.image-hash {
    margin: 1rem 0;
    padding: 10px 16px;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
    word-break: break-all;
    text-align: center;
}

/* Preview image */
.preview-img {
    display: block;
    max-width: 100%;
    max-height: 400px;
    margin: 0 auto;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-color);
    object-fit: contain;
}

/* Register section */
.register-section {
    padding-top: 1.25rem;
    border-top: 1px solid var(--border-color);
}

.register-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 6px;
    color: var(--text-primary);
}

.register-desc {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-bottom: 1rem;
}

/* Blockchain info */
.bc-info {
    padding: 1rem;
    background: var(--blue-bg);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: var(--radius-sm);
    margin-bottom: 1rem;
}

.bc-info p {
    margin: 6px 0;
    font-size: 0.88rem;
    color: var(--text-secondary);
}

.bc-info strong { color: var(--text-primary); }

.bc-hash {
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    word-break: break-all;
    padding: 8px 12px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 6px;
    margin-top: 6px;
    display: block;
    color: var(--text-secondary);
}

/* Success / Error boxes */
.register-success {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: var(--radius-sm);
    background: var(--green-bg);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: var(--green);
    font-size: 0.85rem;
}

.register-error {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: var(--radius-sm);
    background: var(--red-bg);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: var(--red);
    font-size: 0.85rem;
}

/* ELA container */
.ela-section h4 {
    font-size: 0.9rem;
    margin-bottom: 8px;
    color: var(--text-secondary);
}

.ela-section img {
    max-width: 100%;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-color);
}

/* Button overrides */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1.75rem !important;
    transition: all 0.25s ease !important;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99, 102, 241, 0.35) !important;
    border-radius: var(--radius-sm) !important;
    padding: 2rem !important;
    background: transparent !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: rgba(99, 102, 241, 0.06) !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: var(--text-primary) !important;
    background: transparent !important;
}

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Feedback buttons */
.feedback-section {
    padding-top: 1rem;
    border-top: 1px solid var(--border-color);
    margin-top: 1rem;
}

.feedback-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
}

.feedback-desc {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Cached Resources ─────────────────────────────────────────────────────────
@st.cache_resource
def get_blockchain_manager():
    return BlockchainManager()

bc = get_blockchain_manager()

@st.cache_resource
def get_blocklens_manager():
    return BlockLensManager()

blocklens_manager = get_blocklens_manager()

try:
    from transformers import pipeline as hf_pipeline

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
                models[model_name] = hf_pipeline(task, model=model_name)
            except Exception:
                models[model_name] = None
        return models

    pipes = load_models()
except ImportError:
    pipes = {}


# ─── Analysis Functions ────────────────────────────────────────────────────────
def ela_analysis(image):
    try:
        image = image.convert('RGB')
        temp = io.BytesIO()
        image.save(temp, 'JPEG', quality=90)
        temp.seek(0)
        compressed = Image.open(temp)
        ela = ImageChops.difference(image, compressed)
        extrema = ela.getextrema()
        max_diff = max(ex[1] for ex in extrema) or 1
        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)
        ela_array = np.array(ela)
        avg_diff = float(np.mean(ela_array))
        return avg_diff, ela
    except Exception:
        return 0, None


def noise_analysis(image):
    try:
        gray = image.convert('L')
        arr = np.array(gray, dtype=float)
        return float(np.var(arr - np.mean(arr)))
    except Exception:
        return 0


def metadata_analysis(image_file):
    try:
        image_file.seek(0)
        tags = exifread.process_file(image_file)
        software = str(tags.get('Image Software', ''))
        editing_tools = ('photoshop', 'ai', 'gimp')
        is_edited = any(tool in software.lower() for tool in editing_tools)
        return not is_edited, software
    except Exception:
        return True, "Unknown"


def detect_screenshot_heuristic(noise_score, software_tag):
    confidence = 0
    if noise_score < 5.0:
        confidence = 90
    elif noise_score < 15.0:
        confidence = 70
    if any(kw in software_tag.lower() for kw in ('screenshot', 'snip', 'capture')):
        confidence = 95
    return confidence


def _run_model_ensemble(image):
    model_opinions = {}
    votes = []
    for model_name, pipe in pipes.items():
        if not pipe:
            model_opinions[model_name] = {"decision": "unknown", "reasoning": "Model not loaded"}
            continue
        try:
            results = pipe(image)
            top_result = results[0]
            label = top_result['label']
            score = top_result['score']
            decision = "real_image" if 'real' in label.lower() or 'authentic' in label.lower() else "ai_generated"
            votes.append(decision)
            reasoning = f"Predicted '{label}' with confidence {score:.2f}"
            model_opinions[model_name] = {"decision": decision, "reasoning": reasoning}
        except Exception:
            model_opinions[model_name] = {"decision": "unknown", "reasoning": "Model failed"}
    return model_opinions, votes


def _run_gemini_analysis(image_file, model_opinions, votes):
    if not gemini_model:
        return
    try:
        image_bytes = image_file.getvalue()
        mime_type = image_file.type if hasattr(image_file, 'type') else "image/jpeg"
        gemini_prompt = (
            'Analyze this image and classify it as ONE of these 3 categories ONLY:\n'
            '**real_image** - Authentic camera photo (phone/camera taken)\n'
            '**ai_generated** - AI-created/synthesized image\n'
            '**screenshot** - Screen capture/digital composite\n\n'
            'LOOK FOR THESE CLUES:\n'
            '- **Screenshot**: UI elements, perfect edges, compression blocks, browser chrome\n'
            '- **AI Generated**: Anatomical errors, symmetrical artifacts, unnatural lighting\n'
            '- **Real Photo**: Natural noise/grain, lens distortion, organic lighting\n\n'
            'OUTPUT EXACTLY:\n'
            '{\n  "decision": "real_image" | "ai_generated" | "screenshot",\n'
            '  "confidence": 85,\n'
            '  "evidence": "2-3 specific visual clues you saw"\n}\n\n'
            'NEVER say "uncertain" - pick your best guess with realistic confidence.'
        )
        response = gemini_model.generate_content([
            gemini_prompt,
            {"inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(image_bytes).decode()
            }}
        ])
        text = response.text
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip('`').strip()
        result = json.loads(text)
        model_opinions["Gemini"] = {"decision": result["decision"], "reasoning": result.get("evidence", "")}
        votes.append(result["decision"])
    except Exception as exc:
        error_msg = str(exc)
        if "429" in error_msg:
            model_opinions["Gemini"] = {
                "decision": "unknown",
                "reasoning": "Daily AI Quota Exceeded. Please wait."
            }
        else:
            model_opinions["Gemini"] = {
                "decision": "unknown",
                "reasoning": f"Gemini failed: {error_msg}"
            }


def _determine_final_verdict(model_opinions, votes, image, signals):
    gemini_result = model_opinions.get("Gemini", {})
    gemini_decision = gemini_result.get("decision", "unknown")

    if gemini_decision != "unknown":
        final_decision = gemini_decision
        confidence = 100
        supporting_reasoning = gemini_result.get('reasoning', '')
        loss = blocklens_manager.train_step(image, signals, gemini_decision)
        if loss:
            print(f"BlockLens Model trained. Loss: {loss:.4f}")
    elif votes:
        vote_counts = Counter(votes)
        most_common = vote_counts.most_common(1)[0]
        final_decision = most_common[0]
        confidence = int((most_common[1] / len(votes)) * 100)
        supporting_reasoning = f"Gemini unavailable. Consensus reached: {final_decision}."
    else:
        final_decision = "unknown"
        confidence = 0
        supporting_reasoning = "Insufficient data."

    return final_decision, confidence, supporting_reasoning, gemini_decision


def analyze_image(image_file):
    image_file.seek(0)
    image = Image.open(image_file)

    model_opinions, votes = _run_model_ensemble(image)
    _run_gemini_analysis(image_file, model_opinions, votes)

    ela_score, ela_image = ela_analysis(image)
    noise_score = noise_analysis(image)
    meta_ok, software = metadata_analysis(image_file)
    screenshot_conf = detect_screenshot_heuristic(noise_score, software)

    ai_probs = [1.0 if m["decision"] == "ai_generated" else 0.0
                for m in model_opinions.values() if m["decision"] in ("ai_generated", "real_image")]
    avg_ai_prob = sum(ai_probs) / len(ai_probs) if ai_probs else 0.5
    meta_score = 1.0 if not meta_ok else 0.0
    signals = [ela_score, noise_score, screenshot_conf / 100.0, avg_ai_prob, meta_score]

    blocklens_verdict, blocklens_conf = blocklens_manager.predict(image, signals)
    final_decision, confidence, supporting_reasoning, gemini_decision = _determine_final_verdict(
        model_opinions, votes, image, signals
    )

    return {
        "final_decision": final_decision,
        "confidence": confidence,
        "supporting_reasoning": supporting_reasoning,
        "model_opinions": model_opinions,
        "blocklens_verdict": blocklens_verdict,
        "blocklens_confidence": blocklens_conf,
        "signals": signals,
        "gemini_decision": gemini_decision,
        "forensics": {
            "ela_score": ela_score,
            "noise_score": noise_score,
            "metadata_clean": meta_ok,
            "software": software,
            "screenshot_confidence": screenshot_conf,
        }
    }, ela_image


# ─── Helper: encode image to base64 for HTML ──────────────────────────────────
def img_to_b64(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ─── UI RENDERING ─────────────────────────────────────────────────────────────

# Header
logo_b64 = ""
if os.path.exists(LOGO_FILENAME):
    with open(LOGO_FILENAME, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div class="app-header">
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_b64}" alt="BlockLens">
        <h1 class="logo-text">BlockLens</h1>
    </div>
    <p class="tagline">AI Image Detection & Blockchain Verification</p>
</div>
""", unsafe_allow_html=True)

# ─── Upload Section ────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="section-title">Analyze an Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    # Preview image
    image_bytes = uploaded_file.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode()
    mime = uploaded_file.type or "image/jpeg"

    st.markdown(f'<img src="data:{mime};base64,{image_b64}" class="preview-img" alt="Preview">', unsafe_allow_html=True)

    # Hash
    image_hash = "0x" + hashlib.sha256(image_bytes).hexdigest()
    st.markdown(f'<div class="image-hash">Image Hash: {image_hash}</div>', unsafe_allow_html=True)

    # Buttons
    col1, col2 = st.columns([1, 1])
    analyze_clicked = col1.button("Analyze Image", use_container_width=True, type="primary")
    clear_clicked = col2.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.pop('analysis_results', None)
        st.session_state.pop('last_uploaded_file', None)
        st.rerun()

    if analyze_clicked:
        with st.spinner("Analyzing image with AI..."):
            analysis_result, ela_image = analyze_image(uploaded_file)
            st.session_state.analysis_results = {
                "analysis": analysis_result,
                "ela_image": ela_image,
                "image_hash": image_hash,
            }
            st.session_state.last_uploaded_file = uploaded_file.name

st.markdown(DIV_CLOSE, unsafe_allow_html=True)

# ─── Blockchain Status (existing registration) ────────────────────────────────
if uploaded_file is not None:
    image_bytes_check = uploaded_file.getvalue()
    image_hash_check = "0x" + hashlib.sha256(image_bytes_check).hexdigest()
    existing_verdict = bc.get_verdict(image_hash_check)

    if existing_verdict:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Blockchain Status</div>', unsafe_allow_html=True)

        from datetime import datetime
        ts = existing_verdict.get('timestamp', 0)
        ts_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else "N/A"

        st.markdown(f"""
        <div class="bc-info">
            <p><strong>Already registered on blockchain!</strong></p>
            <p><strong>Status:</strong> {existing_verdict['status']}</p>
            <p><strong>Confidence:</strong> {existing_verdict['confidence']}%</p>
            <p><strong>Timestamp:</strong> {ts_str}</p>
            <p><strong>Registrar:</strong></p>
            <span class="bc-hash">{existing_verdict['registrar']}</span>
            <p style="margin-top:8px"><a href="https://sepolia.etherscan.io/address/{existing_verdict['registrar']}" target="_blank" style="color: var(--accent); text-decoration: none;">View on Etherscan →</a></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(DIV_CLOSE, unsafe_allow_html=True)

# ─── Results Section ───────────────────────────────────────────────────────────
if st.session_state.get('analysis_results'):
    results = st.session_state.analysis_results
    analysis = results["analysis"]
    verdict = analysis["final_decision"]
    confidence = analysis["confidence"]
    reasoning = analysis["supporting_reasoning"]
    forensics = analysis.get("forensics", {})
    ela_image = results.get("ela_image")
    image_hash = results.get("image_hash", "")

    # Verdict mapping
    verdict_map = {
        "real_image": ("Authentic Photo", "real"),
        "ai_generated": ("AI-Generated", "ai"),
        "screenshot": ("Screenshot Detected", "screenshot"),
    }
    v_label, v_class = verdict_map.get(verdict, ("Unknown", "unknown"))

    st.markdown(f"""
    <div class="card">
        <div class="section-title">Analysis Results</div>
        <div class="verdict-box {v_class}">
            <div class="verdict-label">{v_label}</div>
            <div class="verdict-confidence">Confidence: {confidence}%</div>
        </div>
        <div class="reasoning-box">{reasoning}</div>
    """, unsafe_allow_html=True)

    # Forensics
    with st.expander("Forensic Analysis Details"):
        ela_score = forensics.get("ela_score", 0)
        noise_score = forensics.get("noise_score", 0)
        meta_clean = forensics.get("metadata_clean", True)
        software = forensics.get("software", "None")
        screenshot_conf = forensics.get("screenshot_confidence", 0)
        gemini_status = "Active" if analysis.get("gemini_decision", "unknown") != "unknown" else "Unavailable"

        st.markdown(f"""
        <div class="forensics-grid">
            <div class="forensic-item">
                <div class="forensic-label">ELA Score</div>
                <div class="forensic-value">{ela_score:.2f}</div>
            </div>
            <div class="forensic-item">
                <div class="forensic-label">Noise Score</div>
                <div class="forensic-value">{noise_score:.2f}</div>
            </div>
            <div class="forensic-item">
                <div class="forensic-label">Metadata</div>
                <div class="forensic-value">{"Clean" if meta_clean else "Edited"}</div>
            </div>
            <div class="forensic-item">
                <div class="forensic-label">Software</div>
                <div class="forensic-value">{software or "None detected"}</div>
            </div>
            <div class="forensic-item">
                <div class="forensic-label">Screenshot Conf.</div>
                <div class="forensic-value">{screenshot_conf}%</div>
            </div>
            <div class="forensic-item">
                <div class="forensic-label">Gemini</div>
                <div class="forensic-value">{gemini_status}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ELA Image
        if ela_image:
            st.markdown('<div class="ela-section"><h4>Error Level Analysis (ELA)</h4></div>', unsafe_allow_html=True)
            st.image(ela_image, use_container_width=True)

        # Model Opinions
        if analysis.get("model_opinions"):
            st.markdown("**Model Opinions:**")
            for model, opinion in analysis["model_opinions"].items():
                st.markdown(f"- **{model}:** {opinion['decision']} — {opinion['reasoning']}")

    # ─── Feedback Section ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="feedback-section">
        <div class="feedback-title">Incorrect Verdict? Provide Feedback</div>
        <div class="feedback-desc">Help improve the BlockLens Student AI by providing the correct label.</div>
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    override_verdict = None

    if col_f1.button("Real Photo"):
        override_verdict = "real_image"
    if col_f2.button("AI Generated"):
        override_verdict = "ai_generated"
    if col_f3.button("Screenshot"):
        override_verdict = "screenshot"

    if override_verdict:
        with st.spinner(f"Retraining Student AI with correction: {override_verdict}..."):
            uploaded_file.seek(0)
            image_for_training = Image.open(uploaded_file)
            loss = blocklens_manager.train_step(image_for_training, analysis['signals'], override_verdict)

            st.session_state.analysis_results['analysis']['final_decision'] = override_verdict
            st.session_state.analysis_results['analysis']['confidence'] = 100
            st.session_state.analysis_results['analysis']['supporting_reasoning'] = f"User manually overrode the verdict to '{override_verdict}'."

            if loss:
                st.markdown(f"""
                <div class="register-success">
                    <strong>Feedback recorded!</strong> Student AI trained (Loss: {loss:.4f}). Verdict updated.
                </div>
                """, unsafe_allow_html=True)
            st.rerun()

    # ─── Blockchain Registration ───────────────────────────────────────────────
    if uploaded_file is not None:
        existing = bc.get_verdict(image_hash) if image_hash else None
        if not existing:
            st.markdown("""
            <div class="register-section">
                <div class="register-title">Register on Blockchain</div>
                <div class="register-desc">Permanently record this verdict on Ethereum Sepolia for public verification.</div>
            </div>
            """, unsafe_allow_html=True)

            if bc.connected and bc.account:
                if st.button("Register to Blockchain", type="primary"):
                    with st.spinner("Recording to Blockchain..."):
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
                        display_hash = tx_hash if tx_hash.startswith('0x') else f'0x{tx_hash}'
                        st.markdown("""
                        <div class="register-success" style="padding-bottom: 0.5rem; border-bottom: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0;">
                            <p><strong>Successfully registered!</strong></p>
                            <p style="margin-top:8px">Transaction Hash:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.code(display_hash, language="text")
                        
                        st.markdown(f"""
                        <div class="register-success" style="padding-top: 0.5rem; border-top: none; border-top-left-radius: 0; border-top-right-radius: 0; margin-top: -1rem;">
                            <p><a href="https://sepolia.etherscan.io/tx/{display_hash}" target="_blank" style="color: var(--green); text-decoration: none;">View on Sepolia Etherscan →</a></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="register-error">Registration failed. Check console logs for details.</div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="register-error">Cannot register: Wallet not connected or configured.</div>
                """, unsafe_allow_html=True)

    st.markdown(DIV_CLOSE, unsafe_allow_html=True)