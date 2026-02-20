import os
import io
import json
import hashlib
import base64
import re

from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)


def _get_gemini_model():
    """Lazy-load Gemini model to avoid cold-start issues."""
    try:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception:
        return None


def _get_blockchain_manager():
    """Lazy-load blockchain manager."""
    try:
        from web3 import Web3

        rpc_url = os.environ.get("RPC_URL", "https://ethereum-sepolia-rpc.publicnode.com")
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            return None, None, None

        contract = None
        contract_address = os.environ.get("CONTRACT_ADDRESS")
        if contract_address:
            abi_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "abi.json")
            if os.path.exists(abi_path):
                with open(abi_path, 'r') as f:
                    contract_abi = json.load(f)
                checksum = w3.to_checksum_address(contract_address)
                contract = w3.eth.contract(address=checksum, abi=contract_abi)

        account = None
        private_key = os.environ.get("PRIVATE_KEY")
        if private_key:
            if not private_key.startswith("0x"):
                private_key = "0x" + private_key
            account = w3.eth.account.from_key(private_key)

        return w3, contract, account
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Image forensics (lightweight, no ML models needed)
# ---------------------------------------------------------------------------

def _ela_analysis(image):
    """Error Level Analysis using Pillow only."""
    try:
        from PIL import ImageChops, ImageEnhance

        image = image.convert('RGB')
        temp = io.BytesIO()
        image.save(temp, 'JPEG', quality=90)
        temp.seek(0)
        from PIL import Image as PILImage
        compressed = PILImage.open(temp)

        ela = ImageChops.difference(image, compressed)
        extrema = ela.getextrema()
        max_diff = max(ex[1] for ex in extrema) or 1
        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)

        ela_array = np.array(ela)
        avg_diff = float(np.mean(ela_array))

        # Convert ELA image to base64 for frontend
        buf = io.BytesIO()
        ela.save(buf, format='PNG')
        ela_b64 = base64.b64encode(buf.getvalue()).decode()

        return avg_diff, ela_b64
    except Exception:
        return 0, None


def _noise_analysis(image):
    """Noise variance analysis using OpenCV."""
    try:
        import cv2
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        noise = cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0))
        return float(np.var(noise))
    except Exception:
        return 0


def _metadata_analysis(image_bytes):
    """EXIF metadata analysis."""
    try:
        import exifread
        tags = exifread.process_file(io.BytesIO(image_bytes))
        software = str(tags.get('Image Software', ''))
        editing_tools = ('photoshop', 'ai', 'gimp')
        is_edited = any(tool in software.lower() for tool in editing_tools)
        return not is_edited, software
    except Exception:
        return True, "Unknown"


def _screenshot_heuristic(noise_score, software_tag):
    """Detect if image is likely a screenshot."""
    confidence = 0
    if noise_score < 5.0:
        confidence = 90
    elif noise_score < 15.0:
        confidence = 70

    software_lower = software_tag.lower()
    if any(kw in software_lower for kw in ('screenshot', 'snip', 'capture')):
        confidence = 95

    return confidence


# ---------------------------------------------------------------------------
# Gemini analysis
# ---------------------------------------------------------------------------

GEMINI_PROMPT = (
    'Analyze this image and classify it as ONE of these 3 categories ONLY:\n'
    '**real_image** - Authentic camera photo (phone/camera taken)\n'
    '**ai_generated** - AI-created/synthesized image\n'
    '**screenshot** - Screen capture/digital composite\n\n'
    'LOOK FOR THESE CLUES:\n'
    '- **Screenshot**: UI elements, perfect edges, compression blocks, browser chrome, low noise variance\n'
    '- **AI Generated**: Anatomical errors (extra fingers, weird hands), symmetrical artifacts, '
    'unnatural lighting/shadows, blurry text/logos\n'
    '- **Real Photo**: Natural noise/grain, lens distortion, organic lighting, camera sensor artifacts\n\n'
    'OUTPUT EXACTLY:\n'
    '{\n  "decision": "real_image" | "ai_generated" | "screenshot",\n'
    '  "confidence": 85,\n'
    '  "evidence": "2-3 specific visual clues you saw"\n}\n\n'
    'NEVER say "uncertain" - pick your best guess with realistic confidence.'
)


def _run_gemini(image_bytes, mime_type="image/jpeg"):
    """Run Gemini analysis on image bytes."""
    model = _get_gemini_model()
    if not model:
        return {"decision": "unknown", "reasoning": "Gemini not configured"}

    try:
        response = model.generate_content([
            GEMINI_PROMPT,
            {"inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(image_bytes).decode()
            }}
        ])
        text = re.sub(r'```json\s*', '', response.text).strip('`').strip()
        result = json.loads(text)
        return {
            "decision": result.get("decision", "unknown"),
            "confidence": result.get("confidence", 50),
            "reasoning": result.get("evidence", "")
        }
    except Exception as exc:
        error_msg = str(exc)
        if "429" in error_msg:
            return {
                "decision": "unknown",
                "reasoning": "Daily AI Quota Exceeded. Please wait or check billing."
            }
        return {"decision": "unknown", "reasoning": f"Gemini error: {error_msg}"}


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze an uploaded image."""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_bytes = file.read()

    if not image_bytes:
        return jsonify({"error": "Empty file"}), 400

    try:
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # Compute image hash
    image_hash = "0x" + hashlib.sha256(image_bytes).hexdigest()

    # Determine mime type
    mime_type = file.content_type or "image/jpeg"

    # Run all analyses
    gemini_result = _run_gemini(image_bytes, mime_type)
    ela_score, ela_image_b64 = _ela_analysis(image)
    noise_score = _noise_analysis(image)
    meta_ok, software = _metadata_analysis(image_bytes)
    screenshot_conf = _screenshot_heuristic(noise_score, software)

    # Determine final verdict
    if gemini_result["decision"] != "unknown":
        final_decision = gemini_result["decision"]
        confidence = gemini_result.get("confidence", 85)
        reasoning = gemini_result.get("reasoning", "")
    else:
        # Fallback to forensic heuristics
        if screenshot_conf >= 70:
            final_decision = "screenshot"
            confidence = screenshot_conf
            reasoning = "Detected via noise/metadata heuristics (Gemini unavailable)."
        elif not meta_ok:
            final_decision = "ai_generated"
            confidence = 60
            reasoning = f"Editing software detected: {software} (Gemini unavailable)."
        else:
            final_decision = "real_image"
            confidence = 50
            reasoning = "No manipulation indicators found (Gemini unavailable)."

    return jsonify({
        "image_hash": image_hash,
        "verdict": final_decision,
        "confidence": confidence,
        "reasoning": reasoning,
        "gemini_available": gemini_result["decision"] != "unknown",
        "forensics": {
            "ela_score": ela_score,
            "ela_image": ela_image_b64,
            "noise_score": noise_score,
            "metadata_clean": meta_ok,
            "software": software,
            "screenshot_confidence": screenshot_conf
        }
    })


@app.route('/api/blockchain/check', methods=['POST'])
def blockchain_check():
    """Check if an image hash has already been registered."""
    data = request.get_json()
    if not data or 'image_hash' not in data:
        return jsonify({"error": "Missing image_hash"}), 400

    image_hash = data['image_hash']
    _w3, contract, _account = _get_blockchain_manager()

    if not contract:
        return jsonify({"registered": False, "error": "Blockchain not connected"})

    try:
        result = contract.functions.getVerdict(image_hash).call()
        timestamp = result[5]
        if timestamp == 0:
            return jsonify({"registered": False})

        return jsonify({
            "registered": True,
            "data": {
                "status": result[0],
                "gemini_verdict": result[1],
                "blocklens_verdict": result[2],
                "supporting_signals": result[3],
                "confidence": result[4],
                "timestamp": timestamp,
                "registrar": result[6]
            }
        })
    except Exception as exc:
        return jsonify({"registered": False, "error": str(exc)})


@app.route('/api/blockchain/register', methods=['POST'])
def blockchain_register():
    """Register a verdict on the blockchain."""
    data = request.get_json()
    required = ('image_hash', 'verdict', 'confidence')
    if not data or not all(k in data for k in required):
        return jsonify({"error": "Missing required fields"}), 400

    w3, contract, account = _get_blockchain_manager()

    if not w3 or not contract or not account:
        return jsonify({"error": "Blockchain not fully configured"}), 500

    try:
        image_hash = data['image_hash']
        verdict_map = {
            "real_image": "Real",
            "ai_generated": "AI-Generated",
            "screenshot": "Screenshot"
        }
        blockchain_verdict = verdict_map.get(data['verdict'], data['verdict'])
        gemini_verdict = data.get('gemini_verdict', 'N/A')
        blocklens_verdict = data.get('blocklens_verdict', 'N/A')
        signals = data.get('signals', '[]')
        if isinstance(signals, (dict, list)):
            signals = json.dumps(signals)
        confidence = int(data['confidence'])

        nonce = w3.eth.get_transaction_count(account.address)
        tx = contract.functions.registerVerdict(
            image_hash,
            blockchain_verdict,
            gemini_verdict,
            blocklens_verdict,
            signals,
            confidence
        ).build_transaction({
            'chainId': 11155111,
            'gas': 500000,
            'gasPrice': w3.to_wei('30', 'gwei'),
            'nonce': nonce,
        })

        signed_tx = w3.eth.account.sign_transaction(tx, account.key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

        return jsonify({
            "success": True,
            "tx_hash": "0x" + tx_hash.hex() if not tx_hash.hex().startswith('0x') else tx_hash.hex()
        })
    except Exception as exc:
        return jsonify({"error": f"Transaction failed: {exc}"}), 500


# For local development
if __name__ == '__main__':
    app.run(debug=True, port=5000)
