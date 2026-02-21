"""
BlockLens Vercel Serverless API
All non-stdlib imports are lazy-loaded inside route handlers.
"""
import os
import io
import json
import hashlib
import base64
import re
import traceback

ABI_FILENAME = "abi.json"

# Flask is the only third-party top-level import
try:
    from flask import Flask, request, jsonify
    app = Flask(__name__)
except Exception as e:
    # If Flask itself fails, create a minimal WSGI app
    import sys
    print(f"Flask import failed: {e}", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health():
    """Health check — shows which deps are available."""
    status = {"status": "ok", "imports": {}}

    for mod_name in ['numpy', 'PIL', 'cv2', 'exifread', 'google.generativeai', 'web3']:
        try:
            __import__(mod_name)
            status["imports"][mod_name] = "ok"
        except Exception as e:
            status["imports"][mod_name] = f"FAIL: {e}"

    status["env"] = {
        "GEMINI_API_KEY": "set" if os.environ.get("GEMINI_API_KEY") else "missing",
        "RPC_URL": "set" if os.environ.get("RPC_URL") else "missing",
        "CONTRACT_ADDRESS": "set" if os.environ.get("CONTRACT_ADDRESS") else "missing",
        "PRIVATE_KEY": "set" if os.environ.get("PRIVATE_KEY") else "missing",
    }

    return jsonify(status)


# ---------------------------------------------------------------------------
# Gemini analysis
# ---------------------------------------------------------------------------

GEMINI_PROMPT = (
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


def _run_gemini(image_bytes, mime_type="image/jpeg"):
    """Run Gemini analysis."""
    try:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"decision": "unknown", "reasoning": "GEMINI_API_KEY not set"}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        response = model.generate_content([
            GEMINI_PROMPT,
            {"inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(image_bytes).decode()
            }}
        ])
        text = response.text
        # Strip markdown code fences if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip('`').strip()
        result = json.loads(text)
        return {
            "decision": result.get("decision", "unknown"),
            "confidence": result.get("confidence", 50),
            "reasoning": result.get("evidence", "")
        }
    except Exception as exc:
        error_msg = str(exc)
        if "429" in error_msg:
            return {"decision": "unknown", "reasoning": "Quota exceeded. Please wait."}
        return {"decision": "unknown", "reasoning": f"Gemini error: {error_msg}"}


# ---------------------------------------------------------------------------
# Image forensics (all lazy, all safe)
# ---------------------------------------------------------------------------

def _ela_analysis(image):
    """Error Level Analysis."""
    try:
        from PIL import ImageChops, ImageEnhance, Image as PILImage
        import numpy as np

        image = image.convert('RGB')
        buf = io.BytesIO()
        image.save(buf, 'JPEG', quality=90)
        buf.seek(0)
        compressed = PILImage.open(buf)

        ela = ImageChops.difference(image, compressed)
        extrema = ela.getextrema()
        max_diff = max(ex[1] for ex in extrema) or 1
        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)

        ela_arr = np.array(ela)
        avg_diff = float(np.mean(ela_arr))

        out = io.BytesIO()
        ela.save(out, format='PNG')
        ela_b64 = base64.b64encode(out.getvalue()).decode()

        return avg_diff, ela_b64
    except Exception:
        return 0, None


def _noise_analysis(image):
    """Noise analysis with fallbacks."""
    try:
        import numpy as np
        gray = image.convert('L')
        arr = np.array(gray, dtype=float)
        return float(np.var(arr - np.mean(arr)))
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
    """Detect screenshots."""
    confidence = 0
    if noise_score < 5.0:
        confidence = 90
    elif noise_score < 15.0:
        confidence = 70
    if any(kw in software_tag.lower() for kw in ('screenshot', 'snip', 'capture')):
        confidence = 95
    return confidence


# ---------------------------------------------------------------------------
# Blockchain helpers
# ---------------------------------------------------------------------------

def _get_blockchain():
    """Lazy-load blockchain connection."""
    try:
        from web3 import Web3

        rpc_url = os.environ.get("RPC_URL", "https://ethereum-sepolia-rpc.publicnode.com")
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            return None, None, None

        contract = None
        contract_address = os.environ.get("CONTRACT_ADDRESS")
        if contract_address:
            # Try multiple paths for abi.json
            possible_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ABI_FILENAME),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), ABI_FILENAME),
                ABI_FILENAME,
            ]
            for abi_path in possible_paths:
                if os.path.exists(abi_path):
                    with open(abi_path, 'r') as f:
                        contract_abi = json.load(f)
                    checksum = w3.to_checksum_address(contract_address)
                    contract = w3.eth.contract(address=checksum, abi=contract_abi)
                    break

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
# Routes
# ---------------------------------------------------------------------------

@app.route('/api/analyze', methods=['GET', 'POST'])
def analyze():
    """Analyze an uploaded image."""
    if request.method == 'GET':
        return jsonify({"message": "BlockLens API — POST an image to analyze."})

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Empty file"}), 400

        # Open with Pillow
        try:
            from PIL import Image
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except Exception as img_err:
            return jsonify({"error": f"Invalid image: {img_err}"}), 400

        image_hash = "0x" + hashlib.sha256(image_bytes).hexdigest()
        mime_type = file.content_type or "image/jpeg"

        # Run analyses
        gemini_result = _run_gemini(image_bytes, mime_type)
        ela_score, ela_b64 = _ela_analysis(image)
        noise_score = _noise_analysis(image)
        meta_ok, software = _metadata_analysis(image_bytes)
        screenshot_conf = _screenshot_heuristic(noise_score, software)

        # Determine verdict
        if gemini_result["decision"] != "unknown":
            final_decision = gemini_result["decision"]
            confidence = gemini_result.get("confidence", 85)
            reasoning = gemini_result.get("reasoning", "")
        else:
            if screenshot_conf >= 70:
                final_decision = "screenshot"
                confidence = screenshot_conf
                reasoning = "Detected via heuristics (Gemini unavailable)."
            elif not meta_ok:
                final_decision = "ai_generated"
                confidence = 60
                reasoning = f"Editing software: {software} (Gemini unavailable)."
            else:
                final_decision = "real_image"
                confidence = 50
                reasoning = "No manipulation indicators (Gemini unavailable)."

        return jsonify({
            "image_hash": image_hash,
            "verdict": final_decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "gemini_available": gemini_result["decision"] != "unknown",
            "gemini_error": gemini_result.get("reasoning", "") if gemini_result["decision"] == "unknown" else None,
            "forensics": {
                "ela_score": ela_score,
                "ela_image": ela_b64,
                "noise_score": noise_score,
                "metadata_clean": meta_ok,
                "software": software,
                "screenshot_confidence": screenshot_conf
            }
        })

    except Exception:
        return jsonify({"error": "Analysis failed", "trace": traceback.format_exc()}), 500


@app.route('/api/blockchain/check', methods=['POST'])
def blockchain_check():
    """Check if image is registered on-chain."""
    try:
        data = request.get_json()
        if not data or 'image_hash' not in data:
            return jsonify({"error": "Missing image_hash"}), 400

        _w3, contract, _account = _get_blockchain()
        if not contract:
            return jsonify({"registered": False, "error": "Blockchain not connected"})

        result = contract.functions.getVerdict(data['image_hash']).call()
        if result[5] == 0:
            return jsonify({"registered": False})

        return jsonify({
            "registered": True,
            "data": {
                "status": result[0],
                "gemini_verdict": result[1],
                "blocklens_verdict": result[2],
                "supporting_signals": result[3],
                "confidence": result[4],
                "timestamp": result[5],
                "registrar": result[6]
            }
        })
    except Exception as exc:
        return jsonify({"registered": False, "error": str(exc)})


@app.route('/api/blockchain/register', methods=['POST'])
def blockchain_register():
    """Register verdict on blockchain."""
    try:
        data = request.get_json()
        if not data or not all(k in data for k in ('image_hash', 'verdict', 'confidence')):
            return jsonify({"error": "Missing required fields"}), 400

        w3, contract, account = _get_blockchain()
        if not w3 or not contract or not account:
            return jsonify({"error": "Blockchain not fully configured"}), 500

        verdict_map = {"real_image": "Real", "ai_generated": "AI-Generated", "screenshot": "Screenshot"}
        signals = data.get('signals', '[]')
        if isinstance(signals, (dict, list)):
            signals = json.dumps(signals)

        nonce = w3.eth.get_transaction_count(account.address)
        tx = contract.functions.registerVerdict(
            data['image_hash'],
            verdict_map.get(data['verdict'], data['verdict']),
            data.get('gemini_verdict', 'N/A'),
            data.get('blocklens_verdict', 'N/A'),
            signals,
            int(data['confidence'])
        ).build_transaction({
            'chainId': 11155111,
            'gas': 500000,
            'gasPrice': w3.to_wei('30', 'gwei'),
            'nonce': nonce,
        })

        signed_tx = w3.eth.account.sign_transaction(tx, account.key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        hex_hash = tx_hash.hex()

        return jsonify({
            "success": True,
            "tx_hash": hex_hash if hex_hash.startswith('0x') else '0x' + hex_hash
        })
    except Exception as exc:
        return jsonify({"error": f"Transaction failed: {exc}"}), 500
