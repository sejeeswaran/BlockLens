/* BlockLens Frontend Logic */

(function () {
    'use strict';

    // DOM elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const imageHashEl = document.getElementById('image-hash');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const loadingSection = document.getElementById('loading-section');
    const blockchainStatus = document.getElementById('blockchain-status');
    const blockchainContent = document.getElementById('blockchain-content');
    const resultsSection = document.getElementById('results-section');
    const verdictBox = document.getElementById('verdict-box');
    const verdictLabel = document.getElementById('verdict-label');
    const verdictConfidence = document.getElementById('verdict-confidence');
    const reasoningEl = document.getElementById('reasoning');
    const forensicsGrid = document.getElementById('forensics-grid');
    const elaContainer = document.getElementById('ela-container');
    const elaImage = document.getElementById('ela-image');
    const registerSection = document.getElementById('register-section');
    const registerBtn = document.getElementById('register-btn');
    const registerResult = document.getElementById('register-result');

    let currentFile = null;
    let analysisData = null;

    // ===== File handling =====

    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    clearBtn.addEventListener('click', resetUI);

    function handleFile(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            alert('Please upload a JPG, PNG, or JPEG image.');
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            dropzone.style.display = 'none';
            resultsSection.style.display = 'none';
            blockchainStatus.style.display = 'none';

            // Compute hash client-side for preview
            computeHash(file).then((hash) => {
                imageHashEl.textContent = 'Image Hash: ' + hash;
            });
        };
        reader.readAsDataURL(file);
    }

    async function computeHash(file) {
        const buffer = await file.arrayBuffer();
        const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return '0x' + hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
    }

    function resetUI() {
        currentFile = null;
        analysisData = null;
        previewContainer.style.display = 'none';
        dropzone.style.display = 'block';
        resultsSection.style.display = 'none';
        blockchainStatus.style.display = 'none';
        loadingSection.style.display = 'none';
        fileInput.value = '';
    }

    // ===== Analyze =====

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Show loading
        analyzeBtn.disabled = true;
        loadingSection.style.display = 'block';
        resultsSection.style.display = 'none';
        blockchainStatus.style.display = 'none';

        try {
            const formData = new FormData();
            formData.append('image', currentFile);

            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.error || err.trace || 'Analysis failed (HTTP ' + response.status + ')');
            }

            analysisData = await response.json();
            displayResults(analysisData);
            checkBlockchain(analysisData.image_hash);
        } catch (err) {
            alert('Error: ' + err.message);
        } finally {
            analyzeBtn.disabled = false;
            loadingSection.style.display = 'none';
        }
    });

    // ===== Display Results =====

    function displayResults(data) {
        resultsSection.style.display = 'block';

        // Verdict styling
        const verdictMap = {
            real_image: { label: 'Authentic Photo', cssClass: 'real' },
            ai_generated: { label: 'AI-Generated', cssClass: 'ai' },
            screenshot: { label: 'Screenshot Detected', cssClass: 'screenshot' },
            unknown: { label: 'Unknown', cssClass: 'unknown' },
        };

        const v = verdictMap[data.verdict] || verdictMap.unknown;
        verdictBox.className = 'verdict-box ' + v.cssClass;
        verdictLabel.textContent = v.label;
        verdictConfidence.textContent = 'Confidence: ' + data.confidence + '%';

        // Reasoning
        reasoningEl.textContent = data.reasoning || 'No additional details available.';

        // Forensics grid
        const forensics = data.forensics || {};
        forensicsGrid.innerHTML = buildForensicItem('ELA Score', forensics.ela_score != null ? forensics.ela_score.toFixed(2) : 'N/A')
            + buildForensicItem('Noise Score', forensics.noise_score != null ? forensics.noise_score.toFixed(2) : 'N/A')
            + buildForensicItem('Metadata', forensics.metadata_clean ? 'Clean' : 'Edited')
            + buildForensicItem('Software', forensics.software || 'None detected')
            + buildForensicItem('Screenshot Conf.', forensics.screenshot_confidence + '%')
            + buildForensicItem('Gemini', data.gemini_available ? 'Active' : 'Unavailable');

        // ELA image
        if (forensics.ela_image) {
            elaContainer.style.display = 'block';
            elaImage.src = 'data:image/png;base64,' + forensics.ela_image;
        } else {
            elaContainer.style.display = 'none';
        }

        // Show register section
        registerSection.style.display = 'block';
        registerResult.style.display = 'none';

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function buildForensicItem(label, value) {
        return '<div class="forensic-item">'
            + '<div class="forensic-label">' + label + '</div>'
            + '<div class="forensic-value">' + value + '</div>'
            + '</div>';
    }

    // ===== Blockchain Check =====

    async function checkBlockchain(imageHash) {
        try {
            const response = await fetch('/api/blockchain/check', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_hash: imageHash }),
            });

            const data = await response.json();

            if (data.registered) {
                blockchainStatus.style.display = 'block';
                const d = data.data;
                blockchainContent.innerHTML =
                    '<div class="bc-info">'
                    + '<p>⛓️ <strong>Already registered on blockchain!</strong></p>'
                    + '<p><strong>Status:</strong> ' + d.status + '</p>'
                    + '<p><strong>Confidence:</strong> ' + d.confidence + '%</p>'
                    + '<p><strong>Timestamp:</strong> ' + new Date(d.timestamp * 1000).toLocaleString() + '</p>'
                    + '<p><strong>Registrar:</strong></p>'
                    + '<span class="bc-hash">' + d.registrar + '</span>'
                    + '<p style="margin-top:8px"><a href="https://sepolia.etherscan.io/address/' + d.registrar + '" target="_blank" rel="noopener">View on Etherscan →</a></p>'
                    + '</div>';

                // Hide register if already registered
                registerSection.style.display = 'none';
            }
        } catch (_err) {
            // Silently fail — blockchain check is optional
        }
    }

    // ===== Blockchain Register =====

    registerBtn.addEventListener('click', async () => {
        if (!analysisData) return;

        registerBtn.disabled = true;
        registerBtn.textContent = 'Registering...';

        try {
            const response = await fetch('/api/blockchain/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_hash: analysisData.image_hash,
                    verdict: analysisData.verdict,
                    confidence: analysisData.confidence,
                    gemini_verdict: analysisData.gemini_available ? analysisData.verdict : 'N/A',
                    blocklens_verdict: 'N/A',
                    signals: JSON.stringify(analysisData.forensics || {}),
                }),
            });

            const data = await response.json();

            registerResult.style.display = 'block';

            if (data.success) {
                registerResult.className = 'register-success';
                registerResult.innerHTML =
                    '<p><strong>Successfully registered!</strong></p>'
                    + '<p>Transaction Hash:</p>'
                    + '<span class="bc-hash">' + data.tx_hash + '</span>'
                    + '<p style="margin-top:8px"><a href="https://sepolia.etherscan.io/" target="_blank" rel="noopener" style="color: var(--green)">View on Sepolia Etherscan →</a></p>';
                registerBtn.style.display = 'none';
            } else {
                registerResult.className = 'register-error';
                registerResult.textContent = '' + (data.error || 'Registration failed');
            }
        } catch (err) {
            registerResult.style.display = 'block';
            registerResult.className = 'register-error';
            registerResult.textContent = 'Network error: ' + err.message;
        } finally {
            registerBtn.disabled = false;
            registerBtn.innerHTML = '<span class="btn-icon"></span> Register to Blockchain';
        }
    });
})();
