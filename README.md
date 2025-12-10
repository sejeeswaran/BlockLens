# BlockLens

BlockLens is an advanced AI-powered image detection and analysis platform that combines multiple machine learning models with blockchain verification for tamper-proof image authenticity records. The system analyzes images to determine if they are authentic photos, AI-generated content, or screenshots, and allows users to register verdicts on the Ethereum blockchain for permanent verification.

## Features

- **Multi-Model AI Analysis**: Combines 6+ pre-trained models including:
  - UMM-Maybe AI Image Detector
  - Facebook DINO Vision Transformer
  - Google Vision Transformer
  - Deep Fake Detector v2
  - Deepfake vs Real Image Detection
  - SDXL Detector

- **Google Gemini Integration**: Advanced visual analysis with Gemini 2.5 Flash for enhanced detection accuracy

- **BlockLens Student AI**: Custom trainable model that learns from user feedback to improve over time

- **Forensic Analysis Tools**:
  - Error Level Analysis (ELA) for compression artifacts
  - Noise variance analysis
  - Metadata examination
  - Screenshot detection heuristics

- **Blockchain Verification**: 
  - Register image verdicts on Ethereum Sepolia testnet
  - Permanent, tamper-proof records
  - Public verification through Etherscan

- **Interactive Web Interface**: Built with Streamlit for easy image upload and analysis

## Installation

### Prerequisites

- Python 3.8+
- Node.js (for contract deployment)
- Git

### Clone the Repository

```bash
git clone <repository-url>
cd BlockLens
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Install Solidity Compiler (for contract deployment)

```bash
npm install -g solc
```

## Setup

### 1. Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Google Gemini API (optional, enhances analysis)
GEMINI_API_KEY=your_gemini_api_key_here

# Blockchain Configuration (for registration features)
RPC_URL=https://ethereum-sepolia-rpc.publicnode.com
CONTRACT_ADDRESS=your_deployed_contract_address
PRIVATE_KEY=your_ethereum_private_key
```

### 2. Deploy Smart Contract (Optional)

If you want blockchain registration functionality:

```bash
# Compile the contract
python setup_compiler.py

# Deploy to Sepolia testnet
python deploy_contract.py
```

Update your `.env` file with the deployed contract address.

### 3. Model Setup

The application will automatically download required models on first run. For faster startup, you can pre-download them:

```bash
python check_models.py
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

### Image Analysis Workflow

1. **Upload Image**: Select a JPG, PNG, or JPEG file
2. **View Hash**: Each image gets a unique SHA-256 hash for blockchain identification
3. **Check Existing Records**: System checks if the image has been analyzed before
4. **Analyze**: Click "Analyze Image" to run the detection pipeline
5. **Review Results**: View the verdict, confidence score, and detailed analysis
6. **Provide Feedback**: If incorrect, override the verdict to train the BlockLens AI
7. **Register on Blockchain**: Permanently record the verdict for public verification

### Analysis Categories

- **Real Image**: Authentic camera/photo images
- **AI Generated**: Synthesized or AI-created content
- **Screenshot**: Screen captures or digital composites

## Architecture

### Core Components

- `app.py`: Main Streamlit application interface
- `BlockLens_ai.py`: Custom trainable AI model
- `blockchain.py`: Web3 integration for Ethereum interaction
- `BlockLens.sol`: Solidity smart contract for verdict storage

### AI Pipeline

1. **Model Ensemble**: Multiple pre-trained models vote on image classification
2. **Gemini Analysis**: Visual inspection for additional evidence
3. **Signal Processing**: ELA, noise, and metadata analysis
4. **BlockLens Prediction**: Custom model combines all signals
5. **Consensus Decision**: Weighted voting with confidence scoring

### Blockchain Integration

- Images are hashed using SHA-256
- Verdicts stored with timestamp and registrar address
- Public verification through contract getters
- Gas-optimized for cost-effective registration

## Testing

### Health Check

```bash
python health_check.py
```

### Manual Testing

```bash
python manual_test.py
```

### Blockchain Tests

```bash
python test_blockchain.py
python verify_blockchain.py
```

## API Reference

### BlockchainManager Class

```python
from blockchain import BlockchainManager

bc = BlockchainManager()
hash = bc.hash_image(image_bytes)
verdict = bc.get_verdict(hash)
tx_hash = bc.register_verdict(hash, status, gemini_verdict, blocklens_verdict, signals, confidence)
```

### BlockLensManager Class

```python
from BlockLens_ai import BlockLensManager

bl = BlockLensManager()
verdict, confidence = bl.predict(image, signals)
loss = bl.train_step(image, signals, correct_label)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Test blockchain interactions on testnet only
- Update requirements.txt for new dependencies

## Security Considerations

- Private keys are stored locally - never commit to version control
- API keys should be kept secure and rotated regularly
- Smart contract interactions include gas limits to prevent excessive costs
- Image hashes prevent duplicate registrations

## License

MIT License - see LICENSE file for details

## Disclaimer

This tool is for educational and research purposes. Always verify critical images through multiple independent methods. AI detection is not 100% accurate and may produce false positives or negatives.
