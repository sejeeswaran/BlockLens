import os
import json
import hashlib
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

class BlockchainManager:
    def __init__(self):
        self.w3 = None
        self.contract = None
        self.account = None
        self.connected = False
        self.setup_connection()

    def setup_connection(self):
        try:
            rpc_url = os.getenv("RPC_URL", "https://ethereum-sepolia-rpc.publicnode.com")
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            if not self.w3.is_connected():
                print("❌ Failed to connect to RPC.")
                return

            self.connected = True
            contract_address = os.getenv("CONTRACT_ADDRESS")

            if contract_address:
                try:
                    with open('abi.json', 'r') as f:
                        contract_abi = json.load(f)
                    checksum_address = self.w3.to_checksum_address(contract_address)
                    self.contract = self.w3.eth.contract(address=checksum_address, abi=contract_abi)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"⚠️ Error loading contract: {e}")
            
            private_key = os.getenv("PRIVATE_KEY")
            if private_key:
                if not private_key.startswith("0x"):
                    private_key = "0x" + private_key
                self.account = self.w3.eth.account.from_key(private_key)

        except Exception as e:
            print(f"❌ Blockchain setup failed: {e}")

    def hash_image(self, image_bytes):
        hash_sha256 = hashlib.sha256(image_bytes)
        return "0x" + hash_sha256.hexdigest()

    def get_verdict(self, image_hash):
        if not self.contract:
            return None
        
        try:
            result = self.contract.functions.getVerdict(image_hash).call()
            timestamp = result[5]
            if timestamp == 0:
                return None
            
            return {
                "status": result[0],
                "gemini_verdict": result[1],
                "blocklens_verdict": result[2],
                "supporting_signals": result[3],
                "confidence": result[4],
                "timestamp": timestamp,
                "registrar": result[6]
            }
        except Exception as e:
            print(f"Error fetching verdict: {e}")
            return None

    def register_verdict(self, image_hash, verdict, gemini_verdict, blocklens_verdict, supporting_signals, confidence):
        if not self.connected or not self.contract or not self.account:
            print("Cannot register: Blockchain not fully configured.")
            return None

        try:
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            if isinstance(supporting_signals, (dict, list)):
                supporting_signals = json.dumps(supporting_signals)

            tx = self.contract.functions.registerVerdict(
                image_hash,
                verdict,
                gemini_verdict,
                blocklens_verdict,
                supporting_signals,
                int(confidence)
            ).build_transaction({
                'chainId': 11155111,
                'gas': 500000, 
                'gasPrice': self.w3.to_wei('30', 'gwei'), 
                'nonce': nonce,
            })

            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)

            return tx_hash.hex()

        except Exception as e:
            print(f"Transaction failed: {e}")
            return None
