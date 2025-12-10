import json
import os
from web3 import Web3
from solcx import compile_standard, install_solc
from dotenv import load_dotenv

# Load env variables
load_dotenv()

RPC_URL = os.getenv("RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

if not RPC_URL or not PRIVATE_KEY:
    print("Error: Missing RPC_URL or PRIVATE_KEY in .env")
    exit(1)

def deploy():
    print("Installing Solidity Compiler...")
    install_solc('0.8.0')

    print("Compiling BlockLens.sol...")
    with open("BlockLens.sol", "r") as file:
        blocklens_file_content = file.read()

    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {"BlockLens.sol": {"content": blocklens_file_content}},
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
                    }
                }
            },
        },
        solc_version="0.8.0",
    )

    bytecode = compiled_sol["contracts"]["BlockLens.sol"]["BlockLensRegistry"]["evm"]["bytecode"]["object"]
    abi = compiled_sol["contracts"]["BlockLens.sol"]["BlockLensRegistry"]["abi"]

    # Save ABI
    with open("abi.json", "w") as f:
        json.dump(abi, f, indent=4)
    print("ABI saved to abi.json")

    # Connect to Blockchain
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print("Failed to connect to blockchain")
        return

    chain_id = w3.eth.chain_id
    my_address = w3.eth.account.from_key(PRIVATE_KEY).address
    print(f"Deploying from address: {my_address}")

    # Build Transaction
    BlockLens = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(my_address)

    transaction = BlockLens.constructor().build_transaction({
        "chainId": chain_id,
        "from": my_address,
        "nonce": nonce,
        "gasPrice": w3.eth.gas_price
    })

    # Sign & Send
    print("Signing transaction...")
    signed_txn = w3.eth.account.sign_transaction(transaction, private_key=PRIVATE_KEY)
    
    print("Sending transaction (Deploying)...")
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    print(f"Transaction Hash: {tx_hash.hex()}")

    # Wait for receipt
    print("Waiting for deployment...")
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print(f"\n✅ Contract Deployed at: {tx_receipt.contractAddress}")
    print("\n👉 Please update your .env file with this new address!")

    # Auto-update .env if possible (optional, but helpful)
    update_env(tx_receipt.contractAddress)

def update_env(new_address):
    env_path = ".env"
    with open(env_path, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.startswith("CONTRACT_ADDRESS="):
            new_lines.append(f"CONTRACT_ADDRESS={new_address}\n")
        else:
            new_lines.append(line)
    
    with open(env_path, "w") as f:
        f.writelines(new_lines)
    print(f"Updated .env with new CONTRACT_ADDRESS: {new_address}")

if __name__ == "__main__":
    deploy()
