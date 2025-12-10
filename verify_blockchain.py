from web3 import Web3
import os
import json
from dotenv import load_dotenv

load_dotenv()

def verify_blockchain():
    print("--- Blockchain Verification ---")

    # 1. Check RPC Connection
    rpc_urls = [
        "https://sepolia.infura.io/v3/cd5402720dc948d888f977c9de2a97ec"
    ]

    connected = False
    for url in rpc_urls:
        print(f"Testing RPC: {url}...")
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 5}))
            if w3.is_connected():
                print(f"[+] Connected to Sepolia RPC: {url}")
                print(f"   Current Block: {w3.eth.block_number}")
                connected = True
                break
        except Exception as e:
            print(f"   Failed: {e}")

    if not connected:
        print("[-] Failed to connect to any RPC endpoint.")
        return

    # 2. Check Contract Configuration
    address = os.getenv("CONTRACT_ADDRESS")

    if not address:
        print("X CONTRACT_ADDRESS is missing in .env")
    else:
        print(f"+ CONTRACT_ADDRESS found: {address}")

    # Check ABI file
    try:
        with open('abi.json', 'r') as f:
            abi = json.load(f)
        print(f"+ ABI loaded successfully from abi.json (Length: {len(str(abi))})")

        if address:
            contract = w3.eth.contract(address=address, abi=abi)
            print("+ Contract object created successfully")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"X ABI error: {e}")

    # 3. Check Private Key
    pk = os.getenv("PRIVATE_KEY")
    if not pk:
        print("X PRIVATE_KEY is missing in .env")
    else:
        try:
            if not pk.startswith("0x"):
                pk = "0x" + pk
            account = w3.eth.account.from_key(pk)
            print(f"[+] Private Key is valid. Address: {account.address}")

            # Check Balance
            balance = w3.eth.get_balance(account.address)
            eth_balance = w3.from_wei(balance, 'ether')
            print(f"   Balance: {eth_balance} ETH")
            if balance == 0:
                print("[!] Warning: Wallet balance is 0 ETH. Transactions will fail.")
        except Exception as e:
            print(f"X Private Key is invalid: {e}")

if __name__ == "__main__":
    verify_blockchain()
