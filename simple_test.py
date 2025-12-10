from web3 import Web3
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing basic Web3 connection...")

rpc_url = os.getenv("RPC_URL", "https://ethereum-sepolia-rpc.publicnode.com")
print(f"RPC URL: {rpc_url}")

try:
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 10}))
    print(f"Is connected: {w3.is_connected()}")

    if w3.is_connected():
        print(f"Current block: {w3.eth.block_number}")
    else:
        print("Connection failed")

except Exception as e:
    print(f"Error: {e}")