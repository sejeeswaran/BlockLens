from web3 import Web3
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing RPC connection...")

rpc_url = os.getenv("RPC_URL")
print(f"RPC URL: {rpc_url}")

if not rpc_url:
    print("ERROR: RPC_URL not found in .env")
    exit(1)

try:
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 10}))
    is_connected = w3.is_connected()
    print(f"Connected: {is_connected}")

    if is_connected:
        block_number = w3.eth.block_number
        print(f"Current block: {block_number}")
        print("SUCCESS: RPC is working correctly")
    else:
        print("ERROR: Cannot connect to RPC")

except Exception as e:
    print(f"ERROR: {e}")