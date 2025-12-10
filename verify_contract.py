from web3 import Web3
import os
from dotenv import load_dotenv
import json

load_dotenv()

print("Verifying contract deployment...")

rpc_url = os.getenv("RPC_URL")
contract_address = os.getenv("CONTRACT_ADDRESS")

print(f"RPC: {rpc_url}")
print(f"Contract Address: {contract_address}")

w3 = Web3(Web3.HTTPProvider(rpc_url))

if not w3.is_connected():
    print("ERROR: Cannot connect to RPC")
    exit(1)

# Check if contract address is valid
try:
    checksum_address = w3.to_checksum_address(contract_address)
    print(f"Checksum address: {checksum_address}")
except Exception as e:
    print(f"ERROR: Invalid contract address: {e}")
    exit(1)

# Check contract code
code = w3.eth.get_code(checksum_address)
if code == b'\x00' or code == '0x':
    print("ERROR: No contract code at this address")
    exit(1)

print(f"Contract code length: {len(code)} bytes")
print("SUCCESS: Contract is deployed at this address")

# Load ABI and test contract interaction
try:
    with open('abi.json', 'r') as f:
        abi = json.load(f)

    contract = w3.eth.contract(address=checksum_address, abi=abi)
    print("Contract object created successfully")

    # Test a call
    test_hash = "0x" + "0" * 64
    result = contract.functions.getVerdict(test_hash).call()
    print(f"Test call result: {result}")

except Exception as e:
    print(f"ERROR: Contract interaction failed: {e}")