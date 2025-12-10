from blockchain import BlockchainManager

print("Checking blockchain status...")

bc = BlockchainManager()

print(f"Connected: {bc.connected}")
print(f"Contract loaded: {bc.contract is not None}")
print(f"Account loaded: {bc.account is not None}")

if bc.account:
    print(f"Account address: {bc.account.address}")

    # Check balance
    try:
        balance = bc.w3.eth.get_balance(bc.account.address)
        eth_balance = bc.w3.from_wei(balance, 'ether')
        print(f"Account balance: {eth_balance} ETH")
    except Exception as e:
        print(f"Error checking balance: {e}")

# Test contract interaction
if bc.contract:
    try:
        # Test a simple call to see if contract exists
        test_hash = "0x" + "0" * 64  # Zero hash
        result = bc.get_verdict(test_hash)
        print(f"Contract call successful: {result}")
    except Exception as e:
        print(f"Contract call failed: {e}")
else:
    print("Contract not loaded")