from blockchain import BlockchainManager

# Test blockchain connection
bc = BlockchainManager()
print(f"Connected: {bc.connected}")
print(f"Contract loaded: {bc.contract is not None}")
print(f"Account loaded: {bc.account is not None}")

if bc.account:
    print(f"Account address: {bc.account.address}")

# Test hash function
test_hash = bc.hash_image(b"test image data")
print(f"Test hash: {test_hash}")

# Test get_verdict (should return None for non-existent hash)
verdict = bc.get_verdict(test_hash)
print(f"Verdict for test hash: {verdict}")