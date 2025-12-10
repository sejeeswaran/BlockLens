from solcx import install_solc, get_installed_solc_versions
import traceback

print("Checking solc versions...")
print(f"Installed: {get_installed_solc_versions()}")

try:
    print("Attempting to install solc 0.8.0...")
    install_solc('0.8.0')
    print("Success!")
except Exception as e:
    print("Installation failed.")
    traceback.print_exc()
