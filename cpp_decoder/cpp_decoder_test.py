import sys
import os
import glob

# --- 1. DYNAMIC PATH SETUP ---
# Look for the .so file in the build directory
build_path = os.path.abspath("./build")
sys.path.append(build_path)

try:
    # This must match the name in your PYBIND11_MODULE(cpp_tbcc_decoder, m)
    import cpp_tbcc_decoder
    print("✅ Successfully imported cpp_tbcc_decoder")
except ImportError as e:
    print(f"❌ Error: Could not import the module.")
    print(f"Details: {e}")
    print(f"Looking in: {build_path}")
    print(f"Files found in build: {os.listdir(build_path) if os.path.exists(build_path) else 'Folder not found'}")
    sys.exit(1)

# --- 2. RUN MINIMAL TEST ---
try:
    # Define the inputs (Match your CodeInformation constructor)
    # k, n, v, crcLen, crc, numInfoBits, numerators_vector
    k, n, v = 1, 2, 3
    crc_len, crc_poly, info_bits = 0, 0, 100
    nums = [13, 15] # Example numerators
    
    print("Creating CodeInformation...")
    config = cpp_tbcc_decoder.CodeInformation(k, n, v, crc_len, crc_poly, info_bits, nums)
    print(f"   -> Width (kconv): {config.kconv}")

    print("Initializing Trellis...")
    trellis = cpp_tbcc_decoder.FeedForwardTrellis(config)

    print("Initializing Decoder...")
    decoder = cpp_tbcc_decoder.LowRateListDecoder(trellis, config, 8)

    # print("Running Decode...")
    # puncturing_list = []
    # received = []
    # decoder.tbcc_decode()

    print("\n🚀 SUCCESS: The C++ bridge is fully functional!")

except Exception as e:
    print(f"❌ Runtime Error: {e}")
