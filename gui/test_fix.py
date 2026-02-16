
import numpy as np
import sys
import os

# Add gui folder to path
sys.path.append(os.getcwd())

from utility import psnrb
import traceback

try:
    img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Try running local psnrb
    val = psnrb(img1, img2)
    print(f"Local PSNR-B: {val}")
except NameError as e:
    print("Caught NameError:")
    traceback.print_exc()
except Exception as e:
    print(f"Caught {type(e).__name__}:")
    traceback.print_exc()
