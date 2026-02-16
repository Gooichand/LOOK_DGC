
import numpy as np
import sewar
import traceback

try:
    img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # sewar expects (H, W, C) or (H, W)
    # The fix in user request mentions copying psnrb to utility.py
    
    # Try running psnrb
    val = sewar.psnrb(img1, img2)
    print(f"PSNR-B: {val}")
except NameError as e:
    print("Caught NameError:")
    traceback.print_exc()
except Exception as e:
    print(f"Caught {type(e).__name__}:")
    traceback.print_exc()
