import sys, numpy as np
from PIL import Image
a=np.asarray(Image.open(sys.argv[1]).convert('RGB')).astype(np.float64)
b=np.asarray(Image.open(sys.argv[2]).convert('RGB')).astype(np.float64)
d=np.abs(a-b); mse=(d**2).mean()
psnr = 99.0 if mse==0 else 10*np.log10(255**2/mse)
print(f"maxdiff={d.max():.0f} meandiff={d.mean():.3f} psnr={psnr:.1f}dB pixels>8: {(d.max(axis=2)>8).mean()*100:.2f}%")
