
import numpy as np

def make_histogram(img):
    L = img.max()
    M = img.shape[0]
    N = img.shape[1]
    hist = []
    hist_hat = []
    for ldx in range(L+2):
        hist.append((img==ldx).sum())
        hist_hat.append((img==ldx).sum()/(M*N))
        
    return hist, hist_hat

def make_histogram_eq(img):
    tmp_img = np.zeros_like(img)
    L = img.max()+2
    hist, hist_hat = make_histogram(img)
    c = []
    t = []
    c_i = 0
    for i in hist_hat:
        c_i += i
        c.append(c_i)
        t.append(round(c_i*(L-1)))
    for pdx, pix in enumerate(t):
        tmp_img += (img==pdx)*np.uint8(pix)
    return hist, c, t, tmp_img