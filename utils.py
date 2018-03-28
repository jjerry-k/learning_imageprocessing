
import numpy as np

def make_histogram(img):
    L = img.max() + 1
    M = img.shape[0]
    N = img.shape[1]
    hist = []
    hist_hat = []
    for ldx in range(L):
        hist.append((img==ldx).sum())
        hist_hat.append((img==ldx).sum()/(M*N))
        
    return hist, hist_hat

def make_histogram_eq(img):
    tmp_img = np.zeros_like(img)
    L = img.max()+1
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

def w0_t(hist_hat, t):
    return sum(hist_hat[:t+1]) + 1e-7

def w1_t(hist_hat, t):
    return sum(hist_hat[t+1:]) + 1e-7

def m0_t(hist_hat, t):
    w0 = w0_t(hist_hat, t) 
    _sum = []
    for i in range(len(hist_hat[:t+1])):
        _sum.append(i*hist_hat[i])
    _sum = sum(_sum)
    return _sum/w0

def m1_t(hist_hat, t):
    w1 = w1_t(hist_hat, t)
    _sum = []
    for i in range(len(hist_hat[t+1:])):
        i += t+1
        _sum.append(i*hist_hat[i])
    _sum = sum(_sum)
    return _sum/w1

def v0_t(hist_hat, t):
    w0 = w0_t(hist_hat, t)
    m0 = m0_t(hist_hat, t)
    _sum = []
    for i in range(len(hist_hat[:t+1])):
        _sum.append(hist_hat[i]*(i-m0)**2)
    _sum = sum(_sum)
    
    return(_sum/w0)

def v1_t(hist_hat, t):
    w1 = w1_t(hist_hat, t)
    m1 = m1_t(hist_hat, t)
    _sum = []
    for i in range(len(hist_hat[t+1:])):
        i += t+1
        _sum.append(hist_hat[i]*(i-m1)**2)
    _sum = sum(_sum)
    
    return(_sum/w1)

def ostu(img):
    tmp_img = img.copy()
    L = img.max() + 1
    _, hist_hat = make_histogram(img)
    
    m = []
    _sum=[]
    for i in range(len(hist_hat)):
        _sum.append(i*hist_hat[i])
    m = sum(_sum)
    
    w0 = w0_t(hist_hat, 0)
    m0 = m0_t(hist_hat, 0)
    m1t = 0
    v_b = []
    for i in range(1, L):
        w0t = w0 + hist_hat[i]
        m0t = (w0*m0 + i*hist_hat[i])/w0t
        m1t = (m - w0t*m0t)/(1-w0t)
        v = w0t*(1-w0t)*(m0t-m1t)**2
        v_b.append(v)
        w0 = w0t
        m0 = m0t
        
    th = v_b.index(max(v_b))
    
    tmp_img[tmp_img<th] = 0
    tmp_img[tmp_img>=th] = 1
    
    return tmp_img, th

def correlation(img, mask):
    '''
    mask is odd number.
    '''
    h, w = img.shape
    m_h, m_w = mask.shape
    p = int((m_h - 1)/2)
    tmp_img = np.zeros([h+2*p, w+2*p])
    out = np.zeros_like(img)                    
    tmp_img[p:-p, p:-p] = img
    tmp_h, tmp_w = tmp_img.shape
    for i in range(tmp_w-m_w+1):
        for j in range(tmp_h-m_h+1):
            out[j,i]=(tmp_img[j:j+m_h,i:i+m_w] * mask).sum()
            
    return out

def convolution(img, mask):
    '''
    mask is odd number.
    '''
    h, w = img.shape
    m_h, m_w = mask.shape
    p = int((m_h - 1)/2)
    tmp_img = np.zeros([h+2*p, w+2*p])
    mask = np.rot90(mask, 2)
    out = np.zeros_like(img)                    
    tmp_img[p:-p, p:-p] = img
    tmp_h, tmp_w = tmp_img.shape
    for i in range(tmp_w-m_w+1):
        for j in range(tmp_h-m_h+1):
            out[j,i]=(tmp_img[j:j+m_h,i:i+m_w] * mask).sum()
            
    return out


def median(img, mask_size):
    '''
    mask_size is odd.
    '''
    h, w = img.shape
    m_h, m_w = mask_size
    p_h = int((m_h - 1)/2)
    p_w = int((m_w - 1)/2)
    tmp_img = np.zeros([h+2*p_h, w+2*p_w])
    out = np.zeros_like(img)                    
    tmp_img[p_h:-p_h, p_w:-p_w] = img
    tmp_h, tmp_w = tmp_img.shape
    
    for i in range(tmp_w-m_w+1):
        for j in range(tmp_h-m_h+1):
            fil = tmp_img[j:j+m_h,i:i+m_w].flatten()
            fil.sort()
            mid = fil[int(len(fil)/2)+1]
            out[j, i] = mid
            
    return out          

def bright(img, a):
    _max = img.max()
    
    img_out = np.uint16(img) + a
    
    img_out[img_out>=_max] = _max
    
    img_out[img_out<=0] = 0
    
    return np.uint8(img_out)

def color_rev(img):
    return img.max()-img

def gamma_corr(img, gamma):
    return img.max() * (img/img.max())**gamma
    