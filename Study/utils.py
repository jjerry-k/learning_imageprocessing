import numpy as np
from matplotlib import pyplot as plt

def my_imshow(img, c_map):
    v_min = 0
    if img.dtype == 'uint8':
        v_max = 255
    elif img.dtype == 'float64':
        v_max = 1
    plt.imshow(img, cmap = c_map, vmin=v_min, vmax=v_max)
    
def im2double(img):
    _min = img.min()
    _max = img.max()
    
    return (img - _min)/(_max-_min)


def to_hist(img):
    L = 256
    hist = np.zeros((L,))
    for pix in range(L):
        hist[pix] = (img==pix).sum()
    return hist, hist/hist.sum()

def hist_eq(img):
    out = img.copy()
    L = 256
    _, hist_norm = to_hist(img)
    pdf = np.array([hist_norm[:x+1].sum() for x in range(L)])
    s = np.round(pdf * (L-1))
    for i, pix in enumerate(s):
        out[img==i] = pix
    hist, _= to_hist(out)
    return out, hist


def convolution(img, fil):
    img = im2double(img)
    fil_h, fil_w = fil.shape
    p = int((fil.shape[0]-1)/2)
    tmp = np.zeros(np.array(img.shape)+2*p)
    tmp[p:-p, p:-p] = img
    out = np.zeros_like(img)
    for i in range(tmp.shape[0]-fil_h+1):
        for j in range(tmp.shape[1]-fil_w+1):
            out[i, j]=np.sum(tmp[i:i+fil_h, j:j+fil_w]*fil)
    return out

def box_fil(size):
    return np.ones([size,size])/(size**2)

def gaussian_function(x, y, sigma):
    
    frac = 1/(2*np.pi*sigma**2)
    expo = -(x**2 + y**2)/(2*sigma**2)
    
    return frac*np.exp(expo)

def gaussian_filter(x, sigma):
    '''
    x is odd
    '''
    mid_point = np.uint8(x/2)
    out = np.zeros([x, x])
    for i in range(x):
        for j in range(x):
            out[i, j] = gaussian_function(j-mid_point, i-mid_point, sigma)
            
    return out

def median(img, mask_size):
    
    h, w = img.shape
    m_h, m_w = mask_size
    p_h = int((m_h - 1)/2)
    p_w = int((m_w - 1)/2)
    tmp_img = np.zeros([h+2*p_h, w+2*p_w])
    tmp_img[p_h:-p_h, p_w:-p_w] = img
    out = np.zeros_like(img)
    tmp_h, tmp_w = tmp_img.shape
    
    for i in range(tmp_w-m_w+1):
        for j in range(tmp_h-m_h+1):
            fil = tmp_img[j:j+m_h,i:i+m_w].flatten()
            fil.sort()
            mid = fil[int(len(fil)/2)+1]
            out[j, i] = mid

    return out

def laplacian(c=1, diagonal=True):
    out = np.ones([3,3])
    if diagonal == True:
        out[1,1] = -8
    else :
        edge = np.array([[0,0],[0,2],[2,0],[2,2]])
        for x, y in edge:
            out[x,y] = 0
        out[1,1] = -4
    return c*out

def usNhb(img, ksize, sigma, k):
    fil = gaussian_filter(ksize, sigma)
    tmp = convolution(img, fil)
    g_mask = img - tmp
    out = img + (k*g_mask)
   
    return out

def roberts():
    out = np.zeros([2,3,3])
    out[0,1,1]=-1
    out[0,2,2]=1
    out[1,1,2]=-1
    out[1,2,1]=1
    return out

def sobel():
    out = np.zeros([2,3,3])
    tmp = np.ones([3,3])
    tmp[:,1] = 2
    tmp[1,:] = 0
    tmp[0,:] *=-1
    out[0] = tmp
    out[1] = np.rot90(tmp)
    return out

def binarize(img, T):
    out = img.copy()
    out[out<T] = 0
    out[out>=T] = 1
    return out
    
def w0_t(hist_hat, t):
    return sum(hist_hat[:t+1]) + 1e-7

def m0_t(hist_hat, t):
    w0 = w0_t(hist_hat, t) 
    _sum = []
    for i in range(len(hist_hat[:t+1])):
        _sum.append(i*hist_hat[i])
    _sum = sum(_sum)
    return _sum/w0

def ostu(img):
    tmp_img = img.copy()
    L = img.max() + 1
    _, hist_hat = to_hist(img)
    
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

