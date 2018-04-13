import numpy as np


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
    fil_h, fil_w = fil.shape
    p = int((fil.shape[0]-1)/2)
    tmp_img = np.zeros(np.array(img.shape)+2*p)
    tmp_img[p:-p, p:-p] = np.double(img)
    out = np.zeros_like(img, float)
    for i in range(tmp_img.shape[0]-fil.shape[0]+1):
        for j in range(tmp_img.shape[1]-fil.shape[1]+1):
            out[i, j]=np.sum(tmp_img[i:i+fil_h, j:j+fil_w]*fil)
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
    out = np.uint16(img) + (k*g_mask)
    out[out>255] = 255
    return np.uint8(out)

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
    out[1] = np.rot(tmp)
    return out