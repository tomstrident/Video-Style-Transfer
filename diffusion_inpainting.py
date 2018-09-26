

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.ndimage

from scipy.ndimage.filters import gaussian_filter
from coherence import make_derivatives_2D, make_derivatives_hp_sym_2D

def image_diffusion(image, mask=None, 
                    alpha=0.005, gamma=0.001, # 0.0005, 0.0001
                    tau=5, end_time=15,        # 100
                    sigma_g=0.7, sigma_t=1.5,   # 1.0, 0.5
                    gauss_mode='reflect',
                    derivatives='symmetric'):
  
  assert len(image.shape) == 3, "dimension error: input has to be single image"

  np_image = np.array(image)
  
  if np.max(np_image) > 1.0:
    np_image = np_image/255.0
  
  M, N, channels = np_image.shape
  img_size = (M, N)
  MxN = M*N
  
  if derivatives is 'forward':
    Kx, Ky = make_derivatives_2D(img_size)
  elif derivatives is 'symmetric':
    Kx, Ky = make_derivatives_hp_sym_2D(img_size)
  
  nabla = scipy.sparse.vstack([Kx, Ky]).tocsc()
  identity = scipy.sparse.eye(MxN, format='csc')
  
  np_noised_image = np_image + np.random.randn(*np_image.shape)*1e-3
  np_image = np.clip(np_noised_image, 0.0, 1.0)
  
  # mask
  if mask is not None:
    assert len(mask.shape) == 2, "dimension error: mask has to be binary"
    assert image.shape[:2] == mask.shape, "dimension error: mask has to be \
                                           same shape as image"
    np_mask_nor = np.array(mask)
    
    if np.max(np_mask_nor) > 1.0:
      np_mask_nor = np_mask_nor/255.0
    
    sp_mask_nor = identity.multiply(np_mask_nor.ravel())
    sp_mask_inv  = identity.multiply((1.0 -  np_mask_nor).ravel())
  
  for t in np.arange(0, end_time, tau):
    
    D = scipy.sparse.csr_matrix((2*MxN, 2*MxN))
    
    for c in range(channels):
  
      u = np_image[:,:,c]
      u_f = gaussian_filter(u, sigma_g, mode=gauss_mode)

      u = u.ravel()
      u_f = u_f.ravel()

      u_x = np.reshape(Kx*u_f, img_size)
      u_y = np.reshape(Ky*u_f, img_size)

      u_x2 = gaussian_filter(u_x**2, sigma_t, mode=gauss_mode).flatten()
      u_xy = gaussian_filter(u_x*u_y, sigma_t, mode=gauss_mode).flatten()
      u_y2 = gaussian_filter(u_y**2, sigma_t, mode=gauss_mode).flatten()

      l1, l2, eigvecs_1, eigvecs_2 = diffusion_eigencomponents(u_x2, u_xy, 
                                                               u_y2, 
                                                               alpha, gamma)
    
      D += diffusion_tensor(l1, l2, eigvecs_1, eigvecs_2, MxN)
    
    D /= channels

    for c in range(channels):
      
      u = np_image[:,:,c].ravel()
      
      if mask is None:
        L = identity + tau*nabla.T*D*nabla
        u = scipy.sparse.linalg.spsolve(L, u)
      else:
        L = sp_mask_nor - tau*sp_mask_inv*nabla.T*D*nabla
        u = scipy.sparse.linalg.spsolve(L, sp_mask_nor*u)

      u = np.reshape(u, img_size)
      
      np_image[:,:,c] = u
    
    print('time', t)
  
  return np.clip(np_image*255.0, 0, 255).astype(np.uint8)

def diffusion_tensor(l1, l2, eigvecs_1, eigvecs_2, base_shape):
  
  evec_1x = eigvecs_1[0]
  evec_1y = eigvecs_1[1]
  evec_2x = eigvecs_2[0]
  evec_2y = eigvecs_2[1]
  
  D11 = l1*(evec_1x**2) + l2*(evec_2x**2)
  D12 = l1*evec_1x*evec_1y + l2*evec_2x*evec_2y
  D21 = D12
  D22 = l1*(evec_1y**2) + l2*(evec_2y**2)

  a = np.hstack((D11, D22))
  b = np.hstack((np.zeros(base_shape), D12))
  c = np.hstack((D21, np.zeros(base_shape)))

  D = scipy.sparse.spdiags(np.array([a, b, c]), 
                           np.array([0, base_shape, -base_shape]), 
                           2*base_shape, 2*base_shape)
  
  return D

def g(s, gamma):
  return np.exp(-(s**2)/(2.0*(gamma**2)))

def diffusion_eigencomponents(u_x2, u_xy, u_y2, alpha, gamma):
  
   # eigenvalues:
  trace = u_x2 + u_y2
  det = u_x2*u_y2 - u_xy**2

  eigvals_1 = (trace + np.sqrt(trace**2 - 4.0*det))/2.0
  eigvals_2 = (trace - np.sqrt(trace**2 - 4.0*det))/2.0

  l1 = alpha
  l2 = alpha + (1.0 - alpha)*(1.0 - g(np.abs(eigvals_1 - eigvals_2), gamma))

  # eigenvectors
  eigvecs_1 = np.vstack(((eigvals_1 - u_y2), u_xy))
  eigvecs_2 = np.vstack(((eigvals_2 - u_y2), u_xy))

  idx = np.where(u_xy == 0)
  eigvecs_1[0, idx] = 1
  eigvecs_1[1, idx] = 0
  eigvecs_2[0, idx] = 0
  eigvecs_2[1, idx] = 1

  # normalize eigenvectors
  eigvecs_1 = eigvecs_1/np.linalg.norm(eigvecs_1, axis=0)
  eigvecs_2 = eigvecs_2/np.linalg.norm(eigvecs_2, axis=0)
  
  return l1, l2, eigvecs_1, eigvecs_2

