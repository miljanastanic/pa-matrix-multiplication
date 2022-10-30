#Zadatak 1

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math
import sys

x = int(input('Broj redova matrice A: '))
y = int(input('Broj kolona matrice A: '))

p = int(input('Broj redova matrice B: '))
q = int(input('Broj kolona matrice B: '))

print('Unesite da li zelite da vam matrica bude transponovana T/N')
transAA = input('Matrica A: ')
transBB = input('Matrica B: ')

alfa = int(input('Alfa: '))

#m - broj redova matrice A *8*
#n - broj kolona matrice A *3*
#k - dimenzija matrice B *5* koja nedostaje (u zavisnosti od transa i transb može biti broj redova ili broj kolona)

mod = SourceModule("""
    __global__ void gemm(int transA, int transB, int m, int n, int k, int alfa, float *a, float *b, float *c) { 
        int x = threadIdx.x;
        int y = threadIdx.y; 

        float elementA, elementB, elementC;

        if((transA == 1) && (transB == 1)){
          if((y < k) && (x < n)) { 
              elementC = 0;
              for(int i = 0; i < m; i++) {
                  
                  int currA = y * m + i;
                  int currB = i * n + x;
                  
                  elementA = a[currA]; 
                  elementB = b[currB]; 
                  
                  elementC += (elementA * elementB) * alfa;
              }
              c[y * n + x] = elementC; 
          }
        }

         if((transA == 2) && (transB == 1)){
          if((y < k) && (x < n)) { 
              elementC = 0;
              for(int i = 0; i < m; i++) {

                  int currA = i * k + y;
                  int currB = i * n + x;

                  elementA = a[currA]; 
                  elementB = b[currB]; 
                  
                  elementC += (elementA * elementB) * alfa;
              }
              c[y * n + x] = elementC;
          }
        }

        if((transA == 1) && (transB == 2)){
          if((y < k) && (x < n)) { 
              elementC = 0;
              for(int i = 0; i < m; i++) {

                  int currA = y * m + i;
                  int currB = y * n + i;

                  elementA = a[currA];
                  elementB = b[currB]; 

                  elementC += (elementA * elementB) * alfa;
              }
              c[y * n + x] = elementC;
          }
        }

        if((transA == 2) && (transB == 2)){
          if((y < k) && (x < n)) { 
              elementC = 0;
              for(int i = 0; i < m; i++) {

                  int currA = i * k + y;
                  int currB = y * n + i;

                  elementA = a[currA];  
                  elementB = b[currB]; 

                  elementC += (elementA * elementB) * alfa;
              }
              c[y * n + x] = elementC; 
          }
        }

    }
""")

a = np.random.rand(x, y).astype(dtype=np.float32)
b = np.random.rand(p, q).astype(dtype=np.float32)

if(transAA == 'N' and transBB == 'N'):
  transA = np.int32(1)
  transB = np.int32(1)
  if(y == p):
    c = np.random.rand(x, q).astype(dtype=np.float32)
    b1 = x
    b2 = q
  else:
    print('Matrice se ne mogu pomnoziti')
    sys.exit()
if(transAA == 'T' and transBB == 'N'): 
  transA = np.int32(2)
  transB = np.int32(1)
  if(x == p):
    c = np.random.rand(y, q).astype(dtype=np.float32)
    b1 = y
    b2 = q
  else:
    print('Matrice se ne mogu pomnoziti')
    sys.exit()
if(transAA == 'N' and transBB == 'T'):
  transA = np.int32(1)
  transB = np.int32(2)
  if(y == q):
    c = np.random.rand(x, p).astype(dtype=np.float32)
    b1 = x
    b2 = p
  else:
    print('Matrice se ne mogu pomnoziti')
    sys.exit()
if(transAA == 'T' and transBB == 'T'):
  transA = np.int32(2)
  transB = np.int32(2)
  if(x == q):
    c = np.random.rand(y, p).astype(dtype=np.float32)  
    b1 = y
    b2 = p  
  else:
    print('Matrice se ne mogu pomnoziti')
    sys.exit()


a = np.round(a, 1)
b = np.round(b, 1)
c = np.zeros_like(c)

print("Matrica A:\n", a)
print("Matrica B:\n", b)
print("Matrica C:\n", c)

a_gpu = cuda.mem_alloc(a.nbytes)  
cuda.memcpy_htod(a_gpu, a) 
  
b_gpu = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(b_gpu, b) 

c_gpu = cuda.mem_alloc(c.nbytes)
cuda.memcpy_htod(c_gpu, c) 

func = mod.get_function("gemm") 

m = np.int32(a.shape[1]) 
n = np.int32(b.shape[1]) 
k = np.int32(a.shape[0]) 
#transA = np.int32(transAA)
#transB = np.int32(transBB) 
alfa = np.int32(alfa)
func(transA, transB, m, n, k, alfa, a_gpu, b_gpu, c_gpu, block=(b2, b1, 1), grid=(1, 1, 1))

cuda.memcpy_dtoh(c, c_gpu)  
c = np.round(c, 2)
print('Matrica C:\n', c)

result = []
if(transAA == 'N' and transBB == 'N'):
  result = np.matmul(a, b)
  result = np.round(result, 2)
  print(np.allclose(result, c))