#Zadatak 3

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

mod = SourceModule("""
    __global__ void gemm(int transA, int transB, int m, int n, int k, int alfa, float *a, float *b, float *c) { 
        
        int const bDim = 32;

        __shared__ float shared_MA[bDim][bDim];

        __shared__ float shared_MB[bDim][bDim];

        int x = bDim * blockIdx.x + threadIdx.x;
         
        int y = bDim * blockIdx.y + threadIdx.y; 
        
        int limit = (m-1)/bDim + 1;

        float elementA, elementB, elementC;
        elementC = 0;


        if((transA == 1) && (transB == 1)){
          for(int i = 0; i < limit; ++i) {

              if ((m > threadIdx.x + i * bDim) && (y < k)){
                  shared_MA[threadIdx.y][threadIdx.x] = a[threadIdx.x + i * bDim + y * m ];
              }else{
                  shared_MA[threadIdx.y][threadIdx.x] = 0;
              }
              
              if ((m > threadIdx.y + i * bDim) && (x < n)){
                  shared_MB[threadIdx.y][threadIdx.x] = b[(threadIdx.y + i * bDim) * n + x];
              }else{
                  shared_MB[threadIdx.y][threadIdx.x] = 0;
              }

              __syncthreads();

              for (int j = 0; j < bDim; ++j){
                  elementA = shared_MA[threadIdx.y][j];
                  elementB = shared_MB[j][threadIdx.x];
                  elementC += alfa * elementA * elementB;
              }

              __syncthreads();
          }
          if ((y < k) && (x < n)){
            c[y * n + x] = elementC;
          }
        }


        if((transA == 2) && (transB == 1)){
          for(int i = 0; i < limit; ++i) {

              if ((k > threadIdx.x + i * bDim) && (x < m)){
                  shared_MA[threadIdx.y][threadIdx.x] = a[(threadIdx.x + i * bDim) * m + x];
              }else{
                  shared_MA[threadIdx.y][threadIdx.x] = 0;
              }
              
              if ((m > threadIdx.y + i * bDim) && (x < n)){
                  shared_MB[threadIdx.y][threadIdx.x] = b[(threadIdx.y + i * bDim) * n + x];
              }else{
                  shared_MB[threadIdx.y][threadIdx.x] = 0;
              }

              __syncthreads();

              for (int j = 0; j < bDim; ++j){
                  elementA = shared_MA[j][threadIdx.y];
                  elementB = shared_MB[j][threadIdx.x];
                  elementC += alfa * elementA * elementB;
              }

              __syncthreads();
          }
          if ((y < k) && (x < n)){
            c[y * n + x] = elementC;
          }
        }


        if((transA == 1) && (transB == 2)){
          for(int i = 0; i < limit; ++i) {

              if ((m > threadIdx.x + i * bDim) && (y < k)){
                  shared_MA[threadIdx.y][threadIdx.x] = a[threadIdx.x + i * bDim + y * m];
              }else{
                  shared_MA[threadIdx.y][threadIdx.x] = 0;
              }
              
              if ((n > threadIdx.y + i * bDim) && (y < m)){
                  shared_MB[threadIdx.y][threadIdx.x] = b[threadIdx.y + i * bDim + y * n];
              }else{
                  shared_MB[threadIdx.y][threadIdx.x] = 0;
              }

              __syncthreads();

              for (int j = 0; j < bDim; ++j){
                  elementA = shared_MA[threadIdx.y][j];
                  elementB = shared_MB[threadIdx.y][j];
                  elementC += alfa * elementA * elementB;
              }

              __syncthreads();
          }
          if ((y < k) && (x < n)){
            c[y * n + x] = elementC;
          }
        }


        if((transA == 2) && (transB == 2)){
          for(int i = 0; i < limit; ++i) {

              if ((k > threadIdx.x + i * bDim) && (x < m)){
                  shared_MA[threadIdx.y][threadIdx.x] = a[(threadIdx.x + i * bDim) * m + x];
              }else{
                  shared_MA[threadIdx.y][threadIdx.x] = 0;
              }
              
              if ((n > threadIdx.y + i * bDim) && (y < m)){
                  shared_MB[threadIdx.y][threadIdx.x] = b[threadIdx.y + i * bDim * n + y * n];
              }else{
                  shared_MB[threadIdx.y][threadIdx.x] = 0;
              }

              __syncthreads();

              for (int j = 0; j < bDim; ++j){
                  elementA = shared_MA[j][threadIdx.y];
                  elementB = shared_MB[threadIdx.y][j];
                  elementC += alfa * elementA * elementB;
              }

              __syncthreads();
          }
          if ((y < k) && (x < n)){
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


m = np.int32(a.shape[1]) #widithA, heightB
n = np.int32(b.shape[1]) #widithB, widithC
k = np.int32(a.shape[0]) #heightA, heightC
alfa = np.int32(alfa)
#transA = np.int32(1)
#transB = np.int32(1)
func(transA, transB, m, n, k, alfa, a_gpu, b_gpu, c_gpu,  block=(32, 32, 1), grid=(math.ceil(c.shape[1] / 32), math.ceil(c.shape[0] / 32), 1))

cuda.memcpy_dtoh(c, c_gpu)  
c = np.round(c, 2)
print('Matrica C:\n', c)

result = []
if(transAA == 'N' and transBB == 'N'):
  result = np.matmul(a, b)
  result = np.round(result, 2)
  print(np.allclose(result, c))