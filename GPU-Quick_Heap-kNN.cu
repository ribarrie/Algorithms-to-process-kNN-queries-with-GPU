/*
* Copyright (C) 2014 Fabricio Bladimir Millaguir Zapata (fabriciomillaguir@gmail.com)
* 
* This program uses a GPU to process kNN queries, and it implements the algorithm 
* GPU-Quick_Heap-kNN presented in the article "GPU-based exhaustive algorithms processing kNN queries".
*
* This program is free software; you can redistribute it and/or modify it
* under the terms of the GNU General Public License (http://www.gnu.org/licenses/gpl.txt)
* as published by the Free Software Foundation; either version 2 of the
* License, or (at your option) any later version.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
**********/


/*
* The parameters of this program are set to be used with a database file of vectors,
* where each vector has 64 coordinates of type int, and one vector per line in the file. The coordinates
* of the vectors must be separated by the character ','. This program reads the coordinates as int, but
* store them as float. If you'd like to change this, go to leerVectorCophir function.
*/


//DIM is the dimension of the elements
#define DIM 64
#define ERROR -1
//N_THREADS is the number of threads per CUDA Block
#define N_THREADS 1024
//N_BLOCKS is the quantity of CUDA Blocks used in each iteration
#define N_BLOCKS 75
#define CANT_ARGS 6

#include <stdio.h>
#include <ctype.h>
#include <cuda.h>
#include <values.h>
#include <stdbool.h>
#include <sys/resource.h>
#include <time.h>
#include <sys/time.h>

//Structure used to save the index of an element and the its distance to the query.
struct _Elem {
      float dist;
      int ind;
};
typedef struct _Elem Elem;

/* Kernel implementing the GPU-Quick_Heap-kNN algorithm */
__global__ void gpuQuickHeapKNN(float *vectorGPU, float *consultaGPU, Elem *distancia, int posicionConsulta, int *cantVectoresGPU, size_t pitch_M, size_t pitch_Q, size_t pitch_D, int k);

/* Function to copy a vector to another */
void copiarValor(float *a, float *b);
/* Function to load a bector from the database file */
int leerVectorCophir(float *dato, FILE *file);
/* Function to read a vector from the database file and store it column-wise in a matrix*/
int leerVectorTransCophir(float **dato, FILE *file, int col);
/*Function that implements the quicksort algorithm */
void quick_sort(Elem *arr,int primero,int ultimo);
/* Function to validate if the device is available */
void validarDevice(int i);
/* Function to validate if the parameter is a file */
bool isFile(char * entrada);
/* Function to validate if the parameter is a number */
bool isNumber(char *number);
/* Function to validate if the input parameters are correct for the execution of this program */
void validarArgumentos(int argc, char *argv[]);
/* Function to validate if the quantity of threads per CUDA Block is correct */
void validarNThreads();
/* Function to print the results */
void imprimirResultados(Elem **resultado, int filas, int columnas);
/* Function to print execution time */
void imprimirTiempoEjecucion(struct rusage r1, struct rusage r2, struct timeval t1, struct timeval t2, int k);

/* GPU function to insert an element in a heap*/
__device__ void pushH(Elem *heap, Elem *elem, int *n_elem, int pitch, int id);
/* GPU function to extract an element in a heap */
__device__ void popH(Elem *heap, int *n_elem, int pitch, int id, Elem *eresult);
/* GPU function to retrieve the top element in a heap */
__device__ float topH(Elem *heap, int id);
/* GPU function to extract and insert en element in a heap in a single operation */
__device__ void popushH(Elem *heap, Elem *elem, int *n_elem, int pitch, int id);

int main(int argc, char *argv[])
{
    validarDevice(0);
    validarArgumentos(argc, argv);
    validarNThreads();   
         
    float **vector;
    float *vectorGPU;
    float **consulta;
    float *consultaGPU;
    Elem *distancia;
    Elem **resultado;
    int cantConsultas;
    int cantVectores[1];
    int *cantVectoresGPU;
    float dato[DIM];
    int i,j;
    int n_threads = N_THREADS;
    int n_blocks = N_BLOCKS;
    int k;
    char str_f[256];
    FILE *f_dist, *fquery;
    size_t pitch_M, pitch_Q, pitch_D;
    /* variables to measure time */
    struct rusage r1, r2;
    struct timeval t1, t2;
                                                         
    cantConsultas = atoi(argv[4]);
    cantVectores[0] = atoi(argv[2]);
    k = atoi(argv[5]);
   
    sprintf(str_f, "%s", argv[1]);
    fflush(stdout);
    f_dist = fopen(str_f, "r");
    fflush(stdout);
 
    /* Allocating memory to the results matrix */
    resultado = (Elem **)malloc(sizeof(Elem *)*cantConsultas);
    for (i = 0; i < cantConsultas; i++)
      resultado[i] = (Elem *)malloc(sizeof(Elem)*k);

    /* Allocating memory to the query matrix */
    consulta = (float **)malloc(sizeof(float *)*cantConsultas);
    for (i = 0; i < cantConsultas; i++)
      consulta[i] = (float *)malloc(sizeof(float)*DIM);

    /* Allocating memory to the vectors matrix */
    vector = (float **)malloc(sizeof(float *)*DIM);
    for (i = 0; i < DIM; i++)
      vector[i] = (float *)malloc(sizeof(float)*cantVectores[0]);

    fflush(stdout);
   
    /* Loading vectors column-wise to CPU memory */
    for(i = 0; i < cantVectores[0]; i++) {
      if (leerVectorTransCophir(vector, f_dist, i) == ERROR) {
        printf("\nError al leer\n");
        return 0;
      }
    }

    fclose(f_dist);
    fflush(stdout);

    if ((fquery=fopen(argv[3], "r")) == NULL)
      printf("Error al abrir para lectura el archivo de qeuries: %s\n", argv[3]);
    fflush(stdout);
    	
    /* Loading queries to CPU memory */ 
    for (i = 0; i < cantConsultas; i++) {
      if (leerVectorCophir(dato, fquery) == -1 || feof(fquery)) {
        printf("\n\nERROR :: cantConsultas mal establecido, Menos queries que las indicadas\n\n");
        fflush(stdout);
        fclose(fquery);
        break;
      }
      copiarValor(consulta[i], dato);
    }
    fclose(fquery);
    fflush(stdout);   	 
  
    /* Allocating GPU memory to the vectors */
    if (cudaSuccess != cudaMallocPitch((void **)&vectorGPU, &pitch_M, cantVectores[0]*sizeof(float), DIM)) {
      printf("\nERROR :: cudaMallocPitch \n");
      cudaThreadExit();
      return 0;
    }

    /* Allocating GPU memory to the queries */
    if (cudaSuccess != cudaMallocPitch((void **)&consultaGPU, &pitch_Q, DIM*sizeof(float), cantConsultas)) {
      printf("\nERROR :: cudaMallocPitch \n");
      cudaThreadExit();
      return 0;
    }

    /* Allocating GPU memory to the array of distances*/
    if (cudaSuccess != cudaMallocPitch((void **)&distancia, &pitch_D, cantVectores[0]*sizeof(Elem), N_BLOCKS*2)) {
      printf("\nERROR :: cudaMallocPitch \n");
      cudaThreadExit();
      return 0;
    }
          
    /* Allocating memory to cantVectoresGPU */
    if(cudaSuccess != cudaMalloc((void **)&cantVectoresGPU, sizeof(int)))  
      printf("\nERROR :: cudaMallocPitch \n");
  
    /* Copying vectors from CPU to GPU */
    for (i = 0; i < DIM; i++)
      if (cudaSuccess != cudaMemcpy((float *)((char *)vectorGPU + (i*(int)pitch_M)), (float *)(vector[i]), sizeof(float)*cantVectores[0], cudaMemcpyHostToDevice))
        printf("\nERROR :: cudaMemcpy\n");

    /* Copying queries from CPU to GPU */
    for (i = 0; i < cantConsultas; i++)
      if (cudaSuccess != cudaMemcpy((float *)((char *)consultaGPU + (i*(int)pitch_Q)), (float *)(consulta[i]), sizeof(float)*DIM, cudaMemcpyHostToDevice))
        printf("\nERROR :: cudaMemcpy\n");

    /*Copying quantity of vectors to process */ 
    if(cudaSuccess != cudaMemcpy(cantVectoresGPU, cantVectores, sizeof(int), cudaMemcpyHostToDevice))
      printf("\n ERROR :: cudaMemcpy\n");

    /* Starting time measure */
    getrusage(RUSAGE_SELF, &r1);
    gettimeofday(&t1, 0);

    /* Processing queries in batches. Each query is processed by one CUDA Block */                                                                                                                                                          
    for(i = 0; i < cantConsultas; i += N_BLOCKS) 
    {
      if((cantConsultas-i) < N_BLOCKS && (cantConsultas-i) > 0) 
        n_blocks = (cantConsultas-i);
      /* Launching kernel that implements the GPU-Quick_Heap-kNN */
      gpuQuickHeapKNN<<<n_blocks, n_threads>>>(vectorGPU, consultaGPU, distancia, i, cantVectoresGPU, pitch_M, pitch_Q, pitch_D, k);

      /* Copying results from GPU to CPU */
      for (j = 0; j < n_blocks; j++) {
        if(cudaSuccess != cudaMemcpy((Elem *)(resultado[j+i]), (Elem *)((char *)distancia + (((j*2)+1)*(int)pitch_D)), sizeof(Elem)*k, cudaMemcpyDeviceToHost))
          printf("\nERROR :: cudaMemcpy device to host\n");
      }
    } 

    cudaDeviceSynchronize();

    /* Stopping measuring time */
    gettimeofday(&t2, 0);
    getrusage(RUSAGE_SELF, &r2);
  
    imprimirTiempoEjecucion(r1, r2, t1, t2, k);                                                                                                                                    
    imprimirResultados(resultado, cantConsultas, k);

    cudaFree(vectorGPU);
    cudaFree(consultaGPU);
    cudaFree(cantVectoresGPU);
    free(vector);
    free(consulta);
    cudaThreadExit();
                                                                                                                                                   
    return 0;
}

__global__ void gpuQuickHeapKNN(float *vectorGPU, float *consultaGPU, Elem *distancia, int posicionConsulta, int *cantVectoresGPU, size_t pitch_M, size_t pitch_Q,size_t pitch_D, int k)
{
    __shared__ float query[DIM];
    __shared__ float pivote;
    __shared__ int cantElementos[2];
    __shared__ int posicion;
    __shared__ Elem heap[N_THREADS][1];
    int tid = threadIdx.x; 
    int i, j;
    int posCantElem1=0;
    int posCantElem2=1;
    int cont=0;
    int cambio[2] = {1,0};
    float distlocal;
    float suma=0;
    float pivoteLocal;
    float f=0.0;
    Elem elem;

    /* Copying query to be solved by the current CUDA Block to shared memory */
    if(tid < DIM )
      query[tid] = ((float *)((char *)consultaGPU + ((blockIdx.x+posicionConsulta)*(int)pitch_Q)))[tid];

    if(tid == 0) {
      cantElementos[0] = cantVectoresGPU[0];
      cantElementos[1] = 0;
      pivote = 0;
      posicion = 0;
    }

    __syncthreads();
   
    /* Calculating distances between vectors and the query (which is being solved by the current CUDA Block) */
    for(i = tid; i < cantVectoresGPU[0]; i += blockDim.x) 
    {
      distlocal = 0.0;
      for (j = 0; j < DIM; j++) 
      {
        f = ( ((float *)((char *)vectorGPU + (j*(int)pitch_M)))[i] - query[j]);    
        distlocal += f*f;
      }
      elem.dist = sqrtf(distlocal);
      elem.ind = 2;			 
      ((Elem *)((char *)distancia + ((blockIdx.x*2)*(int)pitch_D)))[i] = elem;
	    suma += elem.dist;
    }

    /*Summing all the distances*/
    atomicAdd(&pivote, suma);
    __syncthreads();
    
    /* Searching elements that are the candidates to be the k nearest elements. Always the quantity of candidates is higher or equal to k */
    while(cantElementos[posCantElem1] > k) 
    {
      suma = 0;
      cont = 0;
      pivoteLocal = (pivote/cantElementos[posCantElem1]);

      __syncthreads();

      /* Searching elements lower than the pivot */
      for(i = tid; i < cantVectoresGPU[0]; i += blockDim.x) 
      {
        elem = ((Elem *)((char *)distancia + (((blockIdx.x*2))*(int)pitch_D)))[i];		
        if(elem.dist <= pivoteLocal) 
        {
          suma += elem.dist;
          cont++;	
        }
        else 
          if(elem.ind > 0) 
          {
            ((Elem *)((char *)distancia + (((blockIdx.x*2))*(int)pitch_D)))[i].ind--;
          }		  
      }

      atomicAdd(&cantElementos[posCantElem2], cont);
      __syncthreads();

      if(cantElementos[posCantElem2] < k)      
        break;
      /* checking cases where all the distances are equals */
      if(cantElementos[posCantElem2] == cantElementos[posCantElem1])
        break;
      if(tid == 0) 
      {
        cantElementos[posCantElem1] = 0;
        pivote = 0;
      }

      __syncthreads();
        
      posCantElem1 = cambio[posCantElem1];
      posCantElem2 = cambio[posCantElem2];

      atomicAdd(&pivote, suma);
      __syncthreads();
      
    }
 
    __syncthreads(); 
    
    /* Retrieving all the elements which are the candidates to be the results */
    for(i = tid; i < cantVectoresGPU[0]; i += blockDim.x) 
    {
      elem = ((Elem *)((char *)distancia + (((blockIdx.x*2))*(int)pitch_D)))[i];
      if(elem.ind > 0) 
      {
        elem.ind = i;
        ((Elem *)((char *)distancia + (((blockIdx.x*2)+1)*(int)pitch_D)))[atomicAdd(&posicion, 1)] = elem;
      }
    }
 
    __syncthreads();
  
    /* Storing the final results to the heap */
    if(tid == 0) 
    {
      int n_elem = 0;
      for(i = 0; i < posicion; i++) 
      {
        elem = ((Elem *)((char *)distancia + (((blockIdx.x*2)+1)*(int)pitch_D)))[i];
        if (n_elem < k)
          pushH((Elem *)heap, &(elem), &n_elem, sizeof(Elem), 0);
          else
            if (topH((Elem *)heap, 0) > elem.dist)
              popushH((Elem *)heap, &(elem), &n_elem, sizeof(Elem), 0);
      }
    }

    __syncthreads();

    /* Storing results*/
    for (i = tid; i < k; i += blockDim.x) 
    {
      ((Elem *)((char *)distancia + (((blockIdx.x*2)+1)*(int)pitch_D)))[i] = heap[i][0];
    }

    return;
}

void copiarValor(float *a, float *b)
{  
    int i;
    for(i = 0; i < DIM; i++)
      a[i] = b[i];
   
    return;
}

int leerVectorCophir(float *dato, FILE *file)
{
    int i = 0;
    int num_f;

    /* Reading DIM coordinates of type int */
    for (i = 0; i < DIM; i++) {
      if (fscanf(file, "%d", &num_f) < 1)
        return ERROR;

      /* Storing coordinates as float */
      dato[i] = (float)num_f;

      if (i+1 < DIM)
        if (fgetc(file) != ',') {
          printf("\nERROR :: ',' no encontrada\n");
          return ERROR;
        }
    }

    return 1;
}

int leerVectorTransCophir(float **dato, FILE *file, int col)
{
    int i = 0;
    int num_f;

    for (i = 0; i < DIM; i++) {
      if (fscanf(file, "%d", &num_f) < 1)
        return ERROR;
      dato[i][col] = (float)num_f;

      if (i+1 < DIM)
        if (fgetc(file) != ',') {
          printf("\nERROR :: ',' no encontrada\n");
          return ERROR;
        }
    }
    return 1;
}

void quick_sort(Elem *arr,int primero,int ultimo)
{
    int i, j;
    float central;
    Elem aux;
    i = primero;
    j = ultimo;
    central = arr[(int)((primero+ultimo)/2)].dist;
   
    do {
      while(arr[i].dist < central && i < ultimo)
        i++;
      while(central < arr[j].dist && primero < j)
        j--;
      if (i <= j) {
        aux = arr[i];
        arr[i] = arr[j];
        arr[j] = aux;
        i++;
        j--;
      }
    }while(i <= j);
   
    if (primero < j)
      quick_sort(arr, primero, j);
    if (i < ultimo)
      quick_sort(arr, i, ultimo);
    return;
}

__device__ void pushH(Elem *heap, Elem *elem, int *n_elem, int pitch, int id)
{
    int i;
    Elem temp;

    ((Elem *)((char *)heap + (*n_elem)*pitch))[id].dist = elem->dist;
    ((Elem *)((char *)heap + (*n_elem)*pitch))[id].ind = elem->ind;
    (*n_elem)++;
    for (i = *n_elem; i>1 && ((Elem *)((char *)heap + (i-1)*pitch))[id].dist > ((Elem *)((char *)heap + ((i/2)-1)*pitch))[id].dist; i=i/2)
    {
      temp.dist = ((Elem *)((char *)heap + (i-1)*pitch))[id].dist;
      temp.ind = ((Elem *)((char *)heap + (i-1)*pitch))[id].ind;
      ((Elem *)((char *)heap + (i-1)*pitch))[id].dist = ((Elem *)((char *)heap + ((i/2)-1)*pitch))[id].dist;
      ((Elem *)((char *)heap + (i-1)*pitch))[id].ind = ((Elem *)((char *)heap + ((i/2)-1)*pitch))[id].ind;
      ((Elem *)((char *)heap + ((i/2)-1)*pitch))[id].dist = temp.dist;
      ((Elem *)((char *)heap + ((i/2)-1)*pitch))[id].ind = temp.ind;
    }
    return;
}

__device__ void popH(Elem *heap, int *n_elem, int pitch, int id, Elem *eresult)
{
    int i, k;
    Elem temp;
    eresult->dist = ((Elem *)((char *)heap+0))[id].dist;
    eresult->ind = ((Elem *)((char *)heap+0))[id].ind;

    ((Elem *)((char *)heap+0))[id].dist = ((Elem *)((char *)heap + ((*n_elem)-1)*pitch))[id].dist;
    ((Elem *)((char *)heap+0))[id].ind = ((Elem *)((char *)heap + ((*n_elem)-1)*pitch))[id].ind;
    (*n_elem)--;
    i = 1;
    while(2*i <= *n_elem) {
      k = 2*i; 
      if(k+1 <= *n_elem && ((Elem *)((char *)heap + ((k+1)-1)*pitch))[id].dist > ((Elem *)((char *)heap + (k-1)*pitch))[id].dist)
        k = k+1;

      if(((Elem *)((char *)heap + (i-1)*pitch))[id].dist > ((Elem *)((char *)heap + (k-1)*pitch))[id].dist)
        break;

      temp.dist = ((Elem *)((char *)heap + (i-1)*pitch))[id].dist;
      temp.ind = ((Elem *)((char *)heap + (i-1)*pitch))[id].ind;
      ((Elem *)((char *)heap + (i-1)*pitch))[id].dist = ((Elem *)((char *)heap + (k-1)*pitch))[id].dist;
      ((Elem *)((char *)heap + (i-1)*pitch))[id].ind = ((Elem *)((char *)heap + (k-1)*pitch))[id].ind;
      ((Elem *)((char *)heap + (k-1)*pitch))[id].dist = temp.dist;
      ((Elem *)((char *)heap + (k-1)*pitch))[id].ind = temp.ind;
      i = k;
    }
    return;
}

__device__ float topH(Elem *heap, int id)
{
    return ((Elem *)((char *)heap + 0))[id].dist;
}

__device__ void popushH(Elem *heap, Elem *elem, int *n_elem, int pitch, int id)
{
    int i, k;
    Elem temp;

    ((Elem *)((char *)heap+0))[id].dist = elem->dist;
    ((Elem *)((char *)heap+0))[id].ind  = elem->ind;

    i = 1;
    while(2*i <= *n_elem) {
      k = 2*i;
      if(k+1 <= *n_elem && ((Elem *)((char *)heap + ((k+1)-1)*pitch))[id].dist > ((Elem *)((char *)heap + (k-1)*pitch))[id].dist)
        k = k+1;

      if(((Elem *)((char *)heap + (i-1)*pitch))[id].dist > ((Elem *)((char *)heap + (k-1)*pitch))[id].dist)
        break;

      temp.dist = ((Elem *)((char *)heap + (i-1)*pitch))[id].dist;
      temp.ind = ((Elem *)((char *)heap + (i-1)*pitch))[id].ind;
      ((Elem *)((char *)heap + (i-1)*pitch))[id].dist = ((Elem *)((char *)heap + (k-1)*pitch))[id].dist;
      ((Elem *)((char *)heap + (i-1)*pitch))[id].ind = ((Elem *)((char *)heap + (k-1)*pitch))[id].ind;
      ((Elem *)((char *)heap + (k-1)*pitch))[id].dist = temp.dist;
      ((Elem *)((char *)heap + (k-1)*pitch))[id].ind = temp.ind;
      i = k;
    }
    return;
}

bool isFile(char * entrada)
{ 
    FILE *file;
    bool bFile = false;
    if(file = fopen(entrada, "r")) {
      fclose(file);
      bFile = true;
    }

    return bFile;
}

bool isNumber(char *number)
{
    bool formatoNumero = (*number)?true:false;
  
    for(;*number && formatoNumero;number++)
      if(!isdigit(*number))
        formatoNumero = false;
    return formatoNumero;
}

void validarDevice(int i)
{
    if (cudaSuccess != cudaSetDevice(i)) {
      printf("\n\nERROR :: cudaSetDevice\n\n");
      fflush(stdout); 
      exit(EXIT_SUCCESS);  
    }   
}

void validarArgumentos(int argc, char *argv[])
{
    bool ok = false;  
    if (argc == CANT_ARGS) {
      ok = true;    
      /*archivo_BD*/
      if(!isFile(argv[1])) {
        printf("Error en archivo de base de datos (archivo_BD = %s). Debe ser un archivo valido\n",argv[1]);
        ok = false;
      }
      /*Num_elem*/
      if(!isNumber(argv[2])) {
        printf("Error en la cantidad de elementos de base de datos (Num_elem = %s). Debe ser un valor numerico\n",argv[2]);
        ok = false;
      }
      /*archivo_queries*/
      if(!isFile(argv[3])) {
        printf("Error en archivo de consultas (archivo_queries = %s). Debe ser un archivo valido\n",argv[3]);
        ok = false;
      }
      /*Num_queries*/
      if(!isNumber(argv[4])) {
        printf("Error en la cantidad de consultas (Num_queries = %s). Debe ser un valor numerico\n",argv[4]);
        ok = false;
      }   
      /*K*/
      if(!isNumber(argv[5])) {
        printf("Error en K (K = %s). Debe ser un valor numerico\n",argv[5]);
        ok = false;
      }
    } 
    if(!ok) {
      printf("\nError :: Ejecutar como : %s archivo_BD Num_elem archivo_queries Num_queries K\n",argv[0]);
      exit(EXIT_SUCCESS);
    }
}

void validarNThreads()
{
    if(N_THREADS < DIM) {
      printf("ERROR :: N_THREADS debe se mayor o igual a %d\n",DIM);
      exit(EXIT_SUCCESS);
    } 
}

void imprimirResultados(Elem **resultado, int filas, int columnas)
{
    int i, j;
    for(i = 0; i < filas; i++)
      for(j = 0; j < columnas; j++)           
        printf("%d %f\n", resultado[i][j].ind, resultado[i][j].dist);
}

void imprimirTiempoEjecucion(struct rusage r1, struct rusage r2, struct timeval t1, struct timeval t2, int k) {
    float user_time = (r2.ru_utime.tv_sec - r1.ru_utime.tv_sec) + (r2.ru_utime.tv_usec - r1.ru_utime.tv_usec)/1000000.0;
    float sys_time = (r2.ru_stime.tv_sec - r1.ru_stime.tv_sec) + (r2.ru_stime.tv_usec - r1.ru_stime.tv_usec)/1000000.0;
    float real_time = (t2.tv_sec - t1.tv_sec) + (float)(t2.tv_usec - t1.tv_usec)/1000000;

    printf("\nK = %d", k);
    printf("\nCPU Time = %f", sys_time + user_time);
    printf("\nReal Time = %f\n", real_time);
    fflush(stdout);
}
