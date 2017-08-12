/*
* Copyright (C) 2017 Fabricio Bladimir Millaguir Zapata (fabriciomillaguir@gmail.com)
* 
* This program uses a GPU to process kNN queries, and it implements the algorithm 
* GPU-Sel-kNN v2 presented in the article "GPU-based exhaustive algorithms processing kNN queries".
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
//N_WARP is the number of threads in a warp
#define N_WARP 32
//N_BLOCKS is the quantity of CUDA Blocks used in each iteration
#define N_BLOCKS 150
#define CANT_ARGS 6

#include <stdio.h>
#include <cuda.h>
#include <values.h>
#include <stdbool.h>
#include <ctype.h>
#include <sys/resource.h>
#include <time.h>
#include <sys/time.h>

//Structure used to save the index of an element and the its distance to the query.
struct _Elem {
      float dist;
      int ind;
};
typedef struct _Elem Elem;

/*kernel implementing the GPU-Sel-kNN v2 algorithm */
__global__ void gpuSelKNN(float *vectorGPU, float *consultaGPU, Elem *resultadoGPU, 
                          float *distancia, int posicionConsulta, int *cantVectoresGPU, 
                          size_t pitch_M, size_t pitch_Q, size_t pitch_R, 
                          size_t pitch_D, int k);
/* Function to copy a vector to another */
void copiarValor(float *a, float *b);
/* Function to load a bector from the database file */
int leerVectorCophir(float *dato, FILE *file);
/* Function to read a vector from the database file and store it column-wise in a matrix*/
int leerVectorTransCophir(float **dato, FILE *file, int col);
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

int main(int argc, char *argv[])
{
    validarDevice(0);
    validarArgumentos(argc, argv);
    validarNThreads();
         
    float **vector;
    float *vectorGPU;
    float **consulta;
    float *consultaGPU;
    Elem **resultado;
    Elem *resultadoGPU; 
    float *distancia;
    int cantConsultas;
    int cantVectores[1];
    int *cantVectoresGPU;
    int k;
    float dato[DIM];
    int i;
    int n_blocks = N_BLOCKS;
    char str_f[256];
    FILE *f_dist, *fquery;
    size_t pitch_M, pitch_Q, pitch_R, pitch_D;
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
        printf("\nERROR :: reading the vectors database file\n");
        return 0;
      }
    }

    fclose(f_dist);
    fflush(stdout);

    if ((fquery = fopen(argv[3], "r")) == NULL)
      printf("ERROR :: opening the query file: %s\n", argv[3]);
    fflush(stdout);
    	
    /* Loading queries to CPU memory */   
    for (i = 0; i < cantConsultas; i++) {
      if (leerVectorCophir(dato, fquery) == -1 || feof(fquery)) {
        printf("\n\nERROR :: cantConsultas mal establecida, Menos queries que las indicadas\n\n");
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
    if (cudaSuccess != cudaMallocPitch((void **)&distancia, &pitch_D, cantVectores[0]*sizeof(float), N_BLOCKS)) {
      printf("\nERROR :: cudaMallocPitch \n");
      cudaThreadExit();
      return 0;
    }
          
    /* Allocating GPU memory to the results matrix */
    if (cudaSuccess != cudaMallocPitch((void **)&resultadoGPU, &pitch_R, k*sizeof(Elem), cantConsultas)) {
      printf("\nERROR :: cudaMallocPitch \n");
      cudaThreadExit();
      return 0;
    }
        
    /*Allocating memory to cantVectoresGPU*/                                                              
    cudaMalloc((void **)&cantVectoresGPU, sizeof(int)); 

    /* Starting time measure */
    getrusage(RUSAGE_SELF, &r1);
    gettimeofday(&t1, 0);

    /* Copying vectors from CPU to GPU */
    for (i = 0; i < DIM; i++)
      if (cudaSuccess != cudaMemcpy((float *)((char *)vectorGPU + (i*(int)pitch_M)), (float *)(vector[i]), sizeof(float)*cantVectores[0], cudaMemcpyHostToDevice))
        printf("\nERROR :: cudaMemcpy\n");

    /* Copying queries from CPU to GPU */
    for (i = 0; i < cantConsultas; i++)
      if (cudaSuccess != cudaMemcpy((float *)((char *)consultaGPU + (i*(int)pitch_Q)), (float *)(consulta[i]), sizeof(float)*DIM, cudaMemcpyHostToDevice))
        printf("\nERROR :: cudaMemcpy\n");
 
    /*Copying quantity of vectors to process */ 
    cudaMemcpy(cantVectoresGPU, cantVectores, sizeof(int), cudaMemcpyHostToDevice);
   
    /* Processing queries in batches. Each query is processed by one CUDA Block */
    for(i = 0; i < cantConsultas; i += N_BLOCKS) 
    {
      if((cantConsultas-i) < N_BLOCKS && (cantConsultas-i) > 0) 
        n_blocks = (cantConsultas-i);

      /* Launching kernel that implements the GPU-Sel-kNN v2 */
      gpuSelKNN<<<n_blocks, N_THREADS>>>(vectorGPU, consultaGPU, resultadoGPU, distancia, i, cantVectoresGPU, pitch_M, pitch_Q, pitch_R, pitch_D, k);
    }

    /* Copying results from GPU to CPU */
    for (i = 0; i < cantConsultas; i++)
      if (cudaSuccess != cudaMemcpy((Elem *)(resultado[i]), (Elem *)((char *)resultadoGPU + (i*(int)pitch_R)), sizeof(Elem)*k, cudaMemcpyDeviceToHost))
        printf("\nERROR :: cudaMemcpy device to host\n");
              
    cudaDeviceSynchronize();                                                                                                     
 
    /* Stopping measuring time */
    gettimeofday(&t2, 0);
    getrusage(RUSAGE_SELF, &r2);

    imprimirTiempoEjecucion(r1, r2, t1, t2, k);
    imprimirResultados(resultado, cantConsultas, k);

    cudaFree(vectorGPU);
    cudaFree(consultaGPU);
    cudaFree(resultadoGPU);
    cudaFree(cantVectoresGPU);
    free(vector);
    free(consulta);
    cudaThreadExit();
                                                                                                                                                   
    return 0;
}

__global__ void gpuSelKNN(float *vectorGPU, float *consultaGPU, Elem *resultadoGPU, float *distancia, int posicionConsulta, int *cantVectoresGPU, size_t pitch_M, size_t pitch_Q, size_t pitch_R, size_t pitch_D, int k)
{
    __shared__ Elem distemp[N_THREADS];
    __shared__ Elem distemp2[N_WARP];
    __shared__ Elem marcado[N_THREADS];
    __shared__ float query[DIM];
    int tid = threadIdx.x; 
    int bid = blockIdx.x; 
    int i, j;
    int n_db = cantVectoresGPU[0];
    int indRes2 = 0;
    Elem menorlocal;
    float distlocal;
   
    /* Copying query to be solved by the current CUDA Block to shared memory */
    if(tid < DIM )
      query[tid] = ((float *)((char *)consultaGPU + ((bid+posicionConsulta)*(int)pitch_Q)))[tid];
	
    __syncthreads();

    /* Calculating distances between vectors and the query (which is being solved by the current CUDA Block) */
    for(i = tid; i < n_db ; i += blockDim.x) 
    {
      distlocal=0.0;
      for (j = 0; j < DIM; j++) 
      {
        distlocal += ( ((float *)((char *)vectorGPU + (j*(int)pitch_M)))[i] - query[j]) *
                     ( ((float *)((char *)vectorGPU + (j*(int)pitch_M)))[i] - query[j]);
      }			
      ((float *)((char *)distancia + (bid*(int)pitch_D)))[i] = sqrtf(distlocal);
    }

    __syncthreads();
	
    /* Obtaining the k nearest elements to the query */
    while(indRes2 < k) 
    {		
      menorlocal.dist = MAXFLOAT;

      /* Each thread gets the element with the lowest distance from its elements to the query. N_THREADS elements are obtained. */
      for(i = tid; i < n_db ; i += blockDim.x) 
      {
        distlocal = ((float *)((char *)distancia + (bid*(int)pitch_D)))[i]; 
        if(menorlocal.dist > distlocal) 
        {
          menorlocal.dist = distlocal;
          menorlocal.ind = i;
        }
      }			
		
      distemp[tid] = menorlocal;
	
      __syncthreads();

      /* Each thread of the first warp gets the element with the lowest distance from its elements to the query. N_WARP elements are obtained */
      if(tid < N_WARP && tid < blockDim.x) 
      {
        menorlocal = distemp[tid];
        for(i = tid; i < blockDim.x && i < n_db; i += N_WARP) 
        {
          if(distemp[i].dist < menorlocal.dist) 
            menorlocal = distemp[i];
        }
        distemp2[tid] = menorlocal;
      }
      		
      /* The first thread of the CUDA Block gets the element with the lowest distance */
      if(tid == 0) 
      {
        for(i = 1; i < N_WARP; i++) 
        {
          if(distemp2[i].dist < menorlocal.dist) 
            menorlocal = distemp2[i];
        }
        marcado[indRes2] = menorlocal;
        ((float *)((char *)distancia + (bid*(int)pitch_D)))[menorlocal.ind] = MAXFLOAT;
      }
      indRes2++;
      __syncthreads();	
    }
    
    /* Storing results */
    if(tid < k) 
      ((Elem *)((char *)resultadoGPU + ((bid+posicionConsulta)*(int)pitch_R)))[tid] = marcado[tid];

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
          printf("\nERROR :: character ',' not found (the coordinates must be separated by ','.\n");
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
          printf("\nERROR :: character ',' not found between coordinates\n");
          return ERROR;
        }
    }
    return 1;
}

void validarDevice(int i) {
    if (cudaSuccess != cudaSetDevice(i)) {
      printf("\n\nERROR :: cudaSetDevice\n\n");
      fflush(stdout); 
      exit(EXIT_SUCCESS);  
    }
}

bool isFile(char * entrada) { 
    FILE *file;
    bool bFile = false;
  
    if(file = fopen(entrada,"r")) {
      fclose(file);
      bFile = true;
    }

    return bFile;
}

bool isNumber(char *number) {
    bool formatoNumero = (*number)?true:false;
  
    for(; *number && formatoNumero; number++)
      if(!isdigit(*number))
        formatoNumero = false;
    return formatoNumero;
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

void imprimirResultados(Elem **resultado, int filas, int columnas) {
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
