#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N 20000

double Norma(double* Vector) {
	double sum = 0;
	double final_number;
	int i;
	#pragma omp parallel for reduction(+:sum)
	for (i = 0; i < N; ++i) {
		sum += pow(Vector[i], 2);
	}
	final_number = sqrt(sum);

	return final_number;
}

double* Multiplication(double** Matrix, double* Vector) {
	double* final_vector = (double*)calloc(N ,sizeof(double));
	if (!final_vector) return NULL;
	int i;
	#pragma omp parallel for
	for (i = 0; i < N; ++i) {
		double sum = 0;
		for (int j = 0; j < N; ++j) {
			sum += Matrix[i][j] * Vector[j];
		}
		final_vector[i] = sum;
	}

	return final_vector;
}

double* Minus(double* first_vector, double* second_vector) {
    double* final_vector = (double*)malloc(sizeof(double)*N);
	if (!final_vector) return NULL;
	int i;
	#pragma omp parallel for
	for (i = 0; i < N; ++i) {
		final_vector[i] = first_vector[i] - second_vector[i];
	}

	return final_vector;
}

double* Multiplication_Scalar(double* Vector, double Scalar) {
	double* final_vector = (double*)malloc(sizeof(double) * N);
	if (!final_vector) return NULL;
	int i;
	#pragma omp parallel for	
	for (i = 0; i < N; ++i) {
		final_vector[i] = Vector[i] * Scalar;
	}
	
	free(Vector);
	return final_vector;
}


int main(void)
{
	omp_set_num_threads(1);
	double** matrix = (double**)calloc(N, sizeof(double*));
	double* desired_vector = (double*)calloc(N, sizeof(double));
	double* arbitrary_vector = (double*)calloc(N, sizeof(double));
	if (!matrix) return 1;
	if (!desired_vector) return 1;
	if (!arbitrary_vector) return 1;
		
	for (int i = 0; i < N; ++i) {
		matrix[i] = (double*)calloc(N, sizeof(double));
		if (!matrix[i]) return 1;
	}

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			if (i == j) { 
				matrix[j][i] = 2.0;
			}
			else {
				matrix[j][i] = 1.0;
			}
		}
		arbitrary_vector[i] = sin((2 * 3.14159 * i)/N);
	}

	double* vector = Multiplication(matrix, arbitrary_vector);
	double* mult = Multiplication(matrix, desired_vector);
    double* final = Minus(mult, vector);
	if (!vector) return 1;
	if (!mult) return 1;
	if (!final) return 1;

	double start = omp_get_wtime();
	while ((Norma(final) / Norma(vector)) > pow(10, -5)) {

		final = Multiplication_Scalar(final, 0.01);
		desired_vector = Minus(desired_vector, final);

		mult = Multiplication(matrix, desired_vector);
		final = Minus(mult, vector);
	}
	double end = omp_get_wtime();
	printf("%f", end - start);

	for (int i = 0; i < N; ++i) {
		free(matrix[i]);
	}
	free(matrix);
	free(desired_vector);
	free(vector);
	free(mult);
	free(final);
	return 0;
}
