#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "svd3_cuda.h"

__device__ __forceinline__ int compute_determinant(float *c) {
	int det = (c[(0*3+0)] * ((c[(1*3+1)] * c[(2*3+2)]) - (c[(2*3+1)] * c[(1*3+2)]))) -
		(c[(1*3+0)] * ((c[(0*3+1)] * c[(2*3+2)]) - (c[(2*3+1)] * c[(0*3+2)]))) +
		(c[(2*3+0)] * ((c[(0*3+1)] * c[(1*3+2)]) - (c[(1*3+1)] * c[(0*3+2)])));
	return det;
}
__device__ __forceinline__
void compute_matrix_multiplication(float *A, float *B, float *C, bool b_transpose) {
	float trans_B[9] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
	int i = 0;
	int j = 0;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			if (b_transpose == true) {
				trans_B[i * 3 + j] = B[j * 3 + i];
			}
			else {
				trans_B[j * 3 + i] = B[j * 3 + i];
			}
		}
	}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			C[((j)*(3)) + (i)] = 0.0;
			for (int k = 0; k < 3; k++) {
				C[((j)*(3)) + (i)] = (C[((j)*(3)) + (i)] + (A[((k)*(3)) + (i)] * trans_B[((j)*(3)) + (k)]));
			}
		}
	}

}



__global__
void device_compute_rotations(int num_vertices, int col_indices_size, int *dev_row_ptrs, int *dev_column_indices, double *dev_weight_values, double *dev_original_x, double *dev_original_y, double *dev_original_z, double *dev_solution_x, double *dev_solution_y, double *dev_solution_z, double *dev_rot_matrices) {
	int vi_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (vi_id < num_vertices) {
		int i = 0, j = 0;
		float cov_matrix[9] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
		float devPtrA[9] = { 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 };
		float devPtrB[9] = { 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 };
		float devPtrC[9] = { 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 };
		float rotation_matrix[9] = { 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 };
		float pij[3] = { 0.0, 0.0, 0.0 };
		float qij[3] = { 0.0, 0.0, 0.0 };
		int vj_id;
		float wij;
		int vector_indx_q;
		int vector_indx_p;
		int strt = dev_row_ptrs[vi_id]; // start index for adjacency list
		int end = (col_indices_size) - 1; // last index for adjacency list
		if ((vi_id + 1) < num_vertices) {
			end = dev_row_ptrs[(vi_id + 1)] - 1;
		}
		for (; strt <=end; strt++) {
			vj_id = dev_column_indices[strt];
			wij = dev_weight_values[strt];
			pij[0] = dev_original_x[vi_id] - dev_original_x[vj_id];
			pij[1] = dev_original_y[vi_id] - dev_original_y[vj_id];
			pij[2] = dev_original_z[vi_id] - dev_original_z[vj_id];
			qij[0] = dev_solution_x[vi_id] - dev_solution_x[vj_id];
			qij[1] = dev_solution_y[vi_id] - dev_solution_y[vj_id];
			qij[2] = dev_solution_z[vi_id] - dev_solution_z[vj_id];
			vector_indx_q = 0;
			vector_indx_p = 0;
			for (i = 0; i < 3; i++) {
				for (j = 0; j < 3; j++) {
					if (i == 0) {
						devPtrB[((j)*(3)) + (i)] = qij[vector_indx_q++];
					}
					else {
						devPtrB[((j)*(3)) + (i)] = 0.0f;
					}
					if (j == 0) {
						devPtrA[((j)*(3)) + (i)] = pij[vector_indx_p++];
					}
					else {
						devPtrA[((j)*(3)) + (i)] = 0.0f;
					}
				}
			}
			for (i = 0; i < 3; i++) {
				for (j = 0; j < 3; j++) {
					devPtrC[((j)*(3)) + (i)] = 0;
					for (int kk = 0; kk < 3; kk++) {
						devPtrC[((j)*(3)) + (i)] = (devPtrC[((j)*(3)) + (i)] + (devPtrA[((kk)*(3)) + (i)] * devPtrB[((j)*(3)) + (kk)]));
					}
				}
			}
			for (i = 0; i < 3; i++) {
				for (j = 0; j < 3; j++) {
					cov_matrix[((j)*(3)) + (i)] += (devPtrC[((j)*(3)) + (i)] * wij);
				}
			}
		}

		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				devPtrA[((j)*(3)) + (i)] = (float)0.0;
				devPtrB[((j)*(3)) + (i)] = (float)0.0;
				devPtrC[((j)*(3)) + (i)] = (float)0.0;
			}
		}
		svd(cov_matrix[0], cov_matrix[3], cov_matrix[6],
			cov_matrix[1], cov_matrix[4], cov_matrix[7],
			cov_matrix[2], cov_matrix[5], cov_matrix[8],
			devPtrA[0], devPtrA[3], devPtrA[6],
			devPtrA[1], devPtrA[4], devPtrA[7],
			devPtrA[2], devPtrA[5], devPtrA[8],
			devPtrB[0], devPtrB[3], devPtrB[6],
			devPtrB[1], devPtrB[4], devPtrB[7],
			devPtrB[2], devPtrB[5], devPtrB[8],
			devPtrC[0], devPtrC[3], devPtrC[6],
			devPtrC[1], devPtrC[4], devPtrC[7],
			devPtrC[2], devPtrC[5], devPtrC[8]);

		compute_matrix_multiplication(devPtrC, devPtrA, rotation_matrix, true);
//		if (compute_determinant(rotation_matrix) < 0) {
//			// multiply last column of devPtrA with -1 and compute multiplication again
//			devPtrA[3 * 2 + 0] = devPtrA[3 * 2 + 0] * -1;
//			devPtrA[3 * 2 + 1] = devPtrA[3 * 2 + 1] * -1;
//			devPtrA[3 * 2 + 2] = devPtrA[3 * 2 + 2] * -1;
//			compute_matrix_multiplication(devPtrC, devPtrA, rotation_matrix, true);
//		}
//
//		for(int i=0;i<3;i++){
//			for(int j=0;j<3;j++){
//				devPtrB[(j*3 + i)] = dev_agr_rotations[(vi_id * 9)+(j*3 + i)];
//			}
//		}
//
//		compute_matrix_multiplication(devPtrB, rotation_matrix, devPtrC, false);

		int strt_indx = (vi_id * 9);
		int end_indx = strt_indx + 9;
		int cov_indx = 0;
		for (i = strt_indx; i < end_indx; i++) {
			dev_rot_matrices[i] = rotation_matrix[cov_indx++];
		}

	}

}

__global__
void device_compute_energies(int num_vertices, int col_indices_size, int *dev_row_ptrs, int *dev_column_indices, double *dev_weight_values, double *dev_original_x, double *dev_original_y, double *dev_original_z, double *dev_solution_x, double *dev_solution_y, double *dev_solution_z, double *dev_rot_matrices, double *dev_energies) {
	int vi_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (vi_id < num_vertices) {
		int strt_indx = (vi_id * 9);
		double energy=0.0f;
		double pij[3] = { 0.0, 0.0, 0.0 };
		double qij[3] = { 0.0, 0.0, 0.0 };
		double vij[3];
		int strt = dev_row_ptrs[vi_id]; // start index for adjacency list
		int end = (col_indices_size) - 1; // last index for adjacency list
		if ((vi_id + 1) < num_vertices) {
			end = dev_row_ptrs[(vi_id + 1)] - 1;
		}

		for (; strt <=end; strt++) {
			int vj_id = dev_column_indices[strt];
			double wij = dev_weight_values[strt];
			vij[0]=0.0f;vij[1]=0.0f;vij[2]=0.0f;
			pij[0] = dev_original_x[vi_id] - dev_original_x[vj_id];
			pij[1] = dev_original_y[vi_id] - dev_original_y[vj_id];
			pij[2] = dev_original_z[vi_id] - dev_original_z[vj_id];
			qij[0] = dev_solution_x[vi_id] - dev_solution_x[vj_id];
			qij[1] = dev_solution_y[vi_id] - dev_solution_y[vj_id];
			qij[2] = dev_solution_z[vi_id] - dev_solution_z[vj_id];

			for (int i=0;i<3;i++){
				for (int j=0;j<3;j++){
					vij[i]+=( dev_rot_matrices[((j)*(3)) + (i) + strt_indx]*pij[j]);
				}
			}
			energy += wij*(pow((qij[0] - vij[0]),2)+pow((qij[1] - vij[1]),2)+pow((qij[2] - vij[2]),2));

		}

		dev_energies[vi_id] = energy;

	}

}


__global__
void device_compute_rhs(int num_vertices, int col_indices_size, int control_size, int *dev_row_ptrs, int *dev_column_indices, double *dev_weight_values, double *dev_original_x, double *dev_original_y, double *dev_original_z, double *dev_rot_matrices, int *dev_index_map, double *dev_tmp_B, double *dev_control_vals, int *dev_perm){
	int vi_id = blockIdx.x * blockDim.x + threadIdx.x;

	if ((vi_id < num_vertices) && (dev_index_map[vi_id]!=-1)) {
		int i = 0, j = 0;
		double pij[3] = { 0.0, 0.0, 0.0 };
		double cc[3] = { 0.0, 0.0, 0.0 };
		int strt = dev_row_ptrs[vi_id]; // start index for adjacency list
		int end = (col_indices_size) - 1; // last index for adjacency list
		if ((vi_id + 1) < num_vertices) {
			end = dev_row_ptrs[(vi_id + 1)] - 1;
		}
		int vi_indx = vi_id*9;
		for (; strt <=end; strt++) {
			double devPtrA[9] = { 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 };
			int vj_id = dev_column_indices[strt];
			double wij = dev_weight_values[strt];
			pij[0] = dev_original_x[vi_id] - dev_original_x[vj_id];
			pij[1] = dev_original_y[vi_id] - dev_original_y[vj_id];
			pij[2] = dev_original_z[vi_id] - dev_original_z[vj_id];

			int vj_indx = vj_id*9;
			for (i = 0; i < 3; i++) {
				for (j = 0; j < 3; j++) {
					int ij_indx = (i*3 + j);
					devPtrA[ij_indx] += ((wij*dev_rot_matrices[vi_indx+(j*3 + i)]) + (wij*dev_rot_matrices[vj_indx+(j*3 + i)]));
					cc[i] += (devPtrA[ij_indx]*pij[j]);
				}
			}
		}
		int indx = dev_index_map[vi_id];
		int rhs_size = num_vertices-control_size;

		dev_tmp_B[indx] = cc[0] - dev_control_vals[indx];
		dev_tmp_B[indx+rhs_size] = cc[1] - dev_control_vals[indx+rhs_size];
		dev_tmp_B[indx+(rhs_size*2)] = cc[2] - dev_control_vals[indx+(rhs_size*2)];

	}
}


__global__
void device_apply_acceleration(int num_vertices, int b_siz, double *dev_solution_x, double *dev_solution_y, double *dev_solution_z, double *dev_B, int *dev_index_map, bool apply_accel){
	int vi_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (vi_id < num_vertices) {
		if(dev_index_map[vi_id]!=-1){
			if(apply_accel==true){
				dev_solution_x[vi_id] = dev_B[dev_index_map[vi_id]] + (0.938f * (dev_B[dev_index_map[vi_id]] - dev_solution_x[vi_id]));
				dev_solution_y[vi_id] = dev_B[dev_index_map[vi_id]+b_siz]+ (0.938f * (dev_B[dev_index_map[vi_id]+b_siz] - dev_solution_y[vi_id]));
				dev_solution_z[vi_id] = dev_B[dev_index_map[vi_id]+(b_siz*2)]+ (0.938f * (dev_B[dev_index_map[vi_id]+(b_siz*2)] - dev_solution_z[vi_id]));
			}
			else{
				dev_solution_x[vi_id] = dev_B[dev_index_map[vi_id]] ;
				dev_solution_y[vi_id] = dev_B[dev_index_map[vi_id]+b_siz];
				dev_solution_z[vi_id] = dev_B[dev_index_map[vi_id]+(b_siz*2)];
			}

		}

	}
}


__global__
void dev_permute(double *dev_temp_B, double *dev_B, int *dev_perm, int row_num){
	int row_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(row_id < row_num){
		int p = dev_perm[row_id];
		dev_B[row_id] = dev_temp_B[p];
		dev_B[row_id+row_num] = dev_temp_B[p+row_num];
		dev_B[row_id+(row_num*2)] = dev_temp_B[p+(row_num*2)];
	}
}

__global__
void dev_inverse_permute(double *dev_temp_B, double *dev_B, int *dev_perm, int row_num){
	int row_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(row_id < row_num){
		int p = dev_perm[row_id];
		dev_B[p] = dev_temp_B[row_id];
		dev_B[p+row_num] = dev_temp_B[row_id+row_num];
		dev_B[p+(row_num*2)] = dev_temp_B[row_id+(row_num*2)];
	}
}

extern "C" void call_device_compute_rotations(int num_vertices, int col_indices_size, int *dev_row_ptrs, int *dev_column_indices, double *dev_weight_values, double *dev_original_x, double *dev_original_y, double *dev_original_z, double *dev_solution_x, double *dev_solution_y, double *dev_solution_z, double *dev_rot_matrices){
	int blocks = (num_vertices / 128)+1;
	int threads = 128;
	device_compute_rotations<<<blocks, threads>>>(num_vertices, col_indices_size, dev_row_ptrs, dev_column_indices, dev_weight_values, dev_original_x, dev_original_y, dev_original_z, dev_solution_x, dev_solution_y, dev_solution_z, dev_rot_matrices);
	cudaDeviceSynchronize();
}

extern "C" void call_device_compute_rhs(int num_vertices, int col_indices_size, int control_size, int *dev_row_ptrs, int *dev_column_indices, double *dev_weight_values, double *dev_original_x, double *dev_original_y, double *dev_original_z, double *dev_rot_matrices, int *dev_index_map, double *dev_tmp_B, double *dev_control_vals, int *dev_perm){
	int blocks = (num_vertices / 256)+1;
	int threads = 256;
	device_compute_rhs<<<blocks, threads>>>(num_vertices, col_indices_size, control_size, dev_row_ptrs, dev_column_indices, dev_weight_values, dev_original_x, dev_original_y, dev_original_z, dev_rot_matrices, dev_index_map, dev_tmp_B, dev_control_vals, dev_perm);
	cudaDeviceSynchronize();
}


extern "C" void call_dev_permute(double *dev_temp_B, double *dev_B, int *dev_perm, int row_num){
	int blocks = ((row_num) / 256)+1;
	int threads = 256;
	dev_permute<<<blocks, threads>>>(dev_temp_B, dev_B, dev_perm, row_num);
	cudaDeviceSynchronize();
}

extern "C" void call_dev_inverse_permute(double *dev_temp_B, double *dev_B, int *dev_perm, int row_num){
	int threads = 256;
	int blocks = (row_num / threads)+1;
	dev_inverse_permute<<<blocks, threads>>>(dev_temp_B, dev_B, dev_perm, row_num);
	cudaDeviceSynchronize();
}

extern "C" double call_device_compute_energies(int num_vertices, int col_indices_size, int *dev_row_ptrs, int *dev_column_indices, double *dev_weight_values, double *dev_original_x, double *dev_original_y, double *dev_original_z, double *dev_solution_x, double *dev_solution_y, double *dev_solution_z, double *dev_rot_matrices, double *dev_energies){
	int blocks = (num_vertices / 256)+1;
	int threads = 256;
	device_compute_energies<<<blocks, threads>>>(num_vertices, col_indices_size, dev_row_ptrs, dev_column_indices, dev_weight_values, dev_original_x, dev_original_y, dev_original_z, dev_solution_x, dev_solution_y, dev_solution_z, dev_rot_matrices, dev_energies);
	cudaDeviceSynchronize();
	thrust::device_ptr<double> dptr(dev_energies);
	thrust::device_ptr<double> last  = dptr + num_vertices;
	double res = thrust::reduce(dptr, last);
	return res;
}

extern "C" void call_apply_acceleration(int num_vertices, int b_siz, double *dev_solution_x, double *dev_solution_y, double *dev_solution_z, double *dev_B, int *dev_index_map, bool apply_accel){
	int threads = 256;
	int blocks = (num_vertices / threads)+1;
	device_apply_acceleration<<<blocks, threads>>>(num_vertices, b_siz, dev_solution_x, dev_solution_y, dev_solution_z, dev_B, dev_index_map, apply_accel);
	cudaDeviceSynchronize();
}

