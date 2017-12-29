
#define GPU_BLAS

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <cusparse_v2.h>
#include <thread>
#include <mutex>
#include <vector>
#include <map>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include "cholmod.h"


extern "C" void call_device_compute_rotations(int num_vertices, int col_indices_size, int *dev_row_ptrs, int *dev_column_indices, double *dev_weight_values, double *dev_original_x, double *dev_original_y, double *dev_original_z, double *dev_solution_x, double *dev_solution_y, double *dev_solution_z, double *dev_rot_matrices);
extern "C" void call_device_compute_rhs(int num_vertices, int col_indices_size, int control_size, int *dev_row_ptrs, int *dev_column_indices, double *dev_weight_values, double *dev_original_x, double *dev_original_y, double *dev_original_z, double *dev_rot_matrices, int *dev_index_map, double *dev_tmp_B, double *dev_control_vals, int *dev_perm);
extern "C" void call_dev_permute(double *dev_temp_B, double *dev_B, int *dev_perm, int row_num);
extern "C" void call_dev_inverse_permute(double *dev_temp_B, double *dev_B, int *dev_perm, int row_num);
extern "C" double call_device_compute_energies(int num_vertices, int col_indices_size, int *dev_row_ptrs, int *dev_column_indices, double *dev_weight_values, double *dev_original_x, double *dev_original_y, double *dev_original_z, double *dev_solution_x, double *dev_solution_y, double *dev_solution_z, double *dev_rot_matrices, double *dev_energies);
extern "C" void call_apply_acceleration(int num_vertices, int b_siz, double *dev_solution_x, double *dev_solution_y, double *dev_solution_z, double *dev_B, int *dev_index_map, bool apply_accel);


class Timer {
public:
	Timer() : beg_(clock_::now()) {}
	void reset() { beg_ = clock_::now(); }
	double elapsed() const {
		return std::chrono::duration_cast<second_>
			(clock_::now() - beg_).count();
	}

private:
	typedef std::chrono::high_resolution_clock clock_;
	typedef std::chrono::duration<double, std::ratio<1> > second_;
	std::chrono::time_point<clock_> beg_;
};


class Arap_mesh_deformation {

public:
	double avg_solver_time;
	double avg_rotation_time;
	std::vector<double> solution_x;
	std::vector<double> solution_y;
	std::vector<double> solution_z;
	std::vector<bool> is_ctrl_map;

private:
	Eigen::MatrixXd& vertices;
	Eigen::MatrixXi& triangles;
	Eigen::SparseMatrix<double,Eigen::RowMajor> e_weights;
	std::vector<int> vertex_index_map;
	std::vector<int> control_vertices;
	std::vector<int> sorted_control_vertices;
	std::vector<Eigen::Matrix3d> rotation_matrix;
	std::vector<double> original_x;
	std::vector<double> original_y;
	std::vector<double> original_z;
	std::map<int, std::vector<float>> control_columns;
	std::thread *weighting_thread;


	double *control_vals;
	bool preprocess_successful;
	bool is_ids_cal_done;
	bool factorization_done;
	bool laplacian_matrix_ready;
	bool deformation_done;
	double theta;
	int factor_nzmax;

	cholmod_sparse *AA = NULL;
	cholmod_factor *L = NULL;
	cholmod_common common,*c;


	// CUDA VARIABLES
	double *dev_BZ = NULL;
	double *d_z = NULL;
	double *dev_temp_B = NULL;
	int *dev_perm = NULL;

	double *dev_vals_lower = NULL;
	int *dev_col_indices_lower = NULL;
	int *dev_row_indices_lower = NULL;

	double *dev_vals_upper = NULL;
	int *dev_col_indices_upper = NULL;
	int *dev_row_indices_upper = NULL;

	cusparseHandle_t handle = NULL;
	cusparseMatDescr_t descr_L = NULL;
	cusparseMatDescr_t descr_Lt = NULL;

	csrsv2Info_t info_L = NULL;
	csrsv2Info_t info_Lt = NULL;

	int pBufferSize_L, pBufferSize_Lt;

	void *pBuffer = NULL;

	int *dev_index_map = NULL;
	double *dev_control_vals = NULL;
	double *dev_rot_matrices = NULL;
	double *dev_energies = NULL;
	double *dev_original_x = NULL;
	double *dev_original_y = NULL;
	double *dev_original_z = NULL;
	double *dev_solution_x = NULL;
	double *dev_solution_y = NULL;
	double *dev_solution_z = NULL;
	double *dev_weight_values = NULL;
	int *dev_column_indices = NULL;
	int *dev_row_ptrs = NULL;



public:

	Arap_mesh_deformation(Eigen::MatrixXd &V, Eigen::MatrixXi &F) : vertices(V), triangles(F) {
		Eigen::initParallel();
		is_ctrl_map = std::vector<bool>(vertices.rows(), false);
		vertex_index_map = std::vector<int>(vertices.rows(), -1);
		preprocess_successful = false;
		is_ids_cal_done = false;
		factorization_done = false;
		laplacian_matrix_ready = false;
		deformation_done = false;
		avg_rotation_time = 0.0;
		avg_solver_time = 0.0;
		factor_nzmax = 0;
		theta = 0.938;// for condition number=100, theta=0.81 and for condition number=1000, theta=0.938
		weighting_thread = new std::thread(&Arap_mesh_deformation::cal_weights, this);
		c = &common;
		cholmod_start(c);
		c->final_super = 0;
		c->final_ll = 1; //in order to retrieve the factor L

	}

	// insert constraint vertex
	bool insert_control_vertex(int vid) {
		// factorization needs to be done again once this function is called.
		if (is_control_vertex(vid)) {
			return false;
		}
		if (factorization_done == true) {
			factorization_done = false;

			 if(AA!=NULL) {
				cholmod_free_sparse(&AA, c);
				cholmod_free_factor(&L, c);
			}
		}
		control_vertices.push_back(vid);
		is_ctrl_map[vid] = true;
		deformation_done = false;
		return true;
	}

	bool preprocess() {
		init_vars();
		calculate_laplacian_and_factorize();
		return preprocess_successful;
	}

	// set new position for the specified constraint vertex
	void set_target_position(int vid, const Eigen::Vector3d& target_position) {
		// vars should have been initialized before calling this function
		init_vars();
		if (!is_control_vertex(vid)) {
			return;
		}
		solution_x[vid] = target_position(0);
		solution_y[vid] = target_position(1);
		solution_z[vid] = target_position(2);
		update_device_vertices();
	}

	void deform(unsigned int iterations, double tolerance, std::string fname="", bool flg=false)
	{
		if (!preprocess_successful) {
			return;
		}
		if(deformation_done)
			return;

		double curr_energy = 0;
		double prev_energy;
		double e_dif;
		for (int itr = 0; itr < iterations; ++itr) {
			calculate_optimal_rotations();
			if (itr != 0)
				calculate_target_positions(true); // apply acceleration
			else
				calculate_target_positions(false);

			if (tolerance > 0.0 && (itr + 1) < iterations) {
				prev_energy = curr_energy;
				curr_energy = get_energy();
				if (itr != 0) {
					e_dif = std::abs((prev_energy - curr_energy) / curr_energy);
					if (e_dif < tolerance) {
						deformation_done = true;
						break;
					}
				}
			}
		}
		gpuAssert(cudaMemcpy(solution_x.data(), dev_solution_x, vertices.rows() * sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(solution_y.data(), dev_solution_y, vertices.rows() * sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(solution_z.data(), dev_solution_z, vertices.rows() * sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	}


	bool is_control_vertex(int vid) const{
		return is_ctrl_map[vid];
	}

	void perform_cleanup(){
		cuda_free_vertices();
	}

	void set_mesh_data(Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
		cuda_update_vertex_positions(solution_x, solution_y, solution_z, vertices.rows()); // update solution for cuda
	}

	void update_mesh(){
		for(int i=0;i<vertices.rows();i++){
			vertices.row(i) = Eigen::RowVector3d(solution_x[i], solution_y[i], solution_z[i]);
		}
	}
	void update_device_vertices(){
		cuda_update_vertex_positions(solution_x, solution_y, solution_z, vertices.rows()); // update solution for cuda
	}
	//Since the cotangent weights are a representation based on angles, they preserve angles (and, consequently, area)
	//much better during reconstruction. The result is that the surface tends to interpolate more "correctly" and in a more
	//visually appealing way, and in a way that actually appears smoother.
	void cal_weights() {
		int siz = vertices.rows();
		//adjacent_vertices.resize(siz);
		e_weights = Eigen::SparseMatrix<double, Eigen::RowMajor>(siz, siz);
		e_weights.reserve(Eigen::VectorXi::Constant(siz, 15));// assuming at most 15 neighbors per vertex
		Eigen::Matrix<int, 3, 3> edges;
		edges <<
			1, 2, 0,
			2, 0, 1,
			0, 1, 2;
		for (int i = 0; i < triangles.rows(); i++) {
			for (int e = 0; e<edges.rows(); e++) {
				int v0 = triangles(i, edges(e, 0)); // considering v0-v1 is the edge being considered
				int v1 = triangles(i, edges(e, 1));
				int v2 = triangles(i, edges(e, 2));
				double res = get_cot(v0, v2, v1, vertices);
				if (e_weights.coeff(v0, v1) == 0) {
					e_weights.coeffRef(v0, v1) = (res / 2.0);
					e_weights.coeffRef(v1, v0) = (res / 2.0);
				}
				else {
					e_weights.coeffRef(v0, v1) = (((e_weights.coeff(v0, v1) * 2.0) + res) / 2.0);
					e_weights.coeffRef(v1, v0) = (((e_weights.coeff(v1, v0) * 2.0) + res) / 2.0);
				}
			}
		}
		e_weights.makeCompressed();
		// copy weight matrix to GPU in csr format
		cuda_init_weights(e_weights.valuePtr(), e_weights.innerIndexPtr(), e_weights.outerIndexPtr(),e_weights.rows()+1,e_weights.nonZeros());

	}

	~Arap_mesh_deformation(){
		perform_cleanup();
	}

private:

	// Using Cotangent formula from: http://people.eecs.berkeley.edu/~jrs/meshpapers/MeyerDesbrunSchroderBarr.pdf
	// Cot = cos/sin ==> using langrange's identity ==> a.b/sqrt(a^2 * b^2 - (a.b)^2)
	double get_cot(int v0, int v1, int v2, Eigen::MatrixXd &V) {
		typedef Eigen::Vector3d Vector;
		Vector a(V(v0, 0) - V(v1, 0), V(v0, 1) - V(v1, 1), V(v0, 2) - V(v1, 2));
		Vector b(V(v2, 0) - V(v1, 0), V(v2, 1) - V(v1, 1), V(v2, 2) - V(v1, 2));
		double dot_ab = a.dot(b);
		Vector cross_ab = a.cross(b);
		double divider = cross_ab.norm(); 
		if (divider == 0) {
			return 0.0;
		}
		return (std::max)(0.0, (dot_ab / divider));
	}

	void init_vars() {
		if (is_ids_cal_done)
			return;
		is_ids_cal_done = true;
		int ros_size = vertices.rows();

		original_x.resize(ros_size);
		original_y.resize(ros_size);
		original_z.resize(ros_size);
		solution_x.resize(ros_size);
		solution_y.resize(ros_size);
		solution_z.resize(ros_size);

		for (int i = 0; i < ros_size; i++) {
			original_x[i] = vertices(i, 0);
			original_y[i] = vertices(i, 1);
			original_z[i] = vertices(i, 2);

			solution_x[i] = vertices(i, 0);
			solution_y[i] = vertices(i, 1);
			solution_z[i] = vertices(i, 2);

		}
		cuda_init_vertices(original_x, original_y, original_z, solution_x, solution_y, solution_z, ros_size);

	}

	int in_range(int vid) {
		for(int i=sorted_control_vertices.size()-1;i>=0;i--) {
			if(vid > sorted_control_vertices[i]) {
				return (i+1);
			}
		}
		return 0;
	}

	// retrieve factors and permutation vector
	void retrieve_factor(double *contrl_vals){

		if (!cholmod_change_factor (L->xtype, L->is_ll, 0, 1, 1, L, c))
		{
			std::cout << "\nCHANGE FACTOR FAILED!!\n";
			return;
		}

		int rows = (int)L->n;
		int *perm = (int *)L->Perm;
		set_perm(perm, rows);

		cholmod_sparse * Ft = cholmod_factor_to_sparse(L, c);

		double *vals_upper = (double *)Ft->x;
		int *col_indices_upper = (int *)Ft->i;
		int *row_indices_upper = (int *)Ft->p;

		cholmod_sparse *F = cholmod_transpose(Ft, 1, c);

		factor_nzmax = (int)F->nzmax;
		double *vals_lower = (double *)F->x;
		int *col_indices_lower = (int *)F->i;
		int *row_indices_lower = (int *)F->p;

		set_factors(vals_lower, col_indices_lower, row_indices_lower, vals_upper, col_indices_upper, row_indices_upper, perm, contrl_vals, factor_nzmax, rows);

	}

	// compute constraint columns which would be subtracted from RHS of linear system
	void copy_control_sub(int strt, int end,int i, int a_siz, std::vector<float> &vectr){
		for(int k=strt;k<=end;k++){
			control_vals[k] += ((solution_x[control_vertices[i]])*vectr[k]);
			control_vals[k+a_siz] += ((solution_y[control_vertices[i]])*vectr[k]);
			control_vals[k+(a_siz*2)] += ((solution_z[control_vertices[i]])*vectr[k]);
		}
	}

	// factorize Laplacian matrix excluding rows/cols corresponding to constraint vertices
	// and copy cholesky factors to GPU
	void calculate_laplacian_and_factorize() {
		if (factorization_done)
			return;
		factorization_done = true;
		if (weighting_thread->joinable())
			weighting_thread->join();

		size_t siz = vertices.rows();
		sorted_control_vertices = control_vertices;
		std::sort(sorted_control_vertices.begin(), sorted_control_vertices.end());
		int new_size = static_cast<int>((siz-control_vertices.size()));
		for(int z=0;z<control_vertices.size();z++) {
			control_columns[control_vertices[z]] = std::vector<float>(new_size,0.0f);
		}
		int row=0,col=0;
		cholmod_triplet *cholmodTriplets = cholmod_allocate_triplet(new_size, new_size,new_size*10,1,CHOLMOD_REAL,c);
		double *weight_vals = e_weights.valuePtr();
		int *col_indices = e_weights.innerIndexPtr();
		int *row_ptrs = e_weights.outerIndexPtr();
		int nonzeros = e_weights.nonZeros();
		for (int vi = 0; vi < siz; vi++) {
			if(!(is_control_vertex(vi))) {
				col=0;
				double diagonal_val = 0;
				double wij = 0;
				double total_weight = 0;
				int strt = row_ptrs[vi]; // start index for adjacency list
				int end = (nonzeros) - 1; // last index for adjacency list
				if ((vi + 1) < siz) {
					end = row_ptrs[(vi + 1)] - 1;
				}
				for (; strt <=end; strt++) {
					int vj = col_indices[strt];
					wij = weight_vals[strt];
					total_weight = wij + wij; // As wij == wji
					if(!(is_control_vertex(vj))) {
						col = vj - in_range(vj);
						if(row<=col) { // Only store upper triangular part of symmetric matrix
							((int *)cholmodTriplets->i)[cholmodTriplets->nnz] = row;
							((int *)cholmodTriplets->j)[cholmodTriplets->nnz] = col;
							((double *)cholmodTriplets->x)[cholmodTriplets->nnz] = -total_weight;
							cholmodTriplets->nnz+=1;
						}
					}
					else {
						control_columns[vj][row] = -total_weight;// new
					}
					diagonal_val += total_weight;
				}
				((int *)cholmodTriplets->i)[cholmodTriplets->nnz] = row;
				((int *)cholmodTriplets->j)[cholmodTriplets->nnz] = row;
				((double *)cholmodTriplets->x)[cholmodTriplets->nnz] = diagonal_val;
				cholmodTriplets->nnz+=1;
				vertex_index_map[vi] = row;
				row++;
			}
			else{
				vertex_index_map[vi] = -1;
			}
		}
		AA = cholmod_triplet_to_sparse(cholmodTriplets, new_size*10, c);
		cholmod_free_triplet(&cholmodTriplets, c);
		cholmodTriplets = NULL;
		L = cholmod_analyze(AA, c);
		int res = cholmod_factorize(AA, L, c);
		int a_siz = AA->nrow;
		unsigned int n = std::thread::hardware_concurrency();
		unsigned int subparts = a_siz / n;
		control_vals = (double *)calloc(3 * a_siz , sizeof(double));
		for(int i=0;i<control_vertices.size();i++) {
			std::vector<float> &vectr = control_columns[control_vertices[i]];
			std::vector<std::thread> threds;
			int index = 0;
			if (subparts <= 0) {
				threds.push_back(std::thread(&Arap_mesh_deformation::copy_control_sub, this, index, a_siz - 1, i, a_siz, std::ref(vectr)));
			}
			else {
				for (int k = 0; k < n; k += 1) {
					threds.push_back(std::thread(&Arap_mesh_deformation::copy_control_sub, this, index, index + subparts - 1, i, a_siz, std::ref(vectr)));
					index += subparts;
				}
				if (a_siz % n != 0) {
					threds.push_back(std::thread(&Arap_mesh_deformation::copy_control_sub, this, index, a_siz - 1, i, a_siz, std::ref(vectr)));
				}
			}
			for (auto& th : threds) {
				th.join();
			}

		}
		retrieve_factor(control_vals);
		cuda_set_index_map(vertex_index_map);
		preprocess_successful = true;
		free(control_vals);
	}

	// calculate local rotation using GPU
	void calculate_optimal_rotations() {
		int siz = vertices.rows();
		cuda_compute_rotations(siz, e_weights.nonZeros());
	}

	// calculate global positions using GPU
	void calculate_target_positions(bool apply_acceleration) {
		int siz = vertices.rows();
		int nonzeros = e_weights.nonZeros();
		cuda_cal_rhs(siz, nonzeros, control_vertices.size());
		cuda_solve(AA->nrow, factor_nzmax, apply_acceleration);
	}




    /*
	 *
	 *
	 * CUDA FUNCTIONS
	 *
	 *
	 *
	 * */


	void gpuAssert(cudaError_t code, std::string file, int line, bool abort = false)
	{
		if (code != cudaSuccess)
		{
			std::cout << "GPU Error: " << cudaGetErrorString(code) << " in File: " << file << " at Line: " << line << "\n";
		}
	}

	static const char *_cusparseGetErrorStr(cusparseStatus_t error)
	{
	    switch (error)
	    {

	        case CUSPARSE_STATUS_SUCCESS:
	            return "CUSPARSE_STATUS_SUCCESS";

	        case CUSPARSE_STATUS_NOT_INITIALIZED:
	            return "CUSPARSE_STATUS_NOT_INITIALIZED";

	        case CUSPARSE_STATUS_ALLOC_FAILED:
	            return "CUSPARSE_STATUS_ALLOC_FAILED";

	        case CUSPARSE_STATUS_INVALID_VALUE:
	            return "CUSPARSE_STATUS_INVALID_VALUE";

	        case CUSPARSE_STATUS_ARCH_MISMATCH:
	            return "CUSPARSE_STATUS_ARCH_MISMATCH";

	        case CUSPARSE_STATUS_MAPPING_ERROR:
	            return "CUSPARSE_STATUS_MAPPING_ERROR";

	        case CUSPARSE_STATUS_EXECUTION_FAILED:
	            return "CUSPARSE_STATUS_EXECUTION_FAILED";

	        case CUSPARSE_STATUS_INTERNAL_ERROR:
	            return "CUSPARSE_STATUS_INTERNAL_ERROR";

	        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
	            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

	        case CUSPARSE_STATUS_ZERO_PIVOT:
	            return "CUSPARSE_STATUS_ZERO_PIVOT";
	    }

	    return "<unknown>";
	}

	inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
	{
	    if(CUSPARSE_STATUS_SUCCESS != err) {
	    	std::cout << "CUSPARSE error in file " << file << " , line " << line << " Error: " << _cusparseGetErrorStr(err) << "\n";
	    }
	}

	void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }

	// update positions on GPU
	void cuda_update_vertex_positions(std::vector<double> &sol_x, std::vector<double> &sol_y, std::vector<double> &sol_z, size_t num_vertices) {

		gpuAssert(cudaMemcpy(dev_solution_x, sol_x.data(), num_vertices * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_solution_y, sol_y.data(), num_vertices * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_solution_z, sol_z.data(), num_vertices * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);

	}

	// free GPU arrays
	void cuda_free_vertices() {

		gpuAssert(cudaFree(dev_rot_matrices), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_energies), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_original_x), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_original_y), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_original_z), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_solution_x), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_solution_y), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_solution_z), __FILE__, __LINE__);

		gpuAssert(cudaFree(dev_control_vals), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_index_map), __FILE__, __LINE__);

		// weights CSR
		gpuAssert(cudaFree(dev_weight_values), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_column_indices), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_row_ptrs), __FILE__, __LINE__);

		//solver

		gpuAssert(cudaFree(dev_temp_B), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_perm), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_vals_lower), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_col_indices_lower), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_row_indices_lower), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_vals_upper), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_col_indices_upper), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_row_indices_upper), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_BZ), __FILE__, __LINE__);
		gpuAssert(cudaFree(d_z), __FILE__, __LINE__);
		gpuAssert(cudaFree(pBuffer), __FILE__, __LINE__);

		cusparseSafeCall(cusparseDestroy(handle));
		cusparseSafeCall(cusparseDestroyMatDescr(descr_L));
		cusparseSafeCall(cusparseDestroyMatDescr(descr_Lt));
		cusparseSafeCall(cusparseDestroyCsrsv2Info (info_L));
		cusparseSafeCall(cusparseDestroyCsrsv2Info (info_Lt));

	}

	// initialize original and solution arrays for GPU
	void cuda_init_vertices(std::vector<double> &orig_x, std::vector<double> &orig_y, std::vector<double> &orig_z, std::vector<double> &sol_x, std::vector<double> &sol_y, std::vector<double> &sol_z, size_t num_vertices) {

		gpuAssert(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2000 * 1024 * 1024), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_rot_matrices, 9 * num_vertices * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_energies, num_vertices * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_original_x, num_vertices * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_original_y, num_vertices * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_original_z, num_vertices * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_solution_x, num_vertices * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_solution_y, num_vertices * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_solution_z, num_vertices * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_index_map, num_vertices * sizeof(int)), __FILE__, __LINE__);

		gpuAssert(cudaMemcpy(dev_original_x, orig_x.data(), num_vertices * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_original_y, orig_y.data(), num_vertices * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_original_z, orig_z.data(), num_vertices * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_solution_x, sol_x.data(), num_vertices * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_solution_y, sol_y.data(), num_vertices * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_solution_z, sol_z.data(), num_vertices * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);

	}

	// initialize and copy edge weights matrix to GPU in CSR format
	void cuda_init_weights(double *weight_values, int *column_indices, int *row_ptrs, int rows, int nnz) {
		gpuAssert(cudaMalloc((void **)&dev_weight_values, nnz * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_column_indices, nnz * sizeof(int)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_row_ptrs, rows * sizeof(int)), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_weight_values, weight_values, nnz * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_column_indices, column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_row_ptrs, row_ptrs, rows * sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	}

	// set the map containing new indices of each vertex after excluding constraint vertices
	void cuda_set_index_map(std::vector<int> &index_map){
		gpuAssert(cudaMemcpy(dev_index_map, index_map.data(), (index_map.size()* sizeof(int)), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	}

	// call cuda kernel to calculate ARAP energy of whole mesh
	double get_energy(){
		double result = call_device_compute_energies(vertices.rows(), e_weights.nonZeros(), dev_row_ptrs, dev_column_indices, dev_weight_values, dev_original_x, dev_original_y, dev_original_z, dev_solution_x, dev_solution_y, dev_solution_z, dev_rot_matrices, dev_energies);
		return result;
	}

	// call cuda kernel to compute local rotation matrices for all vertices
	int cuda_compute_rotations(size_t num_vertices, size_t col_indices_size) {

		call_device_compute_rotations(num_vertices, col_indices_size, dev_row_ptrs, dev_column_indices, dev_weight_values, dev_original_x, dev_original_y, dev_original_z, dev_solution_x, dev_solution_y, dev_solution_z, dev_rot_matrices);
		cudaError_t error;
		if ((error = cudaGetLastError()) != cudaSuccess)
		{
			fprintf(stderr, "kernel execution error: %s\n", cudaGetErrorString(error));
		}
		return 0;

	}

	// analyze lower and upper triangular cholesky factors using cuSparse
	void cusparse_analyze(double *dev_vals_lower, int *dev_row_indices_lower, int *dev_col_indices_lower, double *dev_vals_upper, int *dev_row_indices_upper, int *dev_col_indices_upper, int rows, int nnz){

		if(handle==0){
			cusparseSafeCall(cusparseCreate(&handle));
			cusparseSafeCall(cusparseCreateMatDescr(&descr_L));
			cusparseSafeCall(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
			cusparseSafeCall(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
			cusparseSafeCall(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
			cusparseSafeCall(cusparseSetMatDiagType (descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT));

			cusparseSafeCall(cusparseCreateMatDescr(&descr_Lt));
			cusparseSafeCall(cusparseSetMatType(descr_Lt, CUSPARSE_MATRIX_TYPE_GENERAL));
			cusparseSafeCall(cusparseSetMatIndexBase(descr_Lt, CUSPARSE_INDEX_BASE_ZERO));
			cusparseSafeCall(cusparseSetMatFillMode(descr_Lt, CUSPARSE_FILL_MODE_UPPER));
			cusparseSafeCall(cusparseSetMatDiagType (descr_Lt, CUSPARSE_DIAG_TYPE_NON_UNIT));

			cusparseSafeCall(cusparseCreateCsrsv2Info (&info_L));
			cusparseSafeCall(cusparseCreateCsrsv2Info (&info_Lt));
		}

		cusparseSafeCall(cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnz, descr_L, dev_vals_lower, dev_row_indices_lower, dev_col_indices_lower, info_L, &pBufferSize_L));
		cusparseSafeCall(cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnz, descr_Lt, dev_vals_upper, dev_row_indices_upper, dev_col_indices_upper, info_Lt, &pBufferSize_Lt));

		int pBufferSize = std::max(pBufferSize_L, pBufferSize_Lt);

		gpuAssert(cudaMalloc((void**)&pBuffer, pBufferSize), __FILE__, __LINE__);
		cusparseSafeCall(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnz, descr_L, dev_vals_lower, dev_row_indices_lower, dev_col_indices_lower, info_L,  CUSPARSE_SOLVE_POLICY_USE_LEVEL,  pBuffer));
		cusparseSafeCall(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnz, descr_Lt, dev_vals_upper, dev_row_indices_upper, dev_col_indices_upper, info_Lt, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));

	}

	// solve triangular system
	void compute_cuSparse_solve(double *dev_vals_lower, int *dev_row_indices_lower, int *dev_col_indices_lower, double *dev_vals_upper, int *dev_row_indices_upper, int *dev_col_indices_upper, double *d_x, int rows, int nnz){
		const double alpha = 1.;
		cusparseSafeCall(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnz, &alpha, descr_L, dev_vals_lower, dev_row_indices_lower, dev_col_indices_lower, info_L, d_x, d_z, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
		cusparseSafeCall(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnz, &alpha, descr_Lt, dev_vals_upper, dev_row_indices_upper, dev_col_indices_upper, info_Lt, d_z, d_x, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
	}

	// set permutation vector
	void set_perm(int *perm, int rows){
		gpuAssert(cudaFree(dev_perm), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_temp_B), __FILE__, __LINE__);
		if(perm == NULL){
			dev_perm = NULL;
		}
		else{
			gpuAssert(cudaMalloc((void **)&dev_perm, rows * sizeof(int)), __FILE__, __LINE__);
			gpuAssert(cudaMalloc((void **)&dev_temp_B, 3 * rows * sizeof(double)), __FILE__, __LINE__);
			gpuAssert(cudaMemcpy(dev_perm, perm, rows * sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		}
	}

	// copy triangular cholesky factors to GPU and analyze them using cuSparse
	void set_factors(double *vals_lower, int *col_indices_lower, int *row_indices_lower, double *vals_upper, int *col_indices_upper, int *row_indices_upper, int *perm, double *control_vals, int nnz, int rows){

		gpuAssert(cudaFree(dev_vals_lower), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_col_indices_lower), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_row_indices_lower), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_vals_upper), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_col_indices_upper), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_row_indices_upper), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_control_vals), __FILE__, __LINE__);
		gpuAssert(cudaFree(dev_BZ), __FILE__, __LINE__);
		gpuAssert(cudaFree(d_z), __FILE__, __LINE__);

		gpuAssert(cudaMalloc((void **)&dev_vals_lower, nnz * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_col_indices_lower, nnz * sizeof(int)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_row_indices_lower, (rows+1) * sizeof(int)), __FILE__, __LINE__);

		gpuAssert(cudaMalloc((void **)&dev_vals_upper, nnz * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_col_indices_upper, nnz * sizeof(int)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_row_indices_upper, (rows+1) * sizeof(int)), __FILE__, __LINE__);

		gpuAssert(cudaMalloc((void **)&dev_control_vals, 3 * AA->nrow * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&dev_BZ, 3* rows * sizeof(double)), __FILE__, __LINE__);
		gpuAssert(cudaMalloc((void **)&d_z, rows * sizeof(double)), __FILE__, __LINE__);

		gpuAssert(cudaMemcpy(dev_vals_lower, vals_lower, nnz * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_col_indices_lower, col_indices_lower, nnz * sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_row_indices_lower, row_indices_lower, (rows+1) * sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);

		gpuAssert(cudaMemcpy(dev_vals_upper, vals_upper, nnz * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_col_indices_upper, col_indices_upper, nnz * sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaMemcpy(dev_row_indices_upper, row_indices_upper, (rows+1) * sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);

		gpuAssert(cudaMemcpy(dev_control_vals, control_vals, 3 * (AA->nrow) * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		gpuAssert(cudaFree(pBuffer), __FILE__, __LINE__);
		cusparse_analyze(dev_vals_lower, dev_row_indices_lower, dev_col_indices_lower, dev_vals_upper, dev_row_indices_upper, dev_col_indices_upper, rows, nnz);
	}

	// calculate right hand sides of the system to be solved
	void cuda_cal_rhs(int num_vertices, int col_indices_size, int control_size){

		if(dev_perm !=NULL){
			call_device_compute_rhs(num_vertices, col_indices_size, control_size, dev_row_ptrs, dev_column_indices, dev_weight_values, dev_original_x, dev_original_y, dev_original_z, dev_rot_matrices, dev_index_map, dev_temp_B, dev_control_vals, dev_perm);
			cudaError_t error;
			if ((error = cudaGetLastError()) != cudaSuccess)
			{
				fprintf(stderr, "kernel execution error for RHS calculation: %s\n", cudaGetErrorString(error));
			}
			call_dev_permute(dev_temp_B, dev_BZ, dev_perm, (num_vertices-control_size));
		}
		else{
			call_device_compute_rhs(num_vertices, col_indices_size, control_size, dev_row_ptrs, dev_column_indices, dev_weight_values, dev_original_x, dev_original_y, dev_original_z, dev_rot_matrices, dev_index_map, dev_BZ, dev_control_vals, dev_perm);
			cudaError_t error;
			if ((error = cudaGetLastError()) != cudaSuccess)
			{
				fprintf(stderr, "kernel execution error for RHS calculation: %s\n", cudaGetErrorString(error));
			}
		}

	}

	// solve triangular linear system using cuSparse
	void cuda_solve(int rows, int nnz, bool apply_accel){
		compute_cuSparse_solve(dev_vals_lower, dev_row_indices_lower, dev_col_indices_lower, dev_vals_upper, dev_row_indices_upper, dev_col_indices_upper, dev_BZ, rows, nnz);
		compute_cuSparse_solve(dev_vals_lower, dev_row_indices_lower, dev_col_indices_lower, dev_vals_upper, dev_row_indices_upper, dev_col_indices_upper, (dev_BZ+rows), rows, nnz);
		compute_cuSparse_solve(dev_vals_lower, dev_row_indices_lower, dev_col_indices_lower, dev_vals_upper, dev_row_indices_upper, dev_col_indices_upper, (dev_BZ+(rows*2)), rows, nnz);
		cudaDeviceSynchronize();
		if(dev_perm !=NULL){
			call_dev_inverse_permute(dev_BZ, dev_temp_B, dev_perm, rows);
			call_apply_acceleration(vertices.rows(), rows, dev_solution_x, dev_solution_y, dev_solution_z, dev_temp_B, dev_index_map, apply_accel);
		}
		else{
			call_apply_acceleration(vertices.rows(), rows, dev_solution_x, dev_solution_y, dev_solution_z, dev_BZ, dev_index_map, apply_accel);
		}

	}

};
