/*
 * MultiresDeformation.h
 *
 *  Created on: Dec 28, 2017
 *      Author: Omer Asif
 */

#ifndef MULTIRESDEFORMATION_H_
#define MULTIRESDEFORMATION_H_

#include "Arap_mesh_deformation.cu"
#include <ANN/ANN.h>
#include "ProgMesh.h"

typedef Arap_mesh_deformation mesh_deformation;


struct NeighborStruct{
	int vid=-1;
	float dist=0.0f;
	float w=0.0f;
};

// class to handle multires deformation using GPU ARAP
// LOD1 is the finest and LOD3 is the coarsest level of detail
class MultiresDeformation {

private:
	mesh_deformation *lod1_deform_mesh = NULL;
	mesh_deformation *lod2_deform_mesh = NULL;
	mesh_deformation *lod3_deform_mesh = NULL;

	Eigen::MatrixXd lod2_V, &lod1_V, lod3_V;
	Eigen::MatrixXi lod2_F, &lod1_F, lod3_F;

	int lod1_vid;
	int lod2_vid;
	int lod3_vid;

	bool lower_lods_done;

	std::vector<std::vector<struct NeighborStruct>> lod1_to_lod2;
	std::vector<std::vector<struct NeighborStruct>> lod2_to_lod1;
	std::vector<std::vector<struct NeighborStruct>> lod2_to_lod3;
	std::vector<std::vector<struct NeighborStruct>> lod3_to_lod2;

public:

	MultiresDeformation(std::string filename, Eigen::MatrixXd &V, Eigen::MatrixXi &F) : lod1_V(V), lod1_F(F) {
		lod1_vid = -1;
		lod2_vid = -1;
		lod3_vid = -1;

		ProgMesh pm(filename);
		pm.set_lod(0.04f);
		pm.extract_mesh(lod3_V, lod3_F);
		pm.set_lod(0.2f);
		pm.extract_mesh(lod2_V, lod2_F);
		pm.set_lod(1.0f);
		pm.extract_mesh(lod1_V, lod1_F);

		generate_mapping();
		setup_mesh_for_deformation();
		lower_lods_done = false;
	}

	void deform_lower_lods(){
		lod3_deform_mesh->deform(300, 1e-3);
		copy_positions_lod2();
		lod2_deform_mesh->deform(300, 1e-3);
		copy_positions_lod1();
		lower_lods_done = true;
	}

	void perform_lod_deformation(){
		if(!lower_lods_done)
			deform_lower_lods();
		lod1_deform_mesh->deform(10, 1e-4);
		lod1_deform_mesh->update_mesh();
	}

	bool setup_mesh_for_deformation() {
		// Create a deformation object
		if (lod1_deform_mesh != NULL) {
			delete lod1_deform_mesh;
			lod1_deform_mesh = NULL;
		}

		if (lod2_deform_mesh != NULL) {
			delete lod2_deform_mesh;
			lod2_deform_mesh = NULL;
		}

		if (lod3_deform_mesh != NULL) {
			delete lod3_deform_mesh;
			lod3_deform_mesh = NULL;
		}

		lod1_deform_mesh = new mesh_deformation(lod1_V, lod1_F);
		lod2_deform_mesh = new mesh_deformation(lod2_V, lod2_F);
		lod3_deform_mesh = new mesh_deformation(lod3_V, lod3_F);
		return true;
	}

	// calculate inverse squared distance weights
	void calculate_IDW(std::vector<struct NeighborStruct> &vec){
		float w_sum=0.0f, tmp_w=0.0f;
		const float MIN_DIST = 0.0001f;
		for(int i=0;i<vec.size();i++){
			if(vec[i].dist == 0.0f){
				vec[i].dist = MIN_DIST;
				tmp_w = 1.0f;
				w_sum=tmp_w;
				vec[i].w = tmp_w;
				break;
			}
			else{
				tmp_w = 1.0f/vec[i].dist;
				w_sum+=tmp_w;
				vec[i].w = tmp_w;
			}

		}
		for(int j=0;j<vec.size();j++){
			vec[j].w = vec[j].w/w_sum;
		}

	}

	// calculate nearest neighbors mapping for each pair of consecutive LOD
	void generate_mapping(){
		lod1_to_lod2.resize(lod1_V.rows());
		lod2_to_lod1.resize(lod2_V.rows());
		lod3_to_lod2.resize(lod3_V.rows());
		lod2_to_lod3.resize(lod2_V.rows());

		int					nPts;					// actual number of data points
		ANNpointArray		lod1_dataPts;				// data points
		ANNpointArray		lod2_dataPts;
		ANNpointArray		lod3_dataPts;
		ANNpoint			queryPt;				// query point
		ANNidxArray			nnIdx;					// near neighbor indices
		ANNdistArray		dists;					// near neighbor distances
		ANNkd_tree*			lod1_kdTree;
		ANNkd_tree*			lod2_kdTree;
		ANNkd_tree*			lod3_kdTree;

		queryPt = annAllocPt(3);					// allocate query point
		lod1_dataPts = annAllocPts(lod1_V.rows(), 3);			// allocate data points
		lod2_dataPts = annAllocPts(lod2_V.rows(), 3);			// allocate data points
		lod3_dataPts = annAllocPts(lod3_V.rows(), 3);
		nnIdx = new ANNidx[8];						// allocate near neigh indices
		dists = new ANNdist[8];						// allocate near neighbor dists


		for(int ii=0;ii<lod1_V.rows();ii++){
			lod1_dataPts[ii][0] = lod1_V(ii, 0);
			lod1_dataPts[ii][1] = lod1_V(ii, 1);
			lod1_dataPts[ii][2] = lod1_V(ii, 2);
		}

		lod1_kdTree = new ANNkd_tree(					// build search structure
							lod1_dataPts,					// the data points
							lod1_V.rows(),						// number of points
							3);						// dimension of space

		for(int ii=0;ii<lod2_V.rows();ii++){
			lod2_dataPts[ii][0] = lod2_V(ii, 0);
			lod2_dataPts[ii][1] = lod2_V(ii, 1);
			lod2_dataPts[ii][2] = lod2_V(ii, 2);
		}
		lod2_kdTree = new ANNkd_tree(					// build search structure
							lod2_dataPts,					// the data points
							lod2_V.rows(),						// number of points
							3);						// dimension of space

		for(int ii=0;ii<lod3_V.rows();ii++){
				lod3_dataPts[ii][0] = lod3_V(ii, 0);
				lod3_dataPts[ii][1] = lod3_V(ii, 1);
				lod3_dataPts[ii][2] = lod3_V(ii, 2);
			}
			lod3_kdTree = new ANNkd_tree(					// build search structure
								lod3_dataPts,					// the data points
								lod3_V.rows(),						// number of points
								3);


		for(int k=0;k<lod1_V.rows();k++){
			queryPt[0] = lod1_V(k, 0);
			queryPt[1] = lod1_V(k, 1);
			queryPt[2] = lod1_V(k, 2);
			lod2_kdTree->annkSearch(						// search
					queryPt,						// query point
					5,								// number of near neighbors
					nnIdx,							// nearest neighbors (returned)
					dists,							// distance (returned)
					0);

			std::vector<struct NeighborStruct> vv;
			for(int m=0;m<5;m++){
				struct NeighborStruct ns;
				ns.dist = dists[m];
				ns.vid = nnIdx[m];
				vv.push_back(ns);
			}
			calculate_IDW(vv);
			lod1_to_lod2[k] = vv;
		}
		for(int k=0;k<lod2_V.rows();k++){
			queryPt[0] = lod2_V(k, 0);
			queryPt[1] = lod2_V(k, 1);
			queryPt[2] = lod2_V(k, 2);
			lod1_kdTree->annkSearch(						// search
					queryPt,						// query point
					5,								// number of near neighbors
					nnIdx,							// nearest neighbors (returned)
					dists,							// distance (returned)
					0);
			std::vector<struct NeighborStruct> vv;
			for(int m=0;m<5;m++){
				struct NeighborStruct ns;
				ns.dist = dists[m];
				ns.vid = nnIdx[m];
				vv.push_back(ns);
			}
			calculate_IDW(vv);
			lod2_to_lod1[k] = vv;
		}
		for(int k=0;k<lod2_V.rows();k++){
			queryPt[0] = lod2_V(k, 0);
			queryPt[1] = lod2_V(k, 1);
			queryPt[2] = lod2_V(k, 2);
			lod3_kdTree->annkSearch(						// search
					queryPt,						// query point
					5,								// number of near neighbors
					nnIdx,							// nearest neighbors (returned)
					dists,							// distance (returned)
					0);
			std::vector<struct NeighborStruct> vv;
			for(int m=0;m<5;m++){
				struct NeighborStruct ns;
				ns.dist = dists[m];
				ns.vid = nnIdx[m];
				vv.push_back(ns);
			}
			calculate_IDW(vv);
			lod2_to_lod3[k] = vv;
		}
		for(int k=0;k<lod3_V.rows();k++){
			queryPt[0] = lod3_V(k, 0);
			queryPt[1] = lod3_V(k, 1);
			queryPt[2] = lod3_V(k, 2);
			lod2_kdTree->annkSearch(						// search
					queryPt,						// query point
					5,								// number of near neighbors
					nnIdx,							// nearest neighbors (returned)
					dists,							// distance (returned)
					0);
			std::vector<struct NeighborStruct> vv;
			for(int m=0;m<5;m++){
				struct NeighborStruct ns;
				ns.dist = dists[m];
				ns.vid = nnIdx[m];
				vv.push_back(ns);
			}
			calculate_IDW(vv);
			lod3_to_lod2[k] = vv;
		}

		delete lod1_kdTree;
		delete lod2_kdTree;
		delete lod3_kdTree;
		lod1_kdTree = NULL;
		lod2_kdTree = NULL;
		lod3_kdTree = NULL;

	}

	void copy_lod2_sub(int strt, int end){
		for(int i=strt;i<=end;i++){
			if(!(lod2_deform_mesh->is_ctrl_map[i])){
				float x=0.0f,y=0.0f,z=0.0f;
				for(int j=0;j<lod2_to_lod3[i].size();j++){
					if((lod2_to_lod3[i])[j].w == 1.0f){
						x = lod3_deform_mesh->solution_x[(lod2_to_lod3[i])[j].vid];
						y = lod3_deform_mesh->solution_y[(lod2_to_lod3[i])[j].vid];
						z = lod3_deform_mesh->solution_z[(lod2_to_lod3[i])[j].vid];
						break;
					}
					else{
						x += (lod2_to_lod3[i])[j].w * lod3_deform_mesh->solution_x[(lod2_to_lod3[i])[j].vid];
						y += (lod2_to_lod3[i])[j].w * lod3_deform_mesh->solution_y[(lod2_to_lod3[i])[j].vid];
						z += (lod2_to_lod3[i])[j].w * lod3_deform_mesh->solution_z[(lod2_to_lod3[i])[j].vid];
					}

				}
				lod2_deform_mesh->solution_x[i] = x;
				lod2_deform_mesh->solution_y[i] = y;
				lod2_deform_mesh->solution_z[i] = z;
			}
		}

	}

	// copy positions from LOD3 to LOD2
	void copy_positions_lod2(){
		std::vector<std::thread> threds;
		unsigned int n = std::thread::hardware_concurrency();
		int siz = lod2_V.rows();
		unsigned int subparts = siz / n;
		int index = 0;
		if (subparts <= 0) {
			threds.push_back(std::thread(&MultiresDeformation::copy_lod2_sub, this, index, siz - 1));
		}
		else {
			for (int k = 0; k < n; k += 1) {
				threds.push_back(std::thread(&MultiresDeformation::copy_lod2_sub, this, index, index + subparts - 1));
				index += subparts;
			}
			if (siz % n != 0) {
				threds.push_back(std::thread(&MultiresDeformation::copy_lod2_sub, this, index, siz - 1));
			}
		}
		for (auto& th : threds) {
			th.join();
		}

		lod2_deform_mesh->update_device_vertices();
	}

	void copy_lod1_sub(int strt, int end){
		for(int i=strt;i<=end;i++){
			if(!(lod1_deform_mesh->is_ctrl_map[i])){
				float x=0.0f,y=0.0f,z=0.0f;
				for(int j=0;j<lod1_to_lod2[i].size();j++){
					if((lod1_to_lod2[i])[j].w == 1.0f){
						x = lod2_deform_mesh->solution_x[(lod1_to_lod2[i])[j].vid];
						y = lod2_deform_mesh->solution_y[(lod1_to_lod2[i])[j].vid];
						z = lod2_deform_mesh->solution_z[(lod1_to_lod2[i])[j].vid];
						break;
					}
					else{
						x += (lod1_to_lod2[i])[j].w * lod2_deform_mesh->solution_x[(lod1_to_lod2[i])[j].vid];
						y += (lod1_to_lod2[i])[j].w * lod2_deform_mesh->solution_y[(lod1_to_lod2[i])[j].vid];
						z += (lod1_to_lod2[i])[j].w * lod2_deform_mesh->solution_z[(lod1_to_lod2[i])[j].vid];
					}

				}
				lod1_deform_mesh->solution_x[i] = x;
				lod1_deform_mesh->solution_y[i] = y;
				lod1_deform_mesh->solution_z[i] = z;
			}
		}

	}

	// copy positions from LOD2 to LOD1
	void copy_positions_lod1(){
		std::vector<std::thread> threds;
		unsigned int n = std::thread::hardware_concurrency();
		int siz = lod1_V.rows();
		unsigned int subparts = siz / n;
		int index = 0;
		if (subparts <= 0) {
			threds.push_back(std::thread(&MultiresDeformation::copy_lod1_sub, this, index, siz - 1));
		}
		else {
			for (int k = 0; k < n; k += 1) {
				threds.push_back(std::thread(&MultiresDeformation::copy_lod1_sub, this, index, index + subparts - 1));
				index += subparts;
			}
			if (siz % n != 0) {
				threds.push_back(std::thread(&MultiresDeformation::copy_lod1_sub, this, index, siz - 1));
			}
		}
		for (auto& th : threds) {
			th.join();
		}
		lod1_deform_mesh->update_device_vertices();
	}

	void preprocess_sub(mesh_deformation *def_mesh){
		bool is_matrix_factorization_OK = def_mesh->preprocess();
		if (!is_matrix_factorization_OK) {
			std::cerr << "Error in preprocessing" << std::endl;
			return ;
		}
	}

	// add constraint vertex
	int set_constraint(int vid, Eigen::Vector3d& pos){

		lower_lods_done = false;
		lod2_vid = (lod1_to_lod2[vid])[0].vid;
		lod3_vid = lod2_to_lod3[lod2_vid][0].vid;
		lod2_vid = lod3_to_lod2[lod3_vid][0].vid;
		lod1_vid = (lod2_to_lod1[lod2_vid])[0].vid;

		lod1_deform_mesh->insert_control_vertex(lod1_vid);
		lod2_deform_mesh->insert_control_vertex(lod2_vid);
		lod3_deform_mesh->insert_control_vertex(lod3_vid);

		lod1_deform_mesh->set_target_position(lod1_vid, pos);
		lod2_deform_mesh->set_target_position(lod2_vid, pos);
		lod3_deform_mesh->set_target_position(lod3_vid, pos);

		std::thread t1(&MultiresDeformation::preprocess_sub, this, lod1_deform_mesh);
		std::thread t2(&MultiresDeformation::preprocess_sub, this, lod2_deform_mesh);
		std::thread t3(&MultiresDeformation::preprocess_sub, this, lod3_deform_mesh);

		t1.join();
		t2.join();
		t3.join();
		return lod1_vid;
	}

	void cleanup(){
		lod3_deform_mesh->perform_cleanup();
		lod2_deform_mesh->perform_cleanup();
		lod1_deform_mesh->perform_cleanup();
	}

	~MultiresDeformation(){
		delete lod1_deform_mesh;
		delete lod2_deform_mesh;
		delete lod3_deform_mesh;

		lod3_deform_mesh = NULL;
		lod2_deform_mesh = NULL;
		lod1_deform_mesh = NULL;
	}


};


#endif /* MULTIRESDEFORMATION_H_ */
