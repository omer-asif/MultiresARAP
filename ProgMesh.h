#include <memory>
#include <string>
#include <Eigen/Eigen>

#include "PMesh.h"
#include "Array.h"
#include "GMesh.h"
#include "FileIO.h"
#include "BinaryIO.h"
#include "MeshOp.h"


// This class is used to read a progressive mesh file and generate different LODs
class ProgMesh {
private:
	hh::PMesh pmesh;
	std::unique_ptr<hh::RFile> g_fi;
	std::unique_ptr<hh::PMeshRStream> pmrs;
	std::unique_ptr<hh::PMeshIter> pmi;
	float pm_lod_level;

public:
	// load a progressive mesh file with given name
	ProgMesh(const std::string& filename){
		pm_lod_level = 0.1f;
		g_fi = std::make_unique<hh::RFile>(filename);
		pmrs = std::make_unique<hh::PMeshRStream>((*g_fi)(), &pmesh);
		pmi = std::make_unique<hh::PMeshIter>(*pmrs);
		pmi->goto_nvertices(pmrs->base_mesh()._vertices.num()+int(pmrs->_info._tot_nvsplits*pm_lod_level+.5f));
	}

	// change the number of vertices in the mesh using the given level of detail from 0.1 to 1.0
	void update_lod() {
	    float flevel = std::min(pm_lod_level, 1.f);
	    int nv0 = pmesh._base_mesh._vertices.num();
	    int nvsplits = pmesh._info._tot_nvsplits;
	    int nv = nv0+int((nvsplits+1)*flevel*.999999f);
	    pmi->goto_nvertices(nv);
	}

	// change the level of detail of the progressive mesh object
	void set_lod(float lod) {
		pm_lod_level = lod;
	    if (!assertw(pm_lod_level>=0.f)) pm_lod_level = 0.f;
	    if (!assertw(pm_lod_level<=1.f)) pm_lod_level = 1.f;
	    update_lod();
	}

	// extract the vertices and triangles of progressive mesh at the current level of detail
	void extract_mesh(Eigen::MatrixXd &V, Eigen::MatrixXi &F){
		V.resize(pmi->_vertices.num(), 3);
		for_int(v, pmi->_vertices.num()){
			hh::Point p = pmi->_vertices[v].attrib.point;
			V.row(v) = Eigen::Vector3d(p[0], p[1], p[2]);
		}
		F.resize(pmi->_faces.num(), 3);
		for_int(f, pmi->_faces.num()){
			int w1 = pmi->_faces[f].wedges[0];
			int w2 = pmi->_faces[f].wedges[1];
			int w3 = pmi->_faces[f].wedges[2];

			int v1 = pmi->_wedges[w1].vertex;
			int v2 = pmi->_wedges[w2].vertex;
			int v3 = pmi->_wedges[w3].vertex;

			F.row(f) = Eigen::Vector3i(v1, v2, v3);
		}
	}

	// helper function to generate mesh file in .m format which is required by
	// Mesh Processing Library by Hughes Hoppe to be able to generate Progressive Mesh file.
	void make_mMesh(Eigen::MatrixXd &V, Eigen::MatrixXi &F, std::string filename){
		hh::GMesh gmesh;

		for(int j=0;j<V.rows();j++){
			float x = V(j, 0);
			float y = V(j, 1);
			float z = V(j, 2);
			hh::Vertex v = gmesh.create_vertex_private(j+1);
			gmesh.set_point(v, hh::Point(x, y, z));
		}

		for(int i=0;i<F.rows();i++){
			int vid0 = F(i, 0);
			int vid1 = F(i, 1);
			int vid2 = F(i, 2);
			hh::Vertex v0 = gmesh.id_vertex(vid0+1);
			hh::Vertex v1 = gmesh.id_vertex(vid1+1);
			hh::Vertex v2 = gmesh.id_vertex(vid2+1);
			gmesh.create_face(v0, v1, v2);
		}

		std::ofstream myfile;
		myfile.open(filename);
		gmesh.write(myfile);
		myfile.close();

	}

	~ProgMesh(){
		g_fi.reset();
		pmrs.reset();
		pmi.reset();
	}
};





