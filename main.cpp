#include <iostream>
#include <future>
#include <igl/readOFF.h>
#include <igl/unproject.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/viewer/Viewer.h>

#include "MultiresDeformation.h"


MultiresDeformation *multiDeform = NULL;
Eigen::MatrixXd VFull, CFull;
Eigen::MatrixXi FFull;
igl::viewer::Viewer viewer;
int last_vid = -1;
bool flag = false;
bool first_time = true;
std::future<void> future;
std::mutex mtex;

void perform_deformation() {
	bool stop_loop = false;
	mtex.lock();
	flag = true;
	mtex.unlock();
	while (!stop_loop)
	{
		multiDeform->perform_lod_deformation();
		viewer.data.set_vertices(VFull);
		if (flag == false)
			stop_loop = true;
	}
}

void stop_thread() {
	mtex.lock();
	flag = false;
	mtex.unlock();
	last_vid = -1;
	viewer.data.set_vertices(VFull);
}


int main(int argc, char *argv[]) {

	if(argc != 2){
		std::cout<<"usage: "<< argv[0] <<" <filename>\n";
		return 0;
	}
	else{
		float original_z = 0.0f;
		bool clicked_outside_mesh = true;
		multiDeform = new MultiresDeformation(argv[1], VFull, FFull);

		// handle mouse up event and do the deformation if mouse down was inside the mesh
		viewer.callback_mouse_up =
		  [&](igl::viewer::Viewer& viewer, int btn, int)->bool
		{
			if (last_vid != -1 && !clicked_outside_mesh) {
				GLint x = viewer.current_mouse_x;
				GLfloat y = viewer.core.viewport(3) - viewer.current_mouse_y;
				Eigen::Vector3f win_d(x, y, original_z);
				Eigen::Vector3f ss;
				Eigen::Matrix4f mvv = viewer.core.view * viewer.core.model;
				ss = igl::unproject(win_d, mvv, viewer.core.proj, viewer.core.viewport);
				Eigen::Vector3d pos = Eigen::Vector3d(ss.x(), ss.y(), ss.z());
				if (!first_time) {
					future.get();
				}

				last_vid = multiDeform->set_constraint(last_vid, pos);

				future = std::async(std::launch::async, &perform_deformation);
				first_time = false;
				return true;
			}
			return false;
		};

		// mouse click down event handler to catch the mouse click on the mesh.
		viewer.callback_mouse_down =
		[&](igl::viewer::Viewer& viewer, int btn, int)->bool
		{
			int face_id;
			Eigen::Vector3f bc;
			double x = viewer.current_mouse_x;
			double y = viewer.core.viewport(3) - viewer.current_mouse_y;
			GLfloat zz;
			glReadPixels(int(x), int(y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &zz); // read the current z value, as mouse only returns x,y according to screen
			original_z = zz;
			if(igl::unproject_onto_mesh( // check if user clicked inside the mesh
			Eigen::Vector2f(x,y),
			viewer.core.view * viewer.core.model,
			viewer.core.proj,
			viewer.core.viewport,
			VFull,
			FFull,
			face_id,
			bc))
			{
				stop_thread();
				clicked_outside_mesh = false;
				int i;
				bc.maxCoeff(&i);
				last_vid = FFull(face_id, i); // retrieve the vertex id clicked, by using the retrieved face_id
				CFull.row(last_vid) = Eigen::RowVector3d(1, 0, 0);
				viewer.data.set_colors(CFull);
				return true;
			}
			else
			{
				clicked_outside_mesh = true;
			}
			return false;
		};
		// Show mesh
		viewer.data.set_mesh(VFull, FFull);
		CFull = Eigen::MatrixXd::Constant(VFull.rows(),3,1);// set mesh color to white
		viewer.data.set_colors(CFull);
		//viewer.core.show_lines = true;
		viewer.launch();
		delete multiDeform;
		multiDeform = NULL;
	}



}
