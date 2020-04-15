// Standard includes
#include <vector>
#include <cassert>
#include <stdio.h>
#include <string.h>
#include<io.h>
#include <iostream>
#include <fstream>
#include <winsock2.h>
//#include  "Ws2tcpip.h"
#include <sstream>
#include <vector>

// PCL includes
// Undef on Win32 min/max for PCL
#ifdef _WIN32
#undef max
#undef min
#endif
/*
#include <pcl/common/common_headers.h>
#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
*/

// OpenCV includes
#include "Object.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer2.hpp"
#include "utils.hpp"

// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>

// OpenPose dependencies
#include <openpose/headers.hpp>

//float thres_score = 0.6;

// Debugging

DEFINE_int32(logging_level, 3, "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
	" 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
	" low priority messages and 4 for important ones.");
// OpenPose
DEFINE_string(model_pose, "COCO", "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
	"`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder, "models/", "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution, /*"320x240"*/"656x368", "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
	" decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
	" closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
	" any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
	" input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
	" e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_string(output_resolution, "-1x-1", "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
	" input image resolution.");
DEFINE_int32(num_gpu_start, 0, "GPU device start number.");
DEFINE_double(scale_gap, 0.3, "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
	" If you want to change the initial scale, you actually want to multiply the"
	" `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number, 1, "Number of scales to average.");
DEFINE_int32(number_people_max, -1, "This parameter will limit the maximum number of people detected, by keeping the people with"
	" top scores. The score is based in person area over the image, body part score, as well as"
	" joint score (between each pair of connected body parts). Useful if you know the exact"
	" number of people in the scene, so it can remove false positives (if all the people have"
	" been detected. However, it might also include false negatives by removing very small or"
	" highly occluded people. -1 will keep them all.");
// OpenPose Rendering
DEFINE_bool(disable_blending, false, "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
	" background, instead of being rendered into the original image. Related: `part_to_show`,"
	" `alpha_pose`, and `alpha_pose`.");
DEFINE_double(render_threshold, 0.2, "Only estimated keypoints whose score confidences are higher than this threshold will be"
	" rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
	" while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
	" more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
	" hide it. Only valid for GPU rendering.");
DEFINE_string(svo_path, "", "SVO filepath");
DEFINE_bool(ogl_ptcloud, true, "Display the point cloud in the OpenGL window");
DEFINE_bool(estimate_floor_plane, false, "Initialize the camera position from the floor plan detected in the scene");
DEFINE_bool(opencv_display, true, "Enable the 2D view of openpose output");
DEFINE_bool(depth_display, false, "Enable the depth display with openCV");


// Using std namespace
using namespace std;
using namespace sl;
using namespace cv;

// Create ZED objects
sl::Camera zed;
sl::Pose camera_pose, zed_pose;
sl::Mat point_cloud_click;
std::thread zed_callback, openpose_callback;
std::mutex data_in_mtx, data_out_mtx;
std::vector<op::Array<float>> netInputArray;
op::Array<float> poseKeypoints;
op::Point<int> imageSize, outputSize, netInputSize, netOutputSize;
op::PoseModel poseModel;
std::vector<double> scaleInputToNetInputs;
PointObject cloud;
PeoplesObject peopleObj;

bool quit = false;		// only true if close() called
string globalObj = "!";
static const float arr[] = { 0.0,0,0,0 };
vector <vector<float>> globalObjpy(arr, arr + sizeof(arr) / sizeof(arr[0]));

// OpenGL window to display camera motion
GLViewer viewer;




const int MAX_CHAR = 128;
const sl::UNIT unit = sl::UNIT_METER;			// meters, centimeters, inch or foot		
const sl::COORDINATE_SYSTEM coord_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP;	//COORDINATE_SYSTEM_IMAGE - used in OpenCV
																					// COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP - Right Handed Y up and Z backward - used in OpenGL.
																					// COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP - Right Handed Z up and Y forward - used in 3DSMax.
const sl::DEPTH_MODE depth_mode = sl::DEPTH_MODE_QUALITY;		// DEPTH_MODE_NONE - does not compute any depth map
															// DEPTH_MODE_PERFORMANCE - mode optimized for speed.
															// DEPTH_MODE_MEDIUM - Balanced quality mode
															// DEPTH_MODE_QUALITY - mode for high quality results
															// DEPTH_MODE_ULTRA - edges and sharpness, more GPU power
const sl::RESOLUTION res = sl::RESOLUTION_HD720;		// RESOLUTION_HD2K (2208 * 1242 at 15fps)
														// RESOLUTION_HD1080 (1920 * 1080 at 15 or 30 fps) -- if click objects
														// RESOLUTION_HD720 (1280 * 720 at 15, 30, 60 fps)
const float MAX_DISTANCE_LIMB = 1; //0.8;
const float MAX_DISTANCE_CENTER = 1.8; //1.5;

									   ////////////////////////////////////////
									   //detect object const
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 10;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;
//names that will appear at the top of each window
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
//////////////////////////////////////////


// Sample functions
void startZED();
void startOpenpose();
void onMouse(int event, int i, int j, int flags, void*);
void run();
void close();
void findpose();
//shared_ptr<pcl::visualization::PCLVisualizer> createRGBVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud2);
inline float convertColor(float colorIn);

int image_width = 1280;
int image_height = 720;

bool need_new_image = true;
bool ready_to_start = false;

#define ENABLE_FLOOR_PLANE_DETECTION 1 // Might be disable to use older ZED SDK

// Debug options
#define DISPLAY_BODY_BARYCENTER 0
#define PATCH_AROUND_KEYPOINT 1 

template<typename T> vector<float> flatten(const vector<T>& vec);
template<typename T> vector<T> flatten(const T& value);

template<typename T>
// Flatten Vector of vectors into 1D
vector<float> flatten(const vector<T>& vec) {

	assert(!vec.empty());

	vector<float> flatVec;
	vector<float> tempVec;

	for (unsigned int i = 0; i < vec.size(); ++i) {
		tempVec = flatten(vec[i]);
		flatVec.insert(flatVec.end(), tempVec.begin(), tempVec.end());
	}

	return flatVec;
}

template<typename T>
vector<T> flatten(const T& value) {

	return vector<T>(1, value);
}

string intToString(int number) {

	std::stringstream ss;
	ss << number;
	return ss.str();
}

void morphOps(cv::Mat &thresh) {

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle
	cv::Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	cv::Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);

	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);
}

void drawObject(vector<Object> theObjects, cv::Mat &frame, cv::Mat &temp, vector< vector<Point> > contours, vector<Vec4i> hierarchy) {

	for (int i = 0; i<theObjects.size(); i++) {
		cv::drawContours(frame, contours, i, theObjects.at(i).getColor(), 3, 8, hierarchy);
		cv::circle(frame, cv::Point(theObjects.at(i).getXPos(), theObjects.at(i).getYPos()), 5, theObjects.at(i).getColor());
		cv::putText(frame, intToString(theObjects.at(i).getXPos()) + " , " + intToString(theObjects.at(i).getYPos()), cv::Point(theObjects.at(i).getXPos(), theObjects.at(i).getYPos() + 20), 1, 1, theObjects.at(i).getColor());
		cv::putText(frame, theObjects.at(i).getType(), cv::Point(theObjects.at(i).getXPos(), theObjects.at(i).getYPos() - 20), 1, 2, theObjects.at(i).getColor());
	}
}

void trackFilteredObject(Object theObject, cv::Mat threshold, cv::Mat HSV, cv::Mat &cameraFeed, sl::Mat depth_map) {

	vector <Object> objects;
	std::vector<vector <float>> coord_obj;
	cv::Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//use moments method to find filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects<MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area <=20 px by 20px just noise
				//if the area >= 3/2 of the image size, probably bad filter
				if (area>MIN_OBJECT_AREA) {

					Object object;

					object.setXPos(moment.m10 / area);
					object.setYPos(moment.m01 / area);
					object.setType(theObject.getType());
					object.setColor(theObject.getColor());

					objects.push_back(object);

					int x_object = object.getXPos();
					int y_object = object.getYPos();
					sl::float4 obj_c;
					float obj_w = 0;
					// Get the 3D point cloud values for pixel (i,j)
					depth_map.getValue(x_object, y_object, &obj_c);
					//cout << x_object <<","<< y_object << endl;

					if (isfinite(obj_c.z))		// if camera can get valid coordinates of object save in coord_obj vector
					{
						coord_obj.push_back({ obj_c.x, obj_c.y,  obj_c.z, obj_w});
						//cout << obj_c.x << "," << obj_c.y << "," << obj_c.z << endl;
						string cor = object.getType();
						vector<float> vec = flatten(coord_obj);
						globalObjpy = coord_obj;
						std::stringstream ss;
						for (size_t i = 0; i < vec.size(); ++i)
						{
							if (i != 0)
								ss << ",";
							ss << vec[i];
						}
						std::string obj_inf = ss.str();
						obj_inf += ",";
						//obj_inf += cor;
						globalObj = obj_inf;
						//cout << "this is object" << obj_inf << endl;
						//int psize3 = sizeof(obj_inf[0])*obj_inf.size();
						// Send frame to python
						//send(sockL, (const char*)obj_inf.data(), psize3, 0);
					}

					objectFound = true;

				}
				else objectFound = false;
			}
			//let user know you found an object
			if (objectFound == true) {
				//draw object location on screen
				drawObject(objects, cameraFeed, temp, contours, hierarchy);
			}

		}
		else putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
	}
}

bool initFloorZED(sl::Camera &zed) {
	bool init = false;
#if ENABLE_FLOOR_PLANE_DETECTION
	sl::Plane plane;
	sl::Transform resetTrackingFloorFrame;
	const int timeout = 20;
	int count = 0;

	std::cout << "Looking for the floor plane to initialize the tracking..." << endl;

	while (!init && count++ < timeout) {
		zed.grab();
		init = (zed.findFloorPlane(plane, resetTrackingFloorFrame) == sl::ERROR_CODE::SUCCESS);
		resetTrackingFloorFrame.getInfos();
		if (init) {
			zed.getPosition(camera_pose, sl::REFERENCE_FRAME_WORLD);
			std::cout << "Floor found at : " << plane.getClosestDistance(camera_pose.pose_data.getTranslation()) << " m" << endl;
			zed.resetTracking(resetTrackingFloorFrame);
		}
		sl::sleep_ms(20);
	}
	if (init) for (int i = 0; i < 4; i++) zed.grab();
	else std::cout << "Floor plane not found, starting anyway" << endl;
#endif
	return init;
}

int main(int argc, char **argv) {

	gflags::ParseCommandLineFlags(&argc, &argv, true);	 // Parsing command line flags

														 // Set configuration parameters for the ZED
	InitParameters initParameters;
	initParameters.camera_resolution = res;
	initParameters.camera_fps = 15;
	initParameters.depth_mode = depth_mode;
	initParameters.coordinate_units = unit;
	initParameters.coordinate_system = coord_system;
	initParameters.sdk_verbose = 0;		// get runtime information in the console if 1
	initParameters.depth_stabilization = true;
	initParameters.svo_real_time_mode = 0;

	// if using svo file (not live)
	if (std::string(FLAGS_svo_path).find(".svo")) {
		std::cout << "Opening " << FLAGS_svo_path << endl;
		initParameters.svo_input_filename.set(std::string(FLAGS_svo_path).c_str());
	}

	// Open the camera
	ERROR_CODE err = zed.open(initParameters);
	if (err != sl::SUCCESS) {
		std::cout << err << std::endl;
		zed.close();
		return 1; // Quit if an error occurred
	}

	if (FLAGS_estimate_floor_plane)
		initFloorZED(zed);

	// Initialize OpenGL viewer
	viewer.init();

	// init OpenPose
	std::cout << "OpenPose : loading models..." << endl;
	// ------------------------- INITIALIZATION -------------------------
	// Applying user defined configuration - GFlags to program variables
	outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
	netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");	// the resolution of the first layer of the deep net. I.e., the frames given to this class must have that size.

																		//std::cout << netInputSize.x << "x" << netInputSize.y << endl;
	netOutputSize = netInputSize;	// netOutputSize - resolution of the last layer, I.e., the resulting heatmaps will have this size.
									// must be set to the same size as netInputSize.
	poseModel = op::flagsToPoseModel(FLAGS_model_pose);
	// Check no contradictory flags enabled
	if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.) op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
	if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1) op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.", __LINE__, __FUNCTION__, __FILE__);

	// Start ZED callback
	startZED();
	startOpenpose();

	// Set the display callback
	glutCloseFunc(close);
	glutMainLoop();

	return 0;
}

//Launch ZED thread. Using a thread here allows to capture a point cloud and update the GL window concurrently.
void startZED() {
	quit = false;

	zed_callback = std::thread(run);
}

void startOpenpose() {
	openpose_callback = std::thread(findpose);
}

void findpose() {

	while (!ready_to_start) sl::sleep_ms(2); // Waiting for the ZED

	op::PoseExtractorCaffe poseExtractorCaffe{ poseModel, FLAGS_model_folder, FLAGS_num_gpu_start,{}, op::ScaleMode::ZeroToOne, 1 };
	//op::HandExtractorCaffe handExtractorCaffe{};
	poseExtractorCaffe.initializationOnThread(); // Initialize resources on desired thread

	while (!quit) {		// quit = true if close() called
		INIT_TIMER;
		//  Estimate poseKeypoints
		if (!need_new_image) { // No new image
			data_in_mtx.lock();
			need_new_image = true;
			poseExtractorCaffe.forwardPass(netInputArray, imageSize, scaleInputToNetInputs);	//run the deep net over the desired target image
			data_in_mtx.unlock();

			// Extract poseKeypoints
			data_out_mtx.lock();
			poseKeypoints = poseExtractorCaffe.getPoseKeypoints();

			data_out_mtx.unlock();
			//STOP_TIMER("OpenPose");
		}
		else sl::sleep_ms(1);
	}
}

// The 3D of the point is not directly taken 'as is'. If the measurement isn't valid, we look around the point in 2D to find a close point with a valid depth
sl::float4 getPatchIdx(const int &center_i, const int &center_j, sl::Mat &xyzrgba) {
	sl::float4 out(NAN, NAN, NAN, NAN);
	bool valid_measure;
	int i, j;

	const int R_max = 10;

	for (int R = 0; R < R_max; R++) {
		for (int y = -R; y <= R; y++) {
			for (int x = -R; x <= R; x++) {
				i = center_i + x;
				j = center_j + y;
				xyzrgba.getValue<sl::float4>(i, j, &out, sl::MEM_CPU);
				valid_measure = isfinite(out.z);
				if (valid_measure) return out;
			}
		}
	}

	out = sl::float4(NAN, NAN, NAN, NAN);
	return out;
}

std::vector<vector <float>> CylTest_arms_legs(std::map<int, sl::float4> keypoints_pos, float radius_sq, float radius_tronco, sl::Mat testpt, int joint_test[])
{
	float dx, dy, dz;	// vector d  from line segment point 1 to point 2
	float pdx, pdy, pdz;	// vector pd from point 1 to test point
	float dot, dsq;
	float len;
	std::vector<sl::float3> pots;
	std::vector<vector <float>> points_body;
	int total = testpt.getResolution().area();		// returns number of total pixels
	float factor = 1;
	pots.resize(total / factor);

	sl::float4* po_mat = testpt.getPtr<sl::float4>(sl::MEM_CPU);		// pointer to xyzrgba matrix: each pixel contains [x,y,z,color]
	sl::float3* pos_f3;
	sl::float4* pos_f4;

	for (int jn = 0; jn < 5; jn += 1)
	{
		if (isfinite(keypoints_pos[joint_test[jn]].z))
		{

			len = sqrt(pow(keypoints_pos[joint_test[jn]].x - keypoints_pos[joint_test[jn + 1]].x, 2.0) + pow(keypoints_pos[joint_test[jn]].y - keypoints_pos[joint_test[jn + 1]].y, 2.0) + pow(keypoints_pos[joint_test[jn]].z - keypoints_pos[joint_test[jn + 1]].z, 2.0));		// length of cylinder

			dx = keypoints_pos[joint_test[jn + 1]].x - keypoints_pos[joint_test[jn]].x;	// translate so pt1 is origin.  Make vector from
			dy = keypoints_pos[joint_test[jn + 1]].y - keypoints_pos[joint_test[jn]].y;     // pt1 to pt2.  Need for this is easily eliminated
			dz = keypoints_pos[joint_test[jn + 1]].z - keypoints_pos[joint_test[jn]].z;


			int j = 0;
			for (int i = 0; i < total; i += factor, j++) {
				pos_f4 = &po_mat[i];		//returns p_mat[i] address
				pos_f3 = &pots[j];
				pos_f3->x = pos_f4->x;
				pos_f3->y = pos_f4->y;
				pos_f3->z = pos_f4->z;

				if (!isfinite(pots[i][2]))
				{
					continue;
				}
				else
				{
					pdx = pots[i][0] - keypoints_pos[joint_test[jn]].x;		// vector from pt1 to test point.
					pdy = pots[i][1] - keypoints_pos[joint_test[jn]].y;
					pdz = pots[i][2] - keypoints_pos[joint_test[jn]].z;

					// Dot the d and pd vectors to see if point lies behind the 
					// cylinder cap at pt1.x, pt1.y, pt1.z

					dot = pdx * dx + pdy * dy + pdz * dz;

					if (dot < 0.0f || dot > len)
						// If dot is less than zero the point is behind the pt1 cap.
						// If greater than the cylinder axis line segment length squared
						// then the point is outside the other end cap at pt2.
					{
						continue;
					}
					else
					{
						// Point lies within the parallel caps, so find
						// distance squared from point to line, using the fact that sin^2 + cos^2 = 1
						// the dot = cos() * |d||pd|, and cross*cross = sin^2 * |d|^2 * |pd|^2
						// Carefull: '*' means mult for scalars and dotproduct for vectors
						// In short, where dist is pt distance to cyl axis: 
						// dist = sin( pd to d ) * |pd|
						// distsq = dsq = (1 - cos^2( pd to d)) * |pd|^2
						// dsq = ( 1 - (pd * d)^2 / (|pd|^2 * |d|^2) ) * |pd|^2
						// dsq = pd * pd - dot * dot / lengthsq
						//  where lengthsq is d*d or |d|^2 that is passed into this function 

						// distance squared to the cylinder axis:

						dsq = (pdx*pdx + pdy*pdy + pdz*pdz) - dot*dot / len;

						if (dsq <= radius_sq)
						{
							points_body.push_back({pots[i].x, pots[i].y,  pots[i].z, 0 });

						}
					}
				}
			}

		}

		else
		{
			continue;
		}
	}

	if (isfinite(keypoints_pos[2].z) || isfinite(keypoints_pos[5].z))
	{

		len = sqrt(pow(keypoints_pos[2].x - keypoints_pos[5].x, 2.0) + pow(keypoints_pos[2].y - keypoints_pos[5].y, 2.0) + pow(keypoints_pos[2].z - keypoints_pos[5].z, 2.0));		// length of cylinder

		dx = keypoints_pos[5].x - keypoints_pos[2].x;	
		dy = keypoints_pos[5].y - keypoints_pos[2].y;     
		dz = keypoints_pos[5].z - keypoints_pos[2].z;

		int jj = 0;
		for (int ii = 0; ii < total; ii += factor, jj++) {

			if (!isfinite(pots[ii][2]))
			{
				continue;
			}
			else
			{
				pdx = pots[ii][0] - keypoints_pos[2].x;		// vector from pt1 to test point.
				pdy = pots[ii][1] - keypoints_pos[2].y;
				pdz = pots[ii][2] - keypoints_pos[2].z;

				// Dot the d and pd vectors to see if point lies behind the 
				// cylinder cap at pt1.x, pt1.y, pt1.z

				dot = pdx * dx + pdy * dy + pdz * dz;

				if (dot < 0.0f || dot > len)
					{continue;}
				else
					{
						dsq = (pdx*pdx + pdy*pdy + pdz*pdz) - dot*dot / len;

						if (dsq <= radius_tronco)
						{
							points_body.push_back({pots[ii].x, pots[ii].y,  pots[ii].z, 0 });
						}
					}
			}
		}

	}

	return points_body;
}


void fill_people_ogl(op::Array<float> &poseKeypoints, sl::Mat &xyz, SOCKET server_pose) {
	// Common parameters needed
	// poseKeyPoints - array float with estimated pose
	const auto numberPeopleDetected = poseKeypoints.getSize(0);
	const auto numberBodyParts = poseKeypoints.getSize(1);
	sl::float4 center_gravity(0, 0, 0, 0);

	std::vector<int> partsLink = {	// link each joint number (according to Pose Output Format (COCO) in openpose/doc/output.md
									//0, 1,
		2, 1,
		1, 5,
		8, 11,
		1, 8,
		11, 1,
		8, 9,
		9, 10,
		11, 12,
		12, 13,
		2, 3,
		3, 4,
		5, 6,
		6, 7,
		//0, 15,
		//15, 17,
		//0, 14,
		//14, 16,
		16, 1,
		17, 1,
		16, 17
	};


	sl::float4 v1, v2;
	int i, j;

	std::vector<sl::float3> vertices;
	std::vector<sl::float3> colr;


	for (int person = 0; person < numberPeopleDetected; person++) {
		std::vector<vector <float>> poses;
		std::map<int, sl::float4> keypoints_position; // (x,y,z) + score for each keypoints(w) - 4Dimensions
		sl::float4 center_gravity(0, 0, 0, 0);
		int count = 0;		// number of joints detected
		float score;		// confidence score
		std::vector<vector <float>> ans;

		for (int body_part = 0; body_part < numberBodyParts; body_part++) {
			score = poseKeypoints[{person, body_part, 2}];		//confidence score in the range [0,1]
			keypoints_position[body_part] = sl::float4(NAN, NAN, NAN, score);
			const auto numberBodyParts = poseKeypoints.getSize(1);

			if (score < FLAGS_render_threshold) {
				poses.push_back({ keypoints_position[body_part].x, keypoints_position[body_part].y,  keypoints_position[body_part].z, keypoints_position[body_part].w });
				continue; // skip low score
			}

			i = round(poseKeypoints[{person, body_part, 0}]);	//x scaled to the original source resolution
			j = round(poseKeypoints[{person, body_part, 1}]);	//y scaled to the original source resolution

#if PATCH_AROUND_KEYPOINT
			xyz.getValue<sl::float4>(i, j, &keypoints_position[body_part], sl::MEM_CPU);	//Returns the value of z for (i,j) point in the matrix.
			if (!isfinite(keypoints_position[body_part].z))	 // if z value not valid: + Inf is "too far", -Inf is "too close", Nan is "unknown/occlusion"
				keypoints_position[body_part] = getPatchIdx((const int)i, (const int)j, xyz);
#else
			xyz.getValue<sl::float4>(i, j, &keypoints_position[body_part], sl::MEM_CPU);
#endif

			keypoints_position[body_part].w = score; // the score was overridden by the getValue

			poses.push_back({ keypoints_position[body_part].x, keypoints_position[body_part].y,  keypoints_position[body_part].z, keypoints_position[body_part].w });

			if (score >= FLAGS_render_threshold && isfinite(keypoints_position[body_part].z)) {
				center_gravity += keypoints_position[body_part];
				count++;
			}
		}
		//std::cout << "joints= " << keypoints_position[4].x << ", " << keypoints_position[4].y << ", " << keypoints_position[4].z << "\n";
		//std::cout << "joints= " << round(poseKeypoints[{person, 4, 0}]) << ", " << round(poseKeypoints[{person, 4, 1}]) << "\n";

		int arms_legs_joint[5] = { 0, 2, 3, 5, 6 }; //, 8, 9, 11, 12};
		ans = CylTest_arms_legs(keypoints_position, 0.05, 0.03, xyz, arms_legs_joint);

		poses.insert(poses.end(), ans.begin(), ans.end());

		//poses.insert(std::end(poses), std::begin(globalObjpy), std::end(globalObjpy));		//add object info to pose
		vector<float> posest = flatten(poses); // flatten vector of vectors into 1D

		if (posest.size() < 570000)
		{
			while (posest.size() < 570000)
			{
				posest.push_back(-1000.00);
			}
		}
				

		/*for (std::vector<float>::const_iterator i = posest.begin(); i != posest.end(); ++i)
			std::cout << *i << ' ';*/

		int psize = sizeof(posest[0])*posest.size();
		//cout << psize << endl;
		// Send frame to python
		//send(server_pose, (const char*)posest.data(), psize, 0);

		///////////////////////////
		center_gravity.x /= (float)count;
		center_gravity.y /= (float)count;
		center_gravity.z /= (float)count;


		for (int part = 0; part < partsLink.size() - 1; part += 2) {
			v1 = keypoints_position[partsLink[part]];
			v2 = keypoints_position[partsLink[part + 1]];

			// Filtering 3D Skeleton
			// Compute euclidian distance
			float distance = sqrt(pow((v1.x - v2.x), 2) + pow((v1.y - v2.y), 2) + pow((v1.z - v2.z), 2));
			float distance_gravity_center = sqrt(pow((v2.x + v1.x)*0.5f - center_gravity.x, 2) +
				pow((v2.y + v1.y)*0.5f - center_gravity.y, 2) +
				pow((v2.z + v1.z)*0.5f - center_gravity.z, 2));
			if (isfinite(distance_gravity_center) && distance_gravity_center < MAX_DISTANCE_CENTER && distance < MAX_DISTANCE_LIMB) {
				vertices.emplace_back(v1.x, v1.y, v1.z);
				vertices.emplace_back(v2.x, v2.y, v2.z);	// vertices now have v1 and v2 articulation points
				colr.push_back(generateColor(person));
				colr.push_back(generateColor(person));		// color red for v1 and v2
			}
		}
	}

	peopleObj.setVert(vertices, colr);
}

void fill_ptcloud(sl::Mat &xyzrgba) {
	std::vector<sl::float3> pts;
	std::vector<sl::float3> clr;
	int total = xyzrgba.getResolution().area();		// returns number of total pixels

	float factor = 1;

	pts.resize(total / factor);
	clr.resize(total / factor);

	sl::float4* p_mat = xyzrgba.getPtr<sl::float4>(sl::MEM_CPU);		// pointer to xyzrgba matrix: each pixel contains [x,y,z,color]

	sl::float3* p_f3;
	sl::float4* p_f4;
	unsigned char *color_uchar;

	int j = 0;
	for (int i = 0; i < total; i += factor, j++) {
		p_f4 = &p_mat[i];		//returns p_mat[i] address
		p_f3 = &pts[j];
		p_f3->x = p_f4->x;
		p_f3->y = p_f4->y;
		p_f3->z = p_f4->z;
		p_f3 = &clr[j];
		color_uchar = (unsigned char *)&p_f4->w;		// The color need to be read as an usigned char[4] representing the RGBA color
		p_f3->x = color_uchar[0] * 0.003921569; // /255
		p_f3->y = color_uchar[1] * 0.003921569;
		p_f3->z = color_uchar[2] * 0.003921569;
	}
	cloud.setVert(pts, clr);
}

void onMouse(int event, int i, int j, int flags, void*)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	cv::Point pt = cv::Point(i, j);
	std::vector<vector <float>> point_w;
	sl::float4 point3D;
	// Get the 3D point cloud values for pixel (i,j)
	point_cloud_click.getValue(i, j, &point3D);
	float x = point3D.x;
	float y = point3D.y;
	float z = point3D.z;
	//float color = point3D.w;

	// Client TCP to Send point cloud and poses to python
	WSADATA WSAData;
	SOCKET server_point;
	SOCKADDR_IN addr_point;

	WSAStartup(MAKEWORD(2, 0), &WSAData);
	server_point = socket(AF_INET, SOCK_STREAM, 0);
	addr_point.sin_addr.s_addr = inet_addr("127.0.0.1");
	addr_point.sin_family = AF_INET;
	addr_point.sin_port = htons(5599);
	connect(server_point, (SOCKADDR *)&addr_point, sizeof(addr_point));
	std::cout << "Connected to server send_point!" << endl;

	point_w.push_back({ point3D.x, point3D.y,  point3D.z });
	vector<float> point_get = flatten(point_w);		// flatten vector of vectors into 1D
	int psize2 = sizeof(point_get[0])*point_get.size();

	// Send frame to python
	//send(server_point, (const char*)point_get.data(), psize2, 0);

	std::cout << "x img col=" << pt.x << "\ty img col=" << pt.y << "\n";
	std::cout << "coordinates in space= " << x << ", " << y << ", " << z << "\n";
	//closesocket(server_point);
}


// This function loops to get image and motion data from the ZED. It is similar to a callback.
void run() {

	//Matrix to store each frame of the webcam feed
	cv::Mat cameraFeed;
	cv::Mat threshold;
	cv::Mat HSV;

	sl::RuntimeParameters rt;
	rt.enable_depth = 1;		// depth map computed if 1
	rt.enable_point_cloud = 1;	// point cloud computed if 1
	rt.measure3D_reference_frame = sl::REFERENCE_FRAME_WORLD;		//3D measures in the desired reference frame
																	// REFERENCE_FRAME_WORLD - the transform of sl::Pose with motion referenced to the world frame.
																	// REFERENCE_FRAME_CAMERA - the transform of sl::Pose with motion referenced to the previous camera frame

	sl::Mat img_buffer, depth_img_buffer, depth_buffer, depth_buffer2;
	op::Array<float> outputArray, outputArray2;
	cv::Mat inputImage, depthImage, inputImageRGBA, outputImage;

	// ---- OPENPOSE INIT (io data + renderer) ----
	op::ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap); //input scales and sizes
	op::CvMatToOpInput cvMatToOpInput;		// Input cvMat to OpenPose input format
	op::CvMatToOpOutput cvMatToOpOutput;	// Output cvMat to OpenPose output format

	op::PoseCpuRenderer poseRenderer{ poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending, (float)FLAGS_alpha_pose }; // After estimating the pose lets you visualize it
	op::OpOutputToCvMat opOutputToCvMat;	// OpenPose output format to cvMat

											// Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
	poseRenderer.initializationOnThread();

	// Init
	imageSize = op::Point<int>{ image_width, image_height };

	// Get desired scale sizes
	std::vector<op::Point<int>> netInputSizes;
	double scaleInputToOutput;
	op::Point<int> outputResolution;
	std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution) = scaleAndSizeExtractor.extract(imageSize);

	bool chrono_zed = false; // timer zed is off if false



	// Client TCP to Send point cloud and poses to python
	WSADATA WSAData;
	SOCKET server_pc, server_pose;
	SOCKADDR_IN addr_pc, addr_pose;

	WSAStartup(MAKEWORD(2, 0), &WSAData);
	server_pc = socket(AF_INET, SOCK_STREAM, 0);
	server_pose = socket(AF_INET, SOCK_STREAM, 0);
	addr_pc.sin_addr.s_addr = inet_addr("127.0.0.1");
	addr_pose.sin_addr.s_addr = inet_addr("127.0.0.1");
	addr_pc.sin_family = AF_INET;
	addr_pose.sin_family = AF_INET;
	addr_pc.sin_port = htons(5577);
	addr_pose.sin_port = htons(5588);
	connect(server_pc, (SOCKADDR *)&addr_pc, sizeof(addr_pc));
	std::cout << "Connected to server point cloud!" << endl;
	connect(server_pose, (SOCKADDR *)&addr_pose, sizeof(addr_pose));
	std::cout << "Connected to server pose!" << endl;

	// PCL Viewer
	// Allocate PCL point cloud at the resolution
	
	/*pcl::PointCloud<pcl::PointXYZRGB>::Ptr p_pcl_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	p_pcl_point_cloud->points.resize(zed.getResolution().area());
	//shared_ptr<pcl::visualization::PCLVisualizer> viewer2 = createRGBVisualizer(p_pcl_point_cloud);*/

	while (!quit && zed.getSVOPosition() != zed.getSVONumberOfFrames() - 1) {		//getSVO returns -1 if not reading from recording
		INIT_TIMER
			/*float cx = zed.getCameraInformation().calibration_parameters.left_cam.cx;
			float cy = zed.getCameraInformation().calibration_parameters.left_cam.cy;
			float fy = zed.getCameraInformation().calibration_parameters.left_cam.fy;
			float fx = zed.getCameraInformation().calibration_parameters.left_cam.fx;*/
			if (need_new_image) {
				if (zed.grab(rt) == SUCCESS) {		// If grab of last image was success

					zed.retrieveImage(img_buffer, VIEW::VIEW_LEFT, sl::MEM_CPU, image_width, image_height); // retrives image
					data_out_mtx.lock();
					depth_buffer2 = depth_buffer;
					zed.retrieveMeasure(depth_buffer, MEASURE::MEASURE_XYZRGBA, sl::MEM_CPU, image_width, image_height); // computes depth into depth_buffer
					point_cloud_click = depth_buffer;
					data_out_mtx.unlock();

					// OBJECT TRACKING
					
					cameraFeed = slMat2cvMat(img_buffer);
					//convert frame from BGR to HSV colorspace
					cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
					Object blue("blue"), yellow("yellow"), red("red"), green("green");
					// find yellow objects
					cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
					inRange(HSV, yellow.getHSVmin(), yellow.getHSVmax(), threshold);
					morphOps(threshold);
					trackFilteredObject(yellow, threshold, HSV, cameraFeed, depth_buffer);

					//then reds
					cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
					inRange(HSV, red.getHSVmin(), red.getHSVmax(), threshold);
					morphOps(threshold);
					trackFilteredObject(red, threshold, HSV, cameraFeed, depth_buffer);
					//then greens
					cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
					inRange(HSV, green.getHSVmin(), green.getHSVmax(), threshold);
					morphOps(threshold);
					trackFilteredObject(green, threshold, HSV, cameraFeed, depth_buffer);
					//find blue objects
					/*
					cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
					inRange(HSV, blue.getHSVmin(), blue.getHSVmax(), threshold);
					morphOps(threshold);
					trackFilteredObject(blue, threshold, HSV, cameraFeed, depth_buffer);*/

					/*sl::float4 point_cloud_value;
					int x = (zed.getResolution().width / 2) + 22; // Center coordinates
					int y = (zed.getResolution().height / 2) + 22;
					depth_buffer.getValue(x, y, &point_cloud_value);
					printf("Z point cloud at (%d, %d): %0.3f, %0.3f, %0.3f mm\n", x, y, point_cloud_value.x, point_cloud_value.y, point_cloud_value.z);
					Sleep(1000);*/

					/*
					pcl::search::Search <pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
					float *p_data_cloud = depth_buffer.getPtr<float>();
					int index = 0;
					// Check and adjust points for PCL format
					for (auto &itw : p_pcl_point_cloud->points) {
						float X = p_data_cloud[index];
						if (!isValidMeasure(X))
						{// Checking if it's a valid point
							itw.x = itw.y = itw.z = itw.rgb = 0;
						}
						else {
							
							itw.x = X;
							itw.y = p_data_cloud[index + 1];
							itw.z = p_data_cloud[index + 2];
							itw.rgb = convertColor(p_data_cloud[index + 3]); // Convert a 32bits float into a pcl .rgb format
						}
						index += 4;
					}

					viewer2->updatePointCloud(p_pcl_point_cloud, "cloud");
					viewer2->spinOnce(10);
					*/
				
					inputImageRGBA = slMat2cvMat(img_buffer);		// converts zed mat to cv mat
					cv::cvtColor(inputImageRGBA, inputImage, CV_RGBA2RGB); // converts color image

					if (FLAGS_depth_display) // if to display depth image
						zed.retrieveImage(depth_img_buffer, VIEW::VIEW_DEPTH, sl::MEM_CPU, image_width, image_height);

					if (FLAGS_opencv_display) {
						data_out_mtx.lock();
						outputArray2 = outputArray;
						data_out_mtx.unlock();
						outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
					}
					data_in_mtx.lock();
					netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
					need_new_image = false;
					data_in_mtx.unlock();

					ready_to_start = true; // allows to initiate findpose thread
					chrono_zed = true;
				}
				else sl::sleep_ms(1); // tells program to wait 1 ms (see if grabs image meanwhile)
			}
			else sl::sleep_ms(1);

			// -------------------------  RENDERING -------------------------------
			// Render poseKeypoints
			if (data_out_mtx.try_lock()) { //if mutex available to lock
				cv::Mat point_cloud = slMat2cvMat(depth_buffer2);
				//fill_people_ogl(poseKeypoints, depth_buffer2, server_pose);
				int imgsize = point_cloud.total()*point_cloud.elemSize();
				cout << imgsize << endl;
				const char *img_send = reinterpret_cast<char*>(point_cloud.data);
				long int before = GetTickCount();
				send(server_pc, img_send, imgsize, 0);		// server_op
				long int after = GetTickCount();
				long int elapsed = after - before;
				cout << elapsed << endl;
				viewer.update(peopleObj);

				if (FLAGS_ogl_ptcloud) {	// if display point cloud
					fill_ptcloud(depth_buffer2);
					viewer.update(cloud);
				}

				if (FLAGS_opencv_display) {
					if (!outputArray2.empty())
						poseRenderer.renderPose(outputArray2, poseKeypoints, scaleInputToOutput);
					// OpenPose output format to cv::Mat
					if (!outputArray2.empty())
						outputImage = opOutputToCvMat.formatToCvMat(outputArray2);
					data_out_mtx.unlock();
					// Show results
					if (!outputImage.empty()) {
						cv::setMouseCallback("Pose", onMouse, 0);
						cv::Mat output_rescale;
						//cv::resize(outputImage, output_rescale, cv::Size(720, 480));
						cv::imshow("Pose", outputImage);
					}
					if (FLAGS_depth_display)
						cv::imshow("Depth", slMat2cvMat(depth_img_buffer));

					cv::waitKey(10);
				}

				// Send frame to python
				/*int imgsize = outputImage.total()*outputImage.elemSize();
				const char *img_send = reinterpret_cast<char*>(outputImage.data);
				send(server_op, img_send, imgsize, 0);*/
			}

			if (chrono_zed) {
				//STOP_TIMER("ZED")
				chrono_zed = false;
			}
	}
}

// This function closes the ZED camera, openpose, their callback (thread) and the GL viewer
void close() {
	quit = true;
	openpose_callback.join();
	zed_callback.join();
	zed.close();
	viewer.exit();
}


/**
*  This function creates a PCL visualizer


shared_ptr<pcl::visualization::PCLVisualizer> createRGBVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud2) {
	// Open 3D viewer and add point cloud
	shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("PCL ZED 3D Viewer"));
	viewer2->setBackgroundColor(0.12, 0.12, 0.12);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(cloud2);
	viewer2->addPointCloud<pcl::PointXYZRGB>(cloud2, rgb2);
	viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5);
	viewer2->addCoordinateSystem(1.0);
	viewer2->initCameraParameters();
	return (viewer2);
}
**/


/**
*  This function convert a RGBA color packed into a packed RGBA PCL compatible format
**/
inline float convertColor(float colorIn) {
	uint32_t color_uint = *(uint32_t *)& colorIn;
	unsigned char *color_uchar = (unsigned char *)&color_uint;
	color_uint = ((uint32_t)color_uchar[0] << 16 | (uint32_t)color_uchar[1] << 8 | (uint32_t)color_uchar[2]);
	return *reinterpret_cast<float *> (&color_uint);
}


