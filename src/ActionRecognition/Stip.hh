/*
 * Stip.hh
 *
 *  Created on: Apr 25, 2017
 *      Author: richard
 */

#ifndef ACTIONRECOGNITION_STIP_HH_
#define ACTIONRECOGNITION_STIP_HH_

#include "Core/CommonHeaders.hh" // Core::Log, Core::Error, Core::Configuration, Types
#include "Math/Matrix.hh"
#include "Math/Vector.hh"
#include "KMeans.hh"

#include "Volume.hh"
#include "Histogram.hh"
#include "ArrayFastest.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>

#include "opencv2/video/tracking.hpp"

namespace ActionRecognition {

struct interest_point {
  int t,x,y; //position of the center in spacial-temporal volume
	float sigma_spacial, sigma_temporal;
	float response;
	cv::Rect bb;
  Histogram descriptor;
} ;


class Stip
{
public:
	Stip();
	virtual ~Stip();
	void run();

private:
	std::string m_video_path;
	float m_sigma_local_spacial;
	float m_sigma_local_temporal;
	std::vector<cv::Mat> m_st_volume; //Spacial-temporal volume corresponding to the whole video

  //convenience method to display an image
	cv::Mat mat2gray(const cv::Mat& src);
  //read video file and append all the frames together
	void create_spacial_temporal_volume(std::vector<cv::Mat>& m_st_volume, std::string m_video_path);
  //smoothes a st-volume in both spacial and temporal domain given certain sigmas
	void gaussian_smooth(std::vector<cv::Mat> vol_in, std::vector<cv::Mat>& vol_out, float m_sigma_local_spacial, float m_sigma_local_temporal);
  //computes the derivatives in all directions in an ST-volume
	void compute_derivatives(std::vector<cv::Mat>& m_st_volume, std::vector<cv::Mat>& Lx, std::vector<cv::Mat>& Ly, std::vector<cv::Mat>& Lt);
  //computes the coefficients needed for the 3x3 harris matrix
	void compute_harris_coefficients(std::vector<cv::Mat>& Lx, std::vector<cv::Mat>& Ly, std::vector<cv::Mat>&Lt,
	                                       std::vector<cv::Mat>& Lx2, std::vector<cv::Mat>& Ly2, std::vector<cv::Mat>& Lt2,
	                                       std::vector<cv::Mat>& LxLy, std::vector<cv::Mat>& LxLt, std::vector<cv::Mat>&LyLt);
  //gives a harris responde for every point in the st-volume
  void compute_harris_responses(std::vector<cv::Mat>& harris_responses, std::vector<cv::Mat>& Lx2, std::vector<cv::Mat>& Ly2,
	                              std::vector<cv::Mat>& Lt2, std::vector<cv::Mat>& LxLy, std::vector<cv::Mat>& LxLt, std::vector<cv::Mat>&LyLt);
  void compute_local_maxima(std::vector<cv::Mat>&  harris_responses,   std::vector<interest_point>& interest_points);
	void draw_interest_points (cv::Mat frame, std::vector<interest_point >& interest_points, int time);

	void non_max_supress(std::vector<interest_point>& interest_points, int max_t, int max_y, int max_x);
	void nms( const std::vector<interest_point>& srcRects, std::vector<interest_point>& resRects, float thresh);

  //DESCRIPTORS
  void compute_grad_orientations_magnitudes(std::vector<cv::Mat> Lx, std::vector<cv::Mat> Ly, std::vector<cv::Mat>& grad_mags, std::vector<cv::Mat>& grad_orientations );
  void compute_flow(std::vector<cv::Mat> m_st_volume, std::vector<cv::Mat>& flow_mags, std::vector<cv::Mat>& flow_orientations );
  void compute_descriptors(std::vector<interest_point>& interest_points, std::vector<cv::Mat> grad_mags, std::vector<cv::Mat> grad_orientations, std::vector<cv::Mat> flow_mags, std::vector<cv::Mat> flow_orientations);
  int write_descriptors_to_file(std::vector<interest_point> interest_points, std::ofstream& file);
  void read_features_from_file(std::string descriptor_file_path, Math::Matrix<Float>& features );
  void read_features_per_video_from_file(std::string descriptor_file_path, std::vector<Math::Matrix<Float> >& features_per_video, int max_nr_videos);


  //Everything
  void task_1_2(std::string descriptor_file_path);
  void task_3_train(std::string descriptor_file_path, KMeans& kmeans);
  void task_3_bow(std::string descriptor_file_path , KMeans& kmeans, std::vector<std::vector <float> >& bow_per_video);


};

} // namespace


#endif /* ACTIONRECOGNITION_STIP_HH_ */
