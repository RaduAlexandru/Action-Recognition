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

#include "Volume.hh"
#include "Histogram.hh"
#include "ArrayFastest.hh"

#include <iostream>
#include <fstream>

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
	std::vector<cv::Mat> m_st_volume; //Spacial-temporal volume correspionding to the whole video

	cv::Mat mat2gray(const cv::Mat& src);
	void create_spacial_temporal_volume(std::vector<cv::Mat>& m_st_volume, std::string m_video_path); //read video file and append all the frames together
	void gaussian_smooth(std::vector<cv::Mat> vol_in, std::vector<cv::Mat>& vol_out, float m_sigma_local_spacial, float m_sigma_local_temporal);
	void compute_derivatives(std::vector<cv::Mat>& m_st_volume, std::vector<cv::Mat>& Lx, std::vector<cv::Mat>& Ly, std::vector<cv::Mat>& Lt);
	void compute_harris_coefficients(std::vector<cv::Mat>& Lx, std::vector<cv::Mat>& Ly, std::vector<cv::Mat>&Lt,
	                                       std::vector<cv::Mat>& Lx2, std::vector<cv::Mat>& Ly2, std::vector<cv::Mat>& Lt2,
	                                       std::vector<cv::Mat>& LxLy, std::vector<cv::Mat>& LxLt, std::vector<cv::Mat>&LyLt);
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
  void write_descriptors_to_file(std::vector<interest_point> interest_points, std::ofstream& file);


};

} // namespace


#endif /* ACTIONRECOGNITION_STIP_HH_ */
