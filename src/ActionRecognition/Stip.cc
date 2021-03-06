/*
 * Stip.cc
 *
 *  Created on: Apr 25, 2017
 *      Author: richard
 */

#include "Stip.hh"

using namespace ActionRecognition;

// constructor
Stip::Stip(){

  //Sigmas used to smooth the  ST-volume
  m_sigma_local_spacial=1.0f;
  m_sigma_local_temporal=2.0f;

}

// empty destructor
Stip::~Stip()
{}

void Stip::run(){
  std::cout << "run stip" << '\n';

  // m_video_path= "../experiments/videos/dummy.avi";
  m_video_path= "../experiments/videos/Torwarttraining_2_(_sterreich)_catch_f_cm_np1_ba_goo_1.avi";

  KMeans kmeans;
  std::string descriptor_file_path="./desc.txt";
  std::vector<std::vector <float> > bow_per_video;


  task_1_2(descriptor_file_path);
  task_3_train(descriptor_file_path, kmeans);
  task_3_bow(descriptor_file_path, kmeans, bow_per_video);
  // task_4(bow_per_video); //TODO


}

//Everything
void Stip::task_1_2(std::string descriptor_file_path){
  std::cout << "task 1 and 2" << '\n';

  std::ofstream desc_file;
  desc_file.open (descriptor_file_path);

  std::string train_file_path = "../experiments/videos/videos/full.txt";
  std::ifstream train_file( train_file_path );
  int nr_vectors=0;
  int vector_dimensions=0;
  int nr_videos=0;


  for( std::string line; getline( train_file, line ); ){
    m_video_path =  "../experiments/videos/videos/" + line;
    std::cout << "computing interest points for video " << line << '\n';

    std::vector<cv::Mat> Lx,Ly,Lt;
    std::vector<cv::Mat> Lx2,Ly2,Lt2,LxLy,LxLt,LyLt;
    std::vector<cv::Mat> Lx2_s,Ly2_s,Lt2_s,LxLy_s,LxLt_s,LyLt_s;
    std::vector<cv::Mat> harris_responses;
    std::vector<interest_point> interest_points;

    create_spacial_temporal_volume(m_st_volume,m_video_path);
    gaussian_smooth(m_st_volume, m_st_volume, m_sigma_local_spacial, m_sigma_local_temporal);
    compute_derivatives(m_st_volume, Lx,Ly,Lt);
    compute_harris_coefficients(Lx,Ly,Lt, Lx2,Ly2,Lt2,LxLy,LxLt,LyLt);

    float sigma_integration_spacial=5.0f;
    float sigma_integration_temporal=2.0f;

    //alexanders settings
    for (size_t i = 1; i < 4; i++) {
      for (size_t j = 1; j < 3; j++) {
        sigma_integration_spacial=i;
        sigma_integration_temporal=j;


        gaussian_smooth(Lx2, Lx2_s, sigma_integration_spacial, sigma_integration_temporal);
        gaussian_smooth(Ly2, Ly2_s, sigma_integration_spacial, sigma_integration_temporal);
        gaussian_smooth(Lt2, Lt2_s, sigma_integration_spacial, sigma_integration_temporal);
        gaussian_smooth(LxLy, LxLy_s, sigma_integration_spacial, sigma_integration_temporal);
        gaussian_smooth(LxLt, LxLt_s, sigma_integration_spacial, sigma_integration_temporal);
        gaussian_smooth(LyLt, LyLt_s, sigma_integration_spacial, sigma_integration_temporal);

        compute_harris_responses(harris_responses,Lx2_s,Ly2_s,Lt2_s,LxLy_s,LxLt_s,LyLt_s);

        int start_idx=interest_points.size();

        compute_local_maxima(harris_responses,interest_points);

        //add the scale to them
        for (size_t i =start_idx; i < interest_points.size(); i++) {
          interest_points[i].sigma_spacial=sigma_integration_spacial;
          interest_points[i].sigma_temporal=sigma_integration_temporal;
        }

      }
    }

    non_max_supress(interest_points, m_st_volume.size(), m_st_volume[0].rows, m_st_volume[0].cols);
    non_max_supress(interest_points, m_st_volume.size(), m_st_volume[0].rows, m_st_volume[0].cols);



    //---------------DESCRIPTORS---------------------------
    std::vector<cv::Mat> grad_mags;
    std::vector<cv::Mat> grad_orientations;
    std::vector<cv::Mat> flow_mags;
    std::vector<cv::Mat> flow_orientations;


    compute_grad_orientations_magnitudes(Lx,Ly, grad_mags, grad_orientations );
    compute_flow(m_st_volume, flow_mags, flow_orientations );
    compute_descriptors(interest_points, grad_mags, grad_orientations, flow_mags, flow_orientations);

    int descriptor_written=write_descriptors_to_file(interest_points, desc_file);

    desc_file << "#"<< std::endl;


    nr_vectors+=descriptor_written;
    vector_dimensions=162;  //TODO remove hardcode
    nr_videos++;



    //Show volume with interest_points
    bool showing=true;
    while (showing)
    for (size_t i = 0; i < harris_responses.size(); i++) {
      // double min, max;
      // cv::minMaxLoc(harris_responses[i], &min, &max);
      // std::cout << "min max is " << min << " " << max << '\n';
      cv::Mat frame;
      frame=mat2gray(m_st_volume[i]);
      cv::cvtColor(frame, frame, CV_GRAY2BGR);
      draw_interest_points(frame,interest_points,i);
      cv::imshow("window", frame);
      // cv::waitKey(0);
      char key = cvWaitKey(50);
          if (key == 27){ // ESC
              showing=false;
              break;
            }
    }

  }

  //Add header to file
  desc_file.seekp(0); //Move at start of file
  desc_file << nr_vectors << " " << vector_dimensions << " " << nr_videos << std::endl;
  desc_file.close();
}
void Stip::task_3_train(std::string descriptor_file_path, KMeans& kmeans){
  Math::Matrix<Float> features;
  std::cout << "reading features..." << '\n';
  read_features_from_file(descriptor_file_path, features);
  std::cout << "training kmeans, grab a beer..." << '\n';
  kmeans.train(features);
}
void Stip::task_3_bow(std::string descriptor_file_path , KMeans& kmeans, std::vector<std::vector <float> >& bow_per_video){
  // BoW each video
  kmeans.loadModel();
  std::vector<Math::Matrix<Float>> features_per_video;
  int n_train_videos=226;
  read_features_per_video_from_file(descriptor_file_path, features_per_video,n_train_videos);

  bow_per_video.resize(n_train_videos);

  for (size_t i = 0; i < n_train_videos; i++) {
    Math::Vector<u32> clusterIndices;
    kmeans.cluster(features_per_video[i], clusterIndices);

    //see how many times does each feature end up in a cluster and make your bow histogram
    std::vector <float> bow_hist(kmeans.nClusters());
    for (size_t j = 0; j < features_per_video[i].nColumns(); j++) {
      int assigned_cluster= clusterIndices.at(j);
      bow_hist[assigned_cluster]++;
    }

    bow_per_video.push_back(bow_hist);

    std::cout << "finished BoW for video " << i << '\n';

  }

}




cv::Mat Stip::mat2gray(const cv::Mat& src){
    cv::Mat dst;
    cv::normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

    return dst;
}

void Stip::create_spacial_temporal_volume(std::vector<cv::Mat>& m_st_volume, std::string m_video_path){
  m_st_volume.clear();

  //read the file
  cv::VideoCapture cap(m_video_path);
  if (!cap.isOpened()){
      std::cout << "!!! Failed to open file: " << m_video_path << std::endl;
      return;
  }

  //append all the frame into a spacial temporal volume
  cv::Mat frame;
  while(true) {
    if (!cap.read(frame))
        break;
    cvtColor(frame,frame,CV_BGR2GRAY);
    frame.convertTo(frame, CV_32FC1);
    cv::normalize(frame, frame, 0.0f, 1.0f, cv::NORM_MINMAX, CV_32FC1); //Normalize to 1 to avoid underflow or overflow

    m_st_volume.push_back(frame.clone());

    // cv::imshow("window", frame);
    // char key = cvWaitKey(30);
    //     if (key == 27) // ESC
    //         break;
  }
}

void Stip::gaussian_smooth(std::vector<cv::Mat> vol_in, std::vector<cv::Mat>& vol_out, float m_sigma_local_spacial, float m_sigma_local_temporal){

  //get kernels
  int kernel_size=15;
  cv::Mat ker_temporal = cv::getGaussianKernel(kernel_size, m_sigma_local_temporal, CV_32F );
  transpose(ker_temporal,ker_temporal);



  if (vol_out.size()!=vol_in.size()){
    vol_out.resize(vol_in.size());
  }


  //spacial
  for (size_t i = 0; i < m_st_volume.size(); i++) {
    GaussianBlur(vol_in[i],vol_out[i], cv::Size(0,0),m_sigma_local_spacial );
  }

  //temporal
  //copy a slice from the y-t dimensions into a cvmat, smooth it and copy it back
  cv::Mat slice=cv::Mat(vol_in[0].rows, vol_in.size(), vol_in[0].depth());
  for (size_t x = 0; x < vol_in[0].cols; x++) {

    //Loop through all the pixels in the y-t slice and copy them into the mat
    for (size_t y = 0; y < vol_in[0].rows; y++) {
      for (size_t t = 0; t < vol_in.size(); t++) {
        slice.at<float>(y,t)= vol_out[t].at<float>(y,x);
      }
    }


    //apply a 1d gausian only in the x direction, which corresponds to time
    filter2D(slice, slice, -1, ker_temporal);


    //copy it back
    for (size_t y = 0; y < m_st_volume[0].rows; y++) {
      for (size_t t = 0; t < m_st_volume.size(); t++) {
        vol_out[t].at<float>(y,x)=slice.at<float>(y,t);
      }
    }
  }

}

void Stip::compute_derivatives(std::vector<cv::Mat>& m_st_volume, std::vector<cv::Mat>& Lx, std::vector<cv::Mat>& Ly, std::vector<cv::Mat>& Lt){

  Lx.clear();
  Ly.clear();
  Lt.clear();

  //derivatives in x and y
  cv::Mat Ix,Iy,It;
  for (size_t i = 0; i < m_st_volume.size(); i++) {
    cv::Sobel( m_st_volume[i], Ix, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel( m_st_volume[i], Iy, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

    Lx.push_back(Ix.clone());
    Ly.push_back(Iy.clone());
  }


  //Temporal derivatives by doing a sobel in a y-t slice
  cv::Mat slice=cv::Mat(m_st_volume[0].rows, m_st_volume.size(), CV_32F);

  //allocate memory for Lt
  for (size_t i = 0; i < Lx.size(); i++) {
    Lt.push_back(cv::Mat(Ix.rows, Ix.cols, CV_32F));
  }

  for (size_t x = 0; x < m_st_volume[0].cols; x++) {

    //Loop through all the pixels in the y-t slice and copy them into the mat
    for (size_t y = 0; y < m_st_volume[0].rows; y++) {
      for (size_t t = 0; t < m_st_volume.size(); t++) {
        slice.at<float>(y,t)= m_st_volume[t].at<float>(y,x);
      }
    }


    cv::Sobel( slice, slice, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);


    //copy it back
    for (size_t y = 0; y < m_st_volume[0].rows; y++) {
      for (size_t t = 0; t < m_st_volume.size(); t++) {
        Lt[t].at<float>(y,x)=slice.at<float>(y,t);
      }
    }
  }





}

void Stip::compute_harris_coefficients(std::vector<cv::Mat>& Lx, std::vector<cv::Mat>& Ly, std::vector<cv::Mat>&Lt,
                                       std::vector<cv::Mat>& Lx2, std::vector<cv::Mat>& Ly2, std::vector<cv::Mat>& Lt2,
                                       std::vector<cv::Mat>& LxLy, std::vector<cv::Mat>& LxLt, std::vector<cv::Mat>&LyLt){

  int num_frames=Lx.size();
  Lx2.resize(num_frames);
  Ly2.resize(num_frames);
  Lt2.resize(num_frames);
  LxLy.resize(num_frames);
  LxLt.resize(num_frames);
  LyLt.resize(num_frames);

  for (size_t i = 0; i < Lx.size(); i++) {
    cv::multiply(Lx[i],Lx[i],Lx2[i]);
    cv::multiply(Ly[i],Ly[i],Ly2[i]);
    cv::multiply(Lt[i],Lt[i],Lt2[i]);

    cv::multiply(Lx[i],Ly[i],LxLy[i]);
    cv::multiply(Lx[i],Lt[i],LxLt[i]);
    cv::multiply(Ly[i],Lt[i],LyLt[i]);
  }
}

void Stip::compute_harris_responses(std::vector<cv::Mat>& harris_responses, std::vector<cv::Mat>& Lx2, std::vector<cv::Mat>& Ly2,
                                    std::vector<cv::Mat>& Lt2, std::vector<cv::Mat>& LxLy, std::vector<cv::Mat>& LxLt, std::vector<cv::Mat>&LyLt){

   harris_responses.resize(Lx2.size());

   float k=0.005;
   cv::Mat tmp1=cv::Mat::zeros(Lx2[0].rows,Lx2[0].cols,CV_32F);
   cv::Mat tmp2=cv::Mat::zeros(Lx2[0].rows,Lx2[0].cols,CV_32F);

   for (size_t i = 0; i < harris_responses.size(); i++) {
     cv::multiply(Lx2[i], Ly2[i], tmp1);
     cv::multiply(Lt2[i], tmp1, tmp1);

     cv::multiply(LxLy[i], LxLt[i], tmp2);
     cv::multiply(LyLt[i], tmp2, tmp2,2);

     cv::add(tmp1,tmp2,tmp1);

     cv::multiply(LyLt[i],LyLt[i],tmp2);
     cv::multiply(Lx2[i],tmp2,tmp2);

     cv::subtract(tmp1,tmp2,tmp1);

     cv::multiply(LxLy[i],LxLy[i],tmp2);
     cv::multiply(Lt2[i],tmp2,tmp2);

     cv::subtract(tmp1,tmp2,tmp1);

     cv::multiply(LxLt[i],LxLt[i],tmp2);
     cv::multiply(Ly2[i],tmp2,tmp2);

     cv::subtract(tmp1,tmp2,tmp1);

     //trace3C=(cxx+cyy+ctt).^3;
     cv::add(Lx2[i],Ly2[i],tmp2);
     cv::add(Lt2[i],tmp2,tmp2);
     cv::pow(tmp2,3,tmp2);

     //H=detC-stharrisbuffer.kparam*trace3C;
    //  cv::scale(tmp2,tmp2,k,0);
    tmp2.convertTo(tmp2,tmp2.depth(),k,0);
    cv::subtract(tmp1,tmp2,harris_responses[i]);
   }

}

void Stip::compute_local_maxima(std::vector<cv::Mat>&  harris_responses,   std::vector<interest_point >& interest_points){

  float thresh=6e-6;  //Need a small treshold so as to avoid eroneous interest points in regions which are static

  for (int t = 0; t < harris_responses.size(); t++) {
    for (int y = 0; y < harris_responses[0].rows; y++) {
      for (int x = 0; x < harris_responses[0].cols; x++) {

        int neighbours_surpassed=0;
        //for points at position x,y,t check all the neighbours around it and if it is bigger than all of them then its an interest point
        for (int n_t = std::max(0,t-1); n_t < std::min((int)harris_responses.size(),t+2); n_t++) {
          for (int n_y = std::max(0,y-1); n_y < std::min(harris_responses[0].rows,y+2); n_y++){
            for (int n_x = std::max(0,x-1); n_x < std::min(harris_responses[0].cols,x+2); n_x++){

              if (n_t==t && n_y==y && n_x==x  ){
                continue;
              }

              if (harris_responses[t].at<float>(y,x) > harris_responses[n_t].at<float>(n_y,n_x) + thresh){
                neighbours_surpassed++;
              }

            }
          }
        }

        //If the point has a higher response than all the neighbours then it is a local maxima
        if (neighbours_surpassed==26){
          interest_point int_p;
          int_p.t=t;
          int_p.y=y;
          int_p.x=x;
          interest_points.push_back(int_p);
        }

      }
    }
  }


}

void Stip::draw_interest_points (cv::Mat frame, std::vector<interest_point >& interest_points, int time){

  int drawn_ips=0;
  for (size_t i = 0; i < interest_points.size(); i++) {
    //the points were detected at this time
    if (time==interest_points[i].t) {
      cv::Point center =  cv::Point(interest_points[i].x,interest_points[i].y);
      float size_spacial=interest_points[i].sigma_spacial;
      cv::circle(frame, center, size_spacial, cv::Scalar(0,0,255));
      drawn_ips++;
    }
  }



}

void Stip::non_max_supress(std::vector<interest_point>& interest_points, int max_t, int max_y, int max_x){

  std::vector<interest_point> ip_spacial_supressed;
  std::vector<interest_point> ip_complete_supressed;

  //supress in spacial domain
  for (size_t t = 0; t < max_t; t++) {
    std::vector<interest_point> ip_time;

    //get all the interest points at this time
    for (size_t i = 0; i < interest_points.size(); i++) {
      if (interest_points[i].t==t){
        ip_time.push_back(interest_points[i]);
      }
    }

    //get thir bounding boxes
    for (size_t i = 0; i < ip_time.size(); i++) {

      int x=ip_time[i].x;
      int y=ip_time[i].y;
      int w=std::ceil(ip_time[i].sigma_spacial);
      int h=std::ceil(ip_time[i].sigma_spacial);

      ip_time[i].bb=cv::Rect(cv::Point(x-w/2,y-h/2), cv::Point(x+w/2,y+h/2));
    }


    //nms
    std::vector<interest_point> ip_nms;
    nms(ip_time,ip_nms,0);

    //append the ones at this time to the new ones
    ip_spacial_supressed.insert(ip_spacial_supressed.end(), ip_nms.begin(), ip_nms.end());

  }

  //temporal-----------
  for (size_t x = 0; x < max_x; x++) {
    std::vector<interest_point> ip_x;

    //get all the interest points at this x value
    for (size_t i = 0; i < ip_spacial_supressed.size(); i++) {
      if (ip_spacial_supressed[i].x==x){
        ip_x.push_back(ip_spacial_supressed[i]);
      }
    }

    //get thir bounding boxes
    for (size_t i = 0; i < ip_x.size(); i++) {

      int x=ip_x[i].t;
      int y=ip_x[i].y;
      int w=std::ceil(ip_x[i].sigma_temporal);
      int h=std::ceil(ip_x[i].sigma_temporal);

      ip_x[i].bb=cv::Rect(cv::Point(x-w/2,y-h/2), cv::Point(x+w/2,y+h/2));
    }


    //nms
    std::vector<interest_point> ip_nms;
    nms(ip_x,ip_nms,0);

    //append the ones at this time to the new ones
    ip_complete_supressed.insert(ip_complete_supressed.end(), ip_nms.begin(), ip_nms.end());

  }


  interest_points=ip_complete_supressed;

}

/*Adapted from https://github.com/Nuzhny007/Non-Maximum-Suppression/blob/master/main.cpp*/
void Stip::nms( const std::vector<interest_point>& srcRects, std::vector<interest_point>& resRects, float thresh){
	resRects.clear();

	const size_t size = srcRects.size();
	if (!size)
	{
		return;
	}

	// Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
	std::multimap<int, size_t> idxs;
	for (size_t i = 0; i < size; ++i)
	{
		idxs.insert(std::pair<int, size_t>(srcRects[i].bb.br().y, i));
	}

	// keep looping while some indexes still remain in the indexes list
	while (idxs.size() > 0)
	{
		// grab the last rectangle
		auto lastElem = --std::end(idxs);
		const interest_point& rect1 = srcRects[lastElem->second];

		resRects.push_back(rect1);

		idxs.erase(lastElem);

		for (auto pos = std::begin(idxs); pos != std::end(idxs); )
		{
			// grab the current rectangle
			const interest_point& rect2 = srcRects[pos->second];

			float intArea = (rect1.bb & rect2.bb).area();
			float unionArea = rect1.bb.area() + rect2.bb.area() - intArea;
			float overlap = intArea / unionArea;

			// if there is sufficient overlap, suppress the current bounding box
			if (overlap > thresh)
			{
				pos = idxs.erase(pos);
			}
			else
			{
				++pos;
			}
		}
	}
}

void Stip::compute_grad_orientations_magnitudes(std::vector<cv::Mat> Lx, std::vector<cv::Mat> Ly, std::vector<cv::Mat>& grad_mags, std::vector<cv::Mat>& grad_orientations ){
  std::cout << "compute_grad_orientations_magnitudes" << '\n';

  grad_mags.clear();
  grad_orientations.clear();

  bool useDegree = true;    // use degree or rad
  grad_mags.resize(Lx.size());
  grad_orientations.resize(Lx.size());

  for (size_t i = 0; i < Lx.size(); i++) {
    // the range of the direction is [0,2pi) or [0, 360)
    cv::cartToPolar(Lx[i], Ly[i], grad_mags[i], grad_orientations[i], useDegree);
  }


}

void Stip::compute_flow(std::vector<cv::Mat> m_st_volume, std::vector<cv::Mat>& flow_mags, std::vector<cv::Mat>& flow_orientations ){
  std::cout << "compute flow" << '\n';

  flow_mags.clear();
  flow_orientations.clear();


  flow_mags.resize(m_st_volume.size());
  flow_orientations.resize(m_st_volume.size());

  cv::Mat flow;
  cv::Mat flow_xy[2];

  for (size_t i = 0; i < m_st_volume.size(); i++) {

    int next_idx=i+1;
    if (next_idx == m_st_volume.size() ){
      next_idx=m_st_volume.size()-1;
    }


    cv::calcOpticalFlowFarneback(m_st_volume[i], m_st_volume[next_idx], flow, 0.4, 1, 12, 2, 8, 1.2, 0);
    cv::split(flow, flow_xy);
    //calculate angle and magnitude
    cv::Mat magnitude, angle;
    cartToPolar(flow_xy[0], flow_xy[1], flow_mags[i], flow_orientations[i], true);

  }

}

void Stip::compute_descriptors(std::vector<interest_point>& interest_points, std::vector<cv::Mat> grad_mags, std::vector<cv::Mat> grad_orientations, std::vector<cv::Mat> flow_mags, std::vector<cv::Mat> flow_orientations){

  std::cout << "compute_descriptor" << '\n';

  int k=9;
  int nbins_hog=4;
  float hist_range=180.0f;

  int st_max_x=grad_orientations[0].cols;
  int st_max_y=grad_orientations[0].rows;
  int st_max_t=grad_orientations.size();

  int cell_per_vol_x=3;
  int cell_per_vol_y=3;
  int cell_per_vol_t=2;


  for (size_t i = 0; i < interest_points.size(); i++) {

    int vol_x_size=2*k*interest_points[i].sigma_spacial;
    int vol_y_size=2*k*interest_points[i].sigma_spacial;
    int vol_t_size=2*k*interest_points[i].sigma_temporal;

    int cell_size_x=std::ceil(vol_x_size/(float)cell_per_vol_x);
    int cell_size_y=std::ceil(vol_y_size/(float)cell_per_vol_y);
    int cell_size_t=std::ceil(vol_t_size/(float)cell_per_vol_t);


    //if the cube arond the interest points is outside the spacial temporal volume we ignore it
    if (interest_points[i].x + vol_x_size/2 > st_max_x  || interest_points[i].x - vol_x_size/2 < 0 ){
      // std::cout << "outise x" << '\n';
      continue;
    }
    if (interest_points[i].y + vol_y_size/2 > st_max_y  || interest_points[i].y - vol_y_size/2 < 0 ){
      // std::cout << "outside y" << '\n';
      continue;
    }
    if (interest_points[i].t + vol_t_size/2 > st_max_t  || interest_points[i].t - vol_t_size/2 < 0 ){
      // std::cout << "outside t" << '\n';
      continue;
    }


    // std::cout << "points is insize the valid area" << '\n';

    //make a vol of histograms. Volume will have dimensions 3x3x2, each element inside it being a histogrm
    utils::Array<Histogram, 3> hist_hog_vol;
    size_t size_hog_vol [3]= { cell_per_vol_t, cell_per_vol_y, cell_per_vol_x }; // Array dimensions
    hist_hog_vol.resize(size_hog_vol,Histogram(nbins_hog, hist_range));


    //HOG
    //loop through all the pixels in the neighbourhoos and add the vlaues into the corresponding histogram
    for (size_t p_x = interest_points[i].x - vol_x_size/2; p_x < interest_points[i].x + vol_x_size/2; p_x++) {
      for (size_t p_y = interest_points[i].y - vol_y_size/2; p_y < interest_points[i].y + vol_y_size/2; p_y++) {
        for (size_t p_t = interest_points[i].t - vol_t_size/2; p_t < interest_points[i].t + vol_t_size/2; p_t++) {

          //get which cell of the 3x3x2 volume does this pixel belong to
          int cell_idx_x= (p_x- (interest_points[i].x - vol_x_size/2 ) )/cell_size_x;
          int cell_idx_y= (p_y- (interest_points[i].y - vol_y_size/2 ) )/cell_size_y;
          int cell_idx_t= (p_t- (interest_points[i].t - vol_t_size/2 ) )/cell_size_t;

          // std::cout << "accesing cell " << cell_idx_t << " " << cell_idx_y << " " << cell_idx_x << '\n';

          float mag=grad_mags[p_t].at<float>(p_y,p_x);
          float orientation=grad_orientations[p_t].at<float>(p_y,p_x);

          //make the orientation the same for oposite directions
          orientation = fmod(orientation,180);

          //add the value to the corresponding histogram
          hist_hog_vol[cell_idx_t][cell_idx_y][cell_idx_x].add_val(orientation,mag);


        }
      }
    }



    //HOF
    //hof for normal flow
    utils::Array<Histogram, 3> hist_hof_vol;
    size_t size_hof_vol [3]= { cell_per_vol_t, cell_per_vol_y, cell_per_vol_x }; // Array dimensions
    hist_hof_vol.resize(size_hof_vol,Histogram(nbins_hog, hist_range));



    //hof with 1 bin for the bin that has low magnitude
    utils::Array<Histogram, 3> hist_hof_low_mag_vol;
    hist_hof_low_mag_vol.resize(size_hof_vol,Histogram(1, 180.0f));


    float mag_thresh_low=1e-8;


    for (size_t p_x = interest_points[i].x - vol_x_size/2; p_x < interest_points[i].x + vol_x_size/2; p_x++) {
      for (size_t p_y = interest_points[i].y - vol_y_size/2; p_y < interest_points[i].y + vol_y_size/2; p_y++) {
        for (size_t p_t = interest_points[i].t - vol_t_size/2; p_t < interest_points[i].t + vol_t_size/2; p_t++) {

          int cell_idx_x= (p_x- (interest_points[i].x - vol_x_size/2 ) )/cell_size_x;
          int cell_idx_y= (p_y- (interest_points[i].y - vol_y_size/2 ) )/cell_size_y;
          int cell_idx_t= (p_t- (interest_points[i].t - vol_t_size/2 ) )/cell_size_t;

          float mag=flow_mags[p_t].at<float>(p_y,p_x);
          float orientation=flow_orientations[p_t].at<float>(p_y,p_x);

          //make the orientation the same for oopsite directions
          orientation = fmod(orientation,180);

          if (mag<mag_thresh_low) {
            hist_hof_low_mag_vol[cell_idx_t][cell_idx_y][cell_idx_x].add_val(orientation,mag);
          }else{
            hist_hof_vol[cell_idx_t][cell_idx_y][cell_idx_x].add_val(orientation,mag);

          }


        }
      }
    }



    //concatenate the the low threshold with the high threshold HOF
    for (size_t t = 0; t < cell_per_vol_t; t++) {
      for (size_t y = 0; y < cell_per_vol_y; y++) {
        for (size_t x = 0; x < cell_per_vol_x; x++) {
          hist_hof_vol[t][y][x].concatenate(hist_hof_low_mag_vol[t][y][x]);
        }
      }
    }

    Histogram hof_full;
    for(auto hist: hist_hof_vol){
      hof_full.concatenate(hist);
    }
    hof_full.normalize();

    Histogram hog_full;
    for(auto hist: hist_hog_vol){
      hog_full.concatenate(hist);
    }
    hog_full.normalize();

    Histogram hof_hog;
    hof_hog.concatenate(hog_full);
    hof_hog.concatenate(hof_full);

    // interest_points[i].descriptor=hof_hog.descriptor();
    interest_points[i].descriptor=hof_hog;

  }

}

int Stip::write_descriptors_to_file(std::vector<interest_point> interest_points, std::ofstream& file){

  int descriptor_written=0;
  for (size_t i = 0; i < interest_points.size(); i++) {
    //if the descriptor is not yet initialized it means it's empty
    if (!interest_points[i].descriptor.to_string().empty()  ){
      // std::cout << "string desc:: " << interest_points[i].descriptor.to_string() << '\n';
      file << interest_points[i].descriptor.to_string() << std::endl;
      descriptor_written++;
    }
  }

  return descriptor_written;


}

void Stip::read_features_from_file(std::string descriptor_file_path, Math::Matrix<Float>& features){


  std::ifstream desc_file( descriptor_file_path );


  int nr_vectors=0;
  int vector_dimensions=0;
  int nr_videos=0;
  std::string line;
  getline( desc_file, line );
  std::istringstream buf(line);
  std::istream_iterator<std::string> beg(buf), end;
  std::vector<std::string> tokens(beg, end); // done!

  nr_vectors=atoi(tokens[0].data());
  vector_dimensions=atoi(tokens[1].data());
  nr_videos=atoi(tokens[2].data());

  // std::cout << "nr_vectors" << nr_vectors << '\n';
  // std::cout << "vector_dimensions" << vector_dimensions << '\n';
  // std::cout << "nr_videos" << nr_videos << '\n';

  features.resize(vector_dimensions,nr_vectors);

  int sample=0;
  while( getline( desc_file, line ) ){

    if (line=="#")
      continue;


    std::istringstream buf(line);
    std::istream_iterator<std::string> beg(buf), end;
    std::vector<std::string> tokens(beg, end); // done!

    for (size_t i = 0; i < tokens.size(); i++) {
      features.at(i,sample)=atof(tokens[i].data());

    }

    sample++;
  }
  desc_file.close();

}

void Stip::read_features_per_video_from_file(std::string descriptor_file_path, std::vector<Math::Matrix<Float> >& features_per_video, int max_nr_videos){
  std::ifstream desc_file( descriptor_file_path );


  int nr_vectors=0;
  int vector_dimensions=0;
  int nr_videos=0;
  std::string line;
  getline( desc_file, line );
  std::istringstream buf(line);
  std::istream_iterator<std::string> beg(buf), end;
  std::vector<std::string> tokens(beg, end); // done!

  nr_vectors=atoi(tokens[0].data());
  vector_dimensions=atoi(tokens[1].data());
  nr_videos=atoi(tokens[2].data());

  std::cout << "nr_vectors" << nr_vectors << '\n';
  std::cout << "vector_dimensions" << vector_dimensions << '\n';
  std::cout << "nr_videos" << nr_videos << '\n';

  features_per_video.resize(nr_videos);

  std::vector<std::vector<float>>features_video;

  // int nr_features_in_video=0;
  int video_nr=0;
  while( getline( desc_file, line ) ){

    if (line=="#"){
      if (!features_video.empty()){
        features_per_video[video_nr].resize(vector_dimensions,features_video.size());

        //get the features_video and put them into the features__video_vector of math::matrices
        for (size_t i = 0; i < vector_dimensions; i++) {
          for (size_t j = 0; j < features_video.size(); j++) {
            features_per_video[video_nr].at(i,j) = features_video[j][i];
          }
        }

        features_video.clear();
        video_nr++;
      }
    }

    if (video_nr>max_nr_videos){
      break;
    }


    std::istringstream buf(line);
    std::istream_iterator<std::string> beg(buf), end;
    std::vector<std::string> tokens(beg, end); // done!

    std::vector<float> tmp;
    for (size_t i = 0; i < tokens.size(); i++) {
      tmp.push_back(atof(tokens[i].data()));
    }
    features_video.push_back(tmp);
  }
  desc_file.close();
}
