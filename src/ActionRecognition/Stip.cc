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
  //TODO read the parameters from a config file

  //for dummy video
  // m_sigma_local_spacial=1.0f;
	// m_sigma_local_temporal=1.0f;

  //0.3 for sobel 0.9 for sepderivatives
  //works well with derivative Lt being just the difference
  // m_sigma_local_spacial=0.3f;
  // m_sigma_local_temporal=0.9f;



  //works well with derivative Lt being the sobel
  m_sigma_local_spacial=1.0f;
  m_sigma_local_temporal=1.5f;



  //Alexander Richard's settings
  m_sigma_local_spacial=1.0f;
  m_sigma_local_temporal=2.0f;
}

// empty destructor
Stip::~Stip()
{}

void Stip::run(){
  std::cout << "run stip" << '\n';

  // m_video_path= "../experiments/videos/dummy.avi";
  // m_video_path= "../experiments/videos/LONGESTYARD_walk_f_nm_np1_fr_med_6.avi";
  m_video_path= "../experiments/videos/Torwarttraining_2_(_sterreich)_catch_f_cm_np1_ba_goo_1.avi";
  // m_video_path= "../experiments/videos/H_I_I_T__Swamis_stairs_with_Max_Wettstein_featuring_Donna_Wettstein_climb_stairs_f_cm_np1_ba_med_4.avi";
  // m_video_path= "../experiments/videos/likebeckam_run_f_cm_np1_fr_med_5.avi";
  // m_video_path = "../experiments/videos/Veoh_Alpha_Dog_1_walk_f_nm_np1_ri_med_24.avi";
  // m_video_path= "../experiments/videos/Ballfangen_catch_u_cm_np1_fr_goo_0.avi";
  // m_video_path= "../experiments/videos/Clay_sBasketballSkillz_shoot_ball_f_nm_np1_ba_med_7.avi";

  std::vector<cv::Mat> Lx,Ly,Lt;
  std::vector<cv::Mat> Lx2,Ly2,Lt2,LxLy,LxLt,LyLt;
  std::vector<cv::Mat> Lx2_s,Ly2_s,Lt2_s,LxLy_s,LxLt_s,LyLt_s;
  std::vector<cv::Mat> harris_responses;
  std::vector<interest_point> interest_points;

  create_spacial_temporal_volume(m_st_volume,m_video_path);
  gaussian_smooth(m_st_volume, m_st_volume, m_sigma_local_spacial, m_sigma_local_temporal);
  compute_derivatives(m_st_volume, Lx,Ly,Lt);
  compute_harris_coefficients(Lx,Ly,Lt, Lx2,Ly2,Lt2,LxLy,LxLt,LyLt);

  //for  Torwarttraining_2_(_sterreich)_catch_f_cm_np1_ba_goo_1
  float sigma_integration_spacial=5.0f;
  float sigma_integration_temporal=2.0f;

  for (size_t i = 1; i < 7; i++) {
    for (size_t j = 1; j < 3; j++) {
      sigma_integration_spacial=std::pow(2,(1+i)/2.0f);
      sigma_integration_temporal=std::pow(2,j/2.0f);


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


      std::cout << "nr of interest points" << interest_points.size() << '\n';

    }
  }

  std::cout << "before nms " << interest_points.size() << '\n';
  non_max_supress(interest_points, m_st_volume.size(), m_st_volume[0].rows, m_st_volume[0].cols);
  non_max_supress(interest_points, m_st_volume.size(), m_st_volume[0].rows, m_st_volume[0].cols);
  std::cout << "aftter nms " << interest_points.size() << '\n';



  //---------------DESCRIPTORS---------------------------
  std::vector<cv::Mat> grad_mags;
  std::vector<cv::Mat> grad_orientations;
  std::vector<cv::Mat> flow_mags;
  std::vector<cv::Mat> flow_orientations;


  //TODO Use Scharr insted of sobel void Scharr(InputArray src, OutputArray dst, int ddepth, int dx, int dy, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )

  //TODO
  /*
  Mat grad_x, grad_y;
    Scharr(gray, grad_x, CV_32FC1, 1, 0);
    Scharr(gray, grad_y, CV_32FC1, 0, 1);
    // 4. calculate gradient magnitude and direction
    Mat magnitude, direction;
    bool useDegree = true;    // use degree or rad
    // the range of the direction is [0,2pi) or [0, 360)
    cartToPolar(grad_x, grad_y, magnitude, direction, useDegree);*/


  //TODO
  /*
  A Mapping of Type to Numbers in OpenCV

 C1	C2	C3	C4
CV_8U	0	8	16	24
CV_8S	1	9	17	25
CV_16U	2	10	18	26
CV_16S	3	11	19	27
CV_32S	4	12	20	28
CV_32F	5	13	21	29
CV_64F	6	14	22	30*/

  //TODO interpolation
  /*
  H(x1) =h(x1) +w*(1- (x-x1)/b  )
  H(x2) =h(x1) +w*((x-x1)/b  )*/




  compute_grad_orientations_magnitudes(Lx,Ly, grad_mags, grad_orientations );
  compute_flow(m_st_volume, flow_mags, flow_orientations );
  compute_descriptors(interest_points, grad_mags, grad_orientations, flow_mags, flow_orientations);








  // // //Show volume
  // for (size_t i = 0; i < harris_responses.size(); i++) {
  //   cv::imshow("window", mat2gray(harris_responses[i]));
  //   cv::waitKey(0);
  //   char key = cvWaitKey(30);
  //       if (key == 27) // ESC
  //           break;
  // }


  // // //Show volume with interest_points
  // while (true)
  // for (size_t i = 0; i < harris_responses.size(); i++) {
  //   // double min, max;
  //   // cv::minMaxLoc(harris_responses[i], &min, &max);
  //   // std::cout << "min max is " << min << " " << max << '\n';
  //   cv::Mat frame;
  //   frame=mat2gray(m_st_volume[i]);
  //   cv::cvtColor(frame, frame, CV_GRAY2BGR);
  //   draw_interest_points(frame,interest_points,i);
  //   cv::imshow("window", frame);
  //   // cv::waitKey(0);
  //   char key = cvWaitKey(50);
  //       if (key == 27) // ESC
  //           break;
  // }


  // // //Show blankc with interest_points
  // while (true)
  // for (size_t i = 0; i < harris_responses.size(); i++) {
  //   double min, max;
  //   cv::minMaxLoc(harris_responses[i], &min, &max);
  //   std::cout << "min max is " << min << " " << max << '\n';
  //   cv::Mat blank= cv::Mat::zeros(m_st_volume[i].rows,m_st_volume[i].cols,CV_32F);
  //   blank=mat2gray(blank);
  //   cv::cvtColor(blank, blank, CV_GRAY2BGR);
  //   draw_interest_points(blank,interest_points,i);
  //   cv::imshow("window", blank);
  //   // cv::waitKey(0);
  //   char key = cvWaitKey(50);
  //       if (key == 27) // ESC
  //           break;
  // }






  //Read video  (reads the whole video in a spacial-temporal_volume)
  //gausian smoth the whole volume
  /*Compute derivatives
    out: Lx volume containing the derivative in x for all pixels
         Ly volume containing the derivative in y for all pixels
         Lt volume containing the derivative in t for all pixels
   )*/
   /*compute the derivatives squared and so on (eg: Lx^2 will be again a big tensor given by the elemt wise multiplication fo Lx and Lx)
    out: 6 big tensors corresponding to Lx^2, LxLy, LxLt, Ly^2, LyLt, Lt^2
   */

   //for each spacial and temporal scale
    //smooth the 6 big tensors using those scales
    //get harris responsed in the whole volume (out: harris_responses volume)
    //get local maxima in the volume




   //find interest points in the volume by going through all the pixels



   // //Show volume
   // for (size_t i = 0; i < m_st_volume.size(); i++) {
   //   cv::imshow("window", m_st_volume[i]);
   //   char key = cvWaitKey(30);
   //       if (key == 27) // ESC
   //           break;
   // }


  // //read the file
  // std::string video_path= "../experiments/videos/dummy.avi";
  // cv::VideoCapture cap(video_path);
  // if (!cap.isOpened()){
  //     std::cout << "!!! Failed to open file: " << video_path << std::endl;
  //     return ;
  // }
  //
  //   cv::Mat frame;
  //   for(;;)
  //   {
  //     if (!cap.read(frame))
  //         break;
  //
  //     process_frame(frame);
  //     //st_buffer.add_frame(frame);
  //
  //
  //     cv::imshow("window", frame);
  //
  //     char key = cvWaitKey(30);
  //     if (key == 27) // ESC
  //         break;
  //   }


}

cv::Mat Stip::mat2gray(const cv::Mat& src){
    cv::Mat dst;
    cv::normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

    return dst;
}

void Stip::create_spacial_temporal_volume(std::vector<cv::Mat>& m_st_volume, std::string m_video_path){
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

    std::cout << "type of frame is "<< frame.type() << '\n';
    // double min, max;
    // cv::minMaxLoc(frame, &min, &max);
    // std::cout << "min max is " << min << " " << max << '\n';


    m_st_volume.push_back(frame.clone()); //need to clone it otherwise all frames will be the same

    // cv::imshow("window", frame);
    // char key = cvWaitKey(30);
    //     if (key == 27) // ESC
    //         break;
  }
  std::cout << "std_vilume has size " << m_st_volume.size() << '\n';
}

void Stip::gaussian_smooth(std::vector<cv::Mat> vol_in, std::vector<cv::Mat>& vol_out, float m_sigma_local_spacial, float m_sigma_local_temporal){
  std::cout << "gaussian smooth" << '\n';

  //get kernels
  int kernel_size=15;
  // cv::Mat ker_spacial = cv::getGaussianKernel(kernel_size, m_sigma_local_spacial, CV_32F );
  cv::Mat ker_temporal = cv::getGaussianKernel(kernel_size, m_sigma_local_temporal, CV_32F );
  cv::Mat ker_dummy = cv::Mat::zeros( kernel_size, 1, CV_32F );
  ker_dummy.at<float>(kernel_size/2)=1;
  // std::cout << "" << '\n';


  //copy the in volume into another on so we can operate on it
  // std::vector<cv::Mat> vol_in


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
    sepFilter2D(slice, slice, slice.depth(), ker_temporal, ker_dummy );

    // //show slice
    // cv::imshow("window", slice);
    // cv::waitKey(0);
    // char key = cvWaitKey(30);
    //     if (key == 27) // ESC
    //         break;



    //copy it back
    for (size_t y = 0; y < m_st_volume[0].rows; y++) {
      for (size_t t = 0; t < m_st_volume.size(); t++) {
        vol_out[t].at<float>(y,x)=slice.at<float>(y,t);
      }
    }
  }

}

void Stip::compute_derivatives(std::vector<cv::Mat>& m_st_volume, std::vector<cv::Mat>& Lx, std::vector<cv::Mat>& Ly, std::vector<cv::Mat>& Lt){

  //derivatives in x and y
  cv::Mat Ix,Iy,It;
  for (size_t i = 0; i < m_st_volume.size(); i++) {
    cv::Sobel( m_st_volume[i], Ix, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel( m_st_volume[i], Iy, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

    // Scharr(m_st_volume[i], Ix, CV_32FC1, 1, 0);
    // Scharr(m_st_volume[i], Iy, CV_32FC1, 0, 1);

    // //Maybe convolving with my own kernel could be better because then the derivatives will have the same range
    // cv::Mat ker_sobel = cv::Mat::zeros( 3, 1, CV_32F );
    // ker_sobel.at<float>(0)=-1;
    // ker_sobel.at<float>(1)=0;
    // ker_sobel.at<float>(2)=1;
    // cv::Mat ker_dummy = cv::Mat::zeros( 3, 1, CV_32F );
    // ker_dummy.at<float>(3/2)=1;
    // sepFilter2D(m_st_volume[i], Ix, CV_32F, ker_sobel, ker_dummy );
    // sepFilter2D(m_st_volume[i], Iy, CV_32F, ker_dummy, ker_sobel );


    Lx.push_back(Ix.clone());
    Ly.push_back(Iy.clone());
  }

  // //Temporal derivatives
  // for (size_t i = 0; i < m_st_volume.size()-1; i++) {
  //   It = (m_st_volume[i+1] - m_st_volume[i]);
  //   Lt.push_back(It.clone());
  // }

  //Second way of doing temporal derivatives by doing a sobel in a y-t slice
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
    // Scharr(slice, slice, CV_32FC1, 1, 0);


    // //show slice
    // cv::imshow("window", mat2gray(slice));
    // cv::waitKey(0);
    // char key = cvWaitKey(30);
    //     if (key == 27) // ESC
    //         break;
    //


    //copy it back
    for (size_t y = 0; y < m_st_volume[0].rows; y++) {
      for (size_t t = 0; t < m_st_volume.size(); t++) {
        Lt[t].at<float>(y,x)=slice.at<float>(y,t);
      }
    }
    // Lt.push_back(It.clone());
  }



  //last frame doesnt have a next one to compute the spacial derivative so we just grab the first one
  It = (m_st_volume[m_st_volume.size()-1] - m_st_volume[0]);
  Lt.push_back(It.clone());


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

   //Not needed if you do it in a vectorized
  //  for (size_t i = 0; i < harris_responses.size(); i++) {
  //    harris_responses[i]=cv::Mat(Lx2[0].rows,Lx2[0].cols,Lx2[0].depth());
  //  }



   float k=0.005;
  //  cv::Mat tmp1,tmp2;
   cv::Mat tmp1=cv::Mat::zeros(Lx2[0].rows,Lx2[0].cols,CV_32F);
   cv::Mat tmp2=cv::Mat::zeros(Lx2[0].rows,Lx2[0].cols,CV_32F);

  //  for (size_t t = 0; t < Lx2.size(); t++) {
  //    for (size_t y = 0; y < Lx2[0].rows; y++) {
  //      for (size_t x = 0; x < Lx2[0].cols; x++) {
   //
  //        cv::Mat harris_matrix(3,3,CV_32F);
   //
  //        harris_matrix.at<float>(0,0)=Lx2[t].at<float>(y,x);
  //        harris_matrix.at<float>(0,1)=LxLy[t].at<float>(y,x);
  //        harris_matrix.at<float>(0,2)=LxLt[t].at<float>(y,x);
   //
  //        harris_matrix.at<float>(1,0)=LxLy[t].at<float>(y,x);
  //        harris_matrix.at<float>(1,1)=Ly2[t].at<float>(y,x);
  //        harris_matrix.at<float>(1,2)=LyLt[t].at<float>(y,x);
   //
  //        harris_matrix.at<float>(2,0)=LxLt[t].at<float>(y,x);
  //        harris_matrix.at<float>(2,1)=LyLt[t].at<float>(y,x);
  //        harris_matrix.at<float>(2,2)=Lt2[t].at<float>(y,x);
   //
  //        float determinant= cv::determinant(harris_matrix);
   //
  //        float trace=harris_matrix.at<float>(0,0) + harris_matrix.at<float>(1,1) + harris_matrix.at<float>(2,2);
  //        float response= determinant -k* std::pow(trace,3);
   //
  //        harris_responses[t].at<float>(y,x)= response;
   //
  //      }
  //    }
  //  }



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

  // int max_surpassed=0;
  // float thresh=8000000.0f;
  float thresh=0.0f;

  for (int t = 0; t < harris_responses.size(); t++) {
    for (int y = 0; y < harris_responses[0].rows; y++) {
      for (int x = 0; x < harris_responses[0].cols; x++) {

        // std::cout << "computing points " << t << " " << y << " " << x << '\n';

        int neighbours_surpassed=0;
        //for points at position x,y,t check all the neighbours around it and if it is bigger than all of them then its an interest point
        for (int n_t = std::max(0,t-1); n_t < std::min((int)harris_responses.size(),t+2); n_t++) {
          for (int n_y = std::max(0,y-1); n_y < std::min(harris_responses[0].rows,y+2); n_y++){
            for (int n_x = std::max(0,x-1); n_x < std::min(harris_responses[0].cols,x+2); n_x++){

              // std::cout << "computing neighbour " << n_t << " " << n_y << " " << n_x << '\n';

              if (n_t==t && n_y==y && n_x==x  ){
                continue;
              }

              //detect local POSITIVE maxima
              // if (harris_responses[t].at<float>(y,x)<=1e-7){
              //   continue;
              // }

              // if (harris_responses[t].at<float>(y,x)<=1e-8){
              //   continue;
              // }

              //with sobel it works pretty good
              if (harris_responses[t].at<float>(y,x)<=1e-4){
                continue;
              }


              if (harris_responses[t].at<float>(y,x) > harris_responses[n_t].at<float>(n_y,n_x)){
                neighbours_surpassed++;

              }

            }
          }
        }

        // std::cout << "neighbours_surpassed " << neighbours_surpassed << '\n';
        // harris_responses[t].at<float>(y,x)=0;

        if (neighbours_surpassed==26){
          interest_point int_p;
          int_p.t=t;
          int_p.y=y;
          int_p.x=x;
          interest_points.push_back(int_p);
          // harris_responses[t].at<float>(y,x)=neighbours_surpassed;
        }

      }
    }
  }

  // std::cout << "max surpassed " << max_surpassed << '\n';

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

  std::cout << "drawn ips " << drawn_ips << '\n';


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
    // std::cout << "before rects " << ip_time.size() << '\n';
    nms(ip_time,ip_nms,0);
    // std::cout << "after rects " << ip_nms.size() << '\n';

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
    // std::cout << "before rects " << ip_time.size() << '\n';
    nms(ip_x,ip_nms,0);
    // std::cout << "after rects " << ip_nms.size() << '\n';

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


  //TODO put the k back to 9
  int k=2;
  int nbins_hog=8;
  float hist_range=360.0f;

  //TODO read these values out of a struct called ST_vol or something similar
  int st_max_x=grad_orientations[0].cols;
  int st_max_y=grad_orientations[0].rows;
  int st_max_t=grad_orientations.size();

  std::cout << "st size is " << st_max_x << " " << st_max_y << " " << st_max_t << '\n';

  int cell_per_vol_x=3;
  int cell_per_vol_y=3;
  int cell_per_vol_t=2;


  for (size_t i = 0; i < interest_points.size(); i++) {


    //TODO I'm not sure if they are sigmas of standard deviations
    int vol_x_size=2*k*interest_points[i].sigma_spacial;
    int vol_y_size=2*k*interest_points[i].sigma_spacial;
    int vol_t_size=2*k*interest_points[i].sigma_temporal;

    int cell_size_x=std::ceil(vol_x_size/(float)cell_per_vol_x);
    int cell_size_y=std::ceil(vol_y_size/(float)cell_per_vol_y);
    int cell_size_t=std::ceil(vol_t_size/(float)cell_per_vol_t);

    std::cout << "volume size " << vol_x_size << " " << vol_y_size << " " << vol_t_size << '\n';
    std::cout << "cell size " << cell_size_x << " " << cell_size_y << " " << cell_size_t << '\n';


    //Borders
    if (interest_points[i].x + vol_x_size/2 > st_max_x  || interest_points[i].x - vol_x_size/2 < 0 ){
      std::cout << "outise x" << '\n';
      continue;
    }
    if (interest_points[i].y + vol_y_size/2 > st_max_y  || interest_points[i].y - vol_y_size/2 < 0 ){
      std::cout << "outside y" << '\n';
      continue;
    }
    if (interest_points[i].t + vol_t_size/2 > st_max_t  || interest_points[i].t - vol_t_size/2 < 0 ){
      std::cout << "outside t" << '\n';
      continue;
    }


    std::cout << "points is insize the valid area" << '\n';

    //make a vol of histograms
    //TODO volume should be a templated class which will be internally represented as a 3-times nested std::vector
    // std::vector<std::vector<std::vector<Histogram > > > hist_hog_vol;
    //
    // hist_hog_vol.resize(cell_per_vol_t);
    // for (size_t t = 0; t < cell_per_vol_t; t++) {
    //   hist_hog_vol[t].resize(cell_per_vol_y);
    //   for (size_t y = 0; y < cell_per_vol_y; y++) {
    //     hist_hog_vol[t][y].resize(cell_per_vol_x);
    //
    //     for (size_t x = 0; x < cell_per_vol_x; x++) {
    //       // std::cout << "created cell " << t << " " << y << " " << x  << '\n';
    //       hist_hog_vol[t][y][x]=Histogram(nbins_hog, hist_range);
    //     }
    //
    //   }
    // }
    utils::Array<Histogram, 3> hist_hog_vol;
    size_t size_hog_vol [3]= { cell_per_vol_t, cell_per_vol_y, cell_per_vol_x }; // Array dimensions
    hist_hog_vol.resize(size_hog_vol,Histogram(nbins_hog, hist_range));         // Can change array size any time



    

    //HOG
    //loop through all the pixels in the neighbourhoos and add the vlaues into the corresponding histogram
    for (size_t p_x = interest_points[i].x - vol_x_size/2; p_x < interest_points[i].x + vol_x_size/2; p_x++) {
      for (size_t p_y = interest_points[i].y - vol_y_size/2; p_y < interest_points[i].y + vol_y_size/2; p_y++) {
        for (size_t p_t = interest_points[i].t - vol_t_size/2; p_t < interest_points[i].t + vol_t_size/2; p_t++) {

          int cell_idx_x= (p_x- (interest_points[i].x - vol_x_size/2 ) )/cell_size_x;
          int cell_idx_y= (p_y- (interest_points[i].y - vol_y_size/2 ) )/cell_size_y;
          int cell_idx_t= (p_t- (interest_points[i].t - vol_t_size/2 ) )/cell_size_t;

          // std::cout << "cell ids are" << cell_idx_t << " " << cell_idx_y << " " << cell_idx_t << '\n';

          float mag=grad_mags[p_t].at<float>(p_y,p_x);
          float orientation=grad_orientations[p_t].at<float>(p_y,p_x);

          std::cout << "accesing cell " << cell_idx_t << " " << cell_idx_y << " " << cell_idx_x << '\n';

          hist_hog_vol[cell_idx_t][cell_idx_y][cell_idx_x].add_val(orientation,mag);


        }
      }
    }



    //HOF

    //hof for normal flow
    std::vector<std::vector<std::vector<Histogram > > > hist_hof_vol;
    hist_hof_vol.resize(cell_per_vol_t);
    for (size_t t = 0; t < cell_per_vol_t; t++) {
      hist_hof_vol[t].resize(cell_per_vol_y);
      for (size_t y = 0; y < cell_per_vol_y; y++) {
        hist_hof_vol[t][y].resize(cell_per_vol_x);

        for (size_t x = 0; x < cell_per_vol_x; x++) {
          hist_hof_vol[t][y][x]=Histogram(nbins_hog, hist_range);
        }

      }
    }

    //hof with 1 bin for the bin that has low magnitude
    std::vector<std::vector<std::vector<Histogram > > > hist_hof_low_mag_vol;
    hist_hof_low_mag_vol.resize(cell_per_vol_t);
    for (size_t t = 0; t < cell_per_vol_t; t++) {
      hist_hof_low_mag_vol[t].resize(cell_per_vol_y);
      for (size_t y = 0; y < cell_per_vol_y; y++) {
        hist_hof_low_mag_vol[t][y].resize(cell_per_vol_x);

        for (size_t x = 0; x < cell_per_vol_x; x++) {
          hist_hof_low_mag_vol[t][y][x]=Histogram(1, hist_range);
        }

      }
    }


    float mag_thresh_low=0.1f;


    for (size_t p_x = interest_points[i].x - vol_x_size/2; p_x < interest_points[i].x + vol_x_size/2; p_x++) {
      for (size_t p_y = interest_points[i].y - vol_y_size/2; p_y < interest_points[i].y + vol_y_size/2; p_y++) {
        for (size_t p_t = interest_points[i].t - vol_t_size/2; p_t < interest_points[i].t + vol_t_size/2; p_t++) {

          int cell_idx_x= (p_x- (interest_points[i].x - vol_x_size/2 ) )/cell_size_x;
          int cell_idx_y= (p_y- (interest_points[i].y - vol_y_size/2 ) )/cell_size_y;
          int cell_idx_t= (p_t- (interest_points[i].t - vol_t_size/2 ) )/cell_size_t;

          // std::cout << "cell ids are" << cell_idx_t << " " << cell_idx_y << " " << cell_idx_t << '\n';

          float mag=flow_mags[p_t].at<float>(p_y,p_x);
          float orientation=flow_orientations[p_t].at<float>(p_y,p_x);

          if (mag<mag_thresh_low) {
            hist_hof_low_mag_vol[cell_idx_t][cell_idx_y][cell_idx_x].add_val(orientation,mag);
          }else{
            hist_hof_vol[cell_idx_t][cell_idx_y][cell_idx_x].add_val(orientation,mag);

          }



        }
      }
    }


    //TODO concatenate the two histogram volumes cell by cell and therefore still end up with a volume of 3x3x2



  }

}
