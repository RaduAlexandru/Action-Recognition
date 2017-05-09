#include "Histogram.hh"

Histogram::Histogram(){

}

Histogram::Histogram(int nbins, float range){
  // std::cout << "hist constructor" << '\n';
  m_hist.resize(nbins);
  m_range=range;
  m_bin_size=range/nbins;
  // std::cout << "m_bins size is" << m_bin_size << '\n';

}

void Histogram::init(int nbins, float range){
  m_hist.resize(nbins);
  m_range=range;
  m_bin_size=range/nbins;
}
void Histogram::add_val(float val, float weight){

  //each bin has a center located at (bin_size/2)*i where i is has range 1---nbins

  // std::cout << "add_val: " << val << " weight: " << weight << '\n';
  // std::cout << "m_bin_size: " << m_bin_size << '\n';

  int closest_bin= val/m_bin_size;
  // std::cout << "closest bin is: " << closest_bin << '\n';

  //decide if it is on the right or left of the center
  float center_closest=(m_bin_size/2.0)*closest_bin;
  int second_closest=-1;
  if (val<center_closest){  //seond closest is to the left
    if (closest_bin==0)
      second_closest=m_hist.size()-1;
    else
      second_closest=closest_bin-1;
  }else{  //second closest is to the right
    if (closest_bin==(m_hist.size()-1))
      second_closest=0;
    else
      second_closest=closest_bin+1;
  }
  // std::cout << "second closest " << second_closest << '\n';

  // //add values to the closest and second closest bins
  m_hist[closest_bin]= m_hist[closest_bin] + weight* (1 - (val- center_closest)/m_bin_size);
  m_hist[second_closest]= m_hist[closest_bin] + weight* ((val- center_closest)/m_bin_size);


}


void Histogram::normalize(){

  //TODO not sure if correct
  //calculate norm
  float norm=0.0;
  for (size_t i = 0; i < m_hist.size(); i++) {
    norm+=  m_hist[i]*m_hist[i];  //  norm
    // norm+=  m_hist[i]; //L1
  }
  norm=std::sqrt(norm);

  std::cout << " norm is " << norm << '\n';

  //normalize
  for (size_t i = 0; i < m_hist.size(); i++) {
    m_hist[i]=m_hist[i]/norm;
  }



}
