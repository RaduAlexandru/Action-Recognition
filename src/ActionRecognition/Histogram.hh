#ifndef HISTOGRAM_HH_
#define HISTOGRAM_HH_

#include <vector>
#include <iostream>
#include <algorithm>


class Histogram
{
public:
  Histogram();
  Histogram(int nbins, float range);
  void init(int nbins, float range);
  void add_val(float val, float weight);
  void normalize();

private:
  float m_range;
  float m_bin_size;
  std::vector<float> m_hist;

};


#endif
