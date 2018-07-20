
#ifndef LIFGROUP_H
#define LIFGROUP_H

#include "defines.h"

namespace blas = boost::numeric::ublas;

class LIFGroup {
  public:

  uint32_t N;
  float adapt;
  float tau;
  blas::matrix<float> theta; 
  float vthr;
  float vrest; 
  float vreset; 
  float refrac_per; 
  float i_offset;
  float tc_theta; 
  float theta_plus_e; 
  float ge_tau;
  float gi_tau; 
  
  blas::matrix<float> ge;
  blas::matrix<float> gi;
  blas::matrix<float> v;
  blas::matrix<float> last_spk;

  LIFGroup(uint32_t N, 
           float adapt, 
           float tau, 
           blas::matrix<float> theta, 
           float vthr, 
           float vrest, 
           float vreset, 
           float refrac_per, 
           float i_offset, 
           float tc_theta, 
           float theta_plus_e);
           
  blas::matrix<float> step(float t, float dt, blas::matrix<float> Iine, blas::matrix<float> Iini);
  blas::matrix<float> step(float t, float dt, blas::matrix<float> Iine);
  void reset();
};

#endif
