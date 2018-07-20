
#ifndef SYNAPSEGROUP_H
#define SYNAPSEGROUP_H

#include "defines.h"

namespace blas = boost::numeric::ublas;

class SynapseGroup {
  public:
  
  uint32_t N;
  uint32_t M;
  blas::matrix<float> w;
  bool stdp;
  float tc_pre_ee;
  float tc_post_1_ee; 
  float tc_post_2_ee; 
  float nu_ee_pre;
  float nu_ee_post; 
  float wmax_ee;
  
  blas::matrix<float> last_pre;
  blas::matrix<float> last_post;
  
  SynapseGroup(uint32_t N, 
               uint32_t M, 
               blas::matrix<float> w, 
               bool stdp, 
               float tc_pre_ee, 
               float tc_post_1_ee, 
               float tc_post_2_ee, 
               float nu_ee_pre, 
               float nu_ee_post, 
               float wmax_ee);
               
  blas::matrix<float> step(float t, float dt, blas::matrix<bool> pre_spk, blas::matrix<bool> post_spk);
  void reset();
};

#endif
