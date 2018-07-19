
#include "SynapseGroup.h"

using namespace std;
using namespace boost::numeric::ublas;

SynapseGroup::SynapseGroup(uint32_t N, 
                           uint32_t M, 
                           matrix<float> w, 
                           bool stdp, 
                           float tc_pre_ee, 
                           float tc_post_1_ee, 
                           float tc_post_2_ee, 
                           float nu_ee_pre, 
                           float nu_ee_post, 
                           float wmax_ee) {
  this->N = N;
  this->M = M;
  this->w = w;
  this->stdp = stdp;
  this->tc_pre_ee = tc_pre_ee;
  this->tc_post_1_ee = tc_post_1_ee;
  this->tc_post_2_ee = tc_post_2_ee;
  this->nu_ee_pre = nu_ee_pre;
  this->nu_ee_post = nu_ee_post;
  this->wmax_ee = wmax_ee;
}
               
matrix<float> step(float t, float dt, matrix<bool> pre_spk, matrix<bool> post_spk)
{
}

void reset()
{
}
