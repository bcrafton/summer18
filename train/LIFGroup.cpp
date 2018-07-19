
#include "LIFGroup.h"

using namespace std;
using namespace boost::numeric::ublas;

LIFGroup::LIFGroup(uint32_t N, 
                   float adapt, 
                   float tau, 
                   float theta, 
                   float vthr, 
                   float vrest, 
                   float vreset, 
                   float refrac_per, 
                   float i_offset, 
                   float tc_theta, 
                   float theta_plus_e) {
  this->N = N;
  this->adapt = adapt;
  this->tau = tau;
  this->theta = theta;
  this->vthr = vthr;
  this->vrest = vrest;
  this->vreset = vreset;
  this->refrac_per = refrac_per;
  this->i_offset = i_offset;
  this->tc_theta = tc_theta;
  this->theta_plus_e = theta_plus_e;
}

matrix<float> LIFGroup::step(float t, float dt, matrix<float> Iine, matrix<float> Iini)
{
}

void LIFGroup::reset()
{
}
