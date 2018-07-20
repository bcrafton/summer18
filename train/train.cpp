
#include "defines.h"
#include "LIFGroup.h"
#include "SynapseGroup.h"
#include "load_mnist.h"

using namespace boost::numeric::ublas;

matrix<float> load_csv(std::string filename, unsigned int rows, unsigned int cols)
{
  FILE* fp = fopen(filename.c_str(), "r");
  matrix<float> m = matrix<float>(rows, cols);
  
  float next;
  
  int ii, jj;
  for(ii=0; ii<rows; ii++) {
    for(jj=0; jj<cols; jj++) {
      assert(fscanf(fp, "%f", &next));
      m(ii, jj) = next;
    }
  }
  
  return m;
}

vector<float> linspace(float start, float end, uint32_t steps)
{
  vector<float> space(steps);
  uint32_t ii;
  
  float delta = (end - start) / steps;
  
  for(ii=0; ii<steps; ii++) {
    space[ii] = start + ii * steps;
  }
  
  return space;
}

int main() {
  uint32_t s, ii, jj;

  matrix<float> w = load_csv("XeAe.csv", 784, 400);
  matrix<float> wei = load_csv("AeAi.csv", 400, 400);
  matrix<float> wie = load_csv("AiAe.csv", 400, 400);
  matrix<float> theta_e = load_csv("theta_A.csv", 400, 1);
  matrix<float> theta_i(400, 1); theta_i.clear();
  
  vector< vector<uint32_t> > training_set;
  vector<uint32_t> training_labels;
  
  uint32_t num_images;
  uint32_t image_size;
  uint32_t num_labels;
  uint8_t** images = read_mnist_images("train-images.idx3-ubyte", num_images, image_size);
  uint8_t* labels = read_mnist_labels("train-labels.idx1-ubyte", num_labels);
  assert(num_labels == num_images);
  
  training_set.resize(num_images);
  training_labels.resize(num_labels);
  
  for(ii=0; ii<num_images; ii++) {
    training_set[ii].resize(image_size);
    for(jj=0; jj<image_size; jj++) {
      training_set[ii][jj] = images[ii][jj];
    }
    
    training_labels[ii] = labels[ii];
  }
  // std::cout << training_set << std::endl;
  
  float dt = 0.0005;
  
  float active_T = 0.35;
  int active_steps = active_T / dt;
  vector<float> active_Ts = linspace(0, active_T, active_steps);
  
  float rest_T = 0.15;
  int rest_steps = rest_T / dt;
  vector<float> rest_Ts = linspace(active_T, active_T + rest_T, rest_steps);

  uint32_t NUM_EX = 1000;
  
  SynapseGroup Syn(784, 400, w, true, 0.02, 0.02, 0.04, 0.0001, 0.01, 1.0);
  LIFGroup lif_exc(400, true, 0.1, theta_e, -0.072, -0.065, -0.065, 0.005, -0.1, 0.0001, 0.00005);
  LIFGroup lif_inh(400, false, 0.01, theta_i, -0.04, -0.06, -0.045, 0.002, -0.085, 0.0001, 0.00005);

  std::cout << "starting sim" << std::endl;

  uint32_t ex = 0;
  float input_intensity = 2.0;
  
  while(ex < NUM_EX) {
  
    printf("#%u / %u\n", ex, NUM_EX);
  
    matrix<uint32_t> spkd(400, 1);
    matrix<float> Iie(400, 1);
    matrix<float> Iei(400, 1);
  
    for(s=0; s<active_steps; s++) {
      float t = active_Ts[s];
      
      vector<float> rates = training_set[ex] / 8.0 * input_intensity;
      
      matrix<uint32_t> spk(784, 1);
      for(ii=0; ii<784; ii++) {
        spk(ii, 0) = ((double) rand() / (RAND_MAX)) < rates[ii];
      }
      
      matrix<float> I = Syn.step(t, dt, spk, spkd);
      /*
      spkd = lif_exc.step(t, dt, I, Iie);
      
      matrix<float> Iei = prod(trans(spkd), wei);
      spkd = lif_inh.step(t, dt, Iei);
      
      matrix<float> Iie = prod(trans(spkd), wie);
      */
      
    } // for(s=0; s<active_steps; s++) {
    ex += 1;
  } // while(ex < NUM_EX) {
}

















