
#include "load_mnist.h"

uint8_t** read_mnist_images(std::string filepath, uint32_t& number_of_images, uint32_t& image_size) {
  auto reverseInt = [](int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  };

  std::ifstream file(filepath, std::ios::binary);

  if(file.is_open()) {
    int magic_number = 0, n_rows = 0, n_cols = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if(magic_number != 2051) {
      fprintf(stderr, "Invalid MNIST image file!");
      assert(magic_number == 2051);
    }

    file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
    file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

    image_size = n_rows * n_cols;

    uint8_t** _dataset = new uint8_t*[number_of_images];
    for(int i = 0; i < number_of_images; i++) {
      _dataset[i] = new uint8_t[image_size];
      file.read((char *)_dataset[i], image_size);
    }
    return _dataset;
  } 
  else {
    fprintf(stderr, "Cannot open file %s\n", filepath.c_str());
    assert(false);
  }
}
