
#ifndef LOAD_MNIST_H
#define LOAD_MNIST_H

#include "defines.h"

uint8_t** read_mnist_images(std::string filepath, uint32_t& number_of_images, uint32_t& image_size);
uint8_t* read_mnist_labels(std::string full_path, uint32_t& number_of_labels);

#endif
