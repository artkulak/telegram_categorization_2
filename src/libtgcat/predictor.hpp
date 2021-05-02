#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include "../../resources/fastText/src/fasttext.h"

using namespace fasttext;

class Predictor
{
public:
  Predictor(const std::string name, const std::string model_path);

  virtual std::vector<std::pair<real, std::string>>
  predict(const std::string &data, const int32_t k = 1, const real threshold = 0.0) noexcept;

  virtual bool loadModel(const std::string &path) noexcept;

protected:
  std::string _name{"Language predictor"};
  FastText _ft;
};

#endif // PREDICTOR_HPP
