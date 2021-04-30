#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include "../../resources/fastText/src/fasttext.h"
#include "../text_lightgbm/application.h"
using namespace fasttext;
using namespace LightGBM;

class IPredict
{
public:
  virtual std::vector<std::pair<real, std::string>>
  predict(const std::string &data, const int32_t k = 1, const real threshold = 0.0) noexcept {}

  virtual bool loadModel(const std::string &path) noexcept {}
};

class PredictorLanguage : public IPredict
{
public:
  PredictorLanguage(const std::string name, const std::string model_path);

  virtual std::vector<std::pair<real, std::string>>
  predict(const std::string &data, const int32_t k = 1, const real threshold = 0.0) noexcept;

  virtual bool loadModel(const std::string &path) noexcept;

protected:
  std::string _name{"Predictor"};
  FastText _ft;
};

class PredictCategory : public IPredict
{
public:
  PredictCategory(const std::string name, const std::string model_path);

  virtual std::vector<std::pair<real, std::string>>
  predict(const std::string &data, const int32_t k = 1, const real threshold = 0.0) noexcept;

  virtual bool loadModel(const std::string &path) noexcept;

protected:
  std::string _name{"Predictor"};
  ApplicationLightGBM _text_lightgbm;
};

#endif // PREDICTOR_HPP
