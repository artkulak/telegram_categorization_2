
#ifndef AAPPLICATION_H_
#define AAPPLICATION_H_

#include <LightGBM/boosting.h>
#include <LightGBM/dataset.h>
#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/cuda/vector_cudahost.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/text_reader.h>
#include <LightGBM/config.h>
#include <LightGBM/meta.h>

#include <memory>
#include <vector>

#include "tfidf_vectorizer.h"

namespace LightGBM
{
  class ApplicationLightGBM
  {
  public:
    ApplicationLightGBM();

    ~ApplicationLightGBM();

    void LoadModel(const std::string& model_path);

    std::vector<std::pair<float, std::string>> Predict(std::string& line);
    
    void InitPredict();
    
  private:
    Config first_config_;
    TfIdfVectorizer vectorizer_;
    std::vector<Config> config_;
    std::unique_ptr<Dataset> train_data_;
    std::vector<std::unique_ptr<Dataset>> valid_datas_;
    std::vector<std::unique_ptr<Metric>> train_metric_;
    std::vector<std::vector<std::unique_ptr<Metric>>> valid_metrics_;
    std::vector<std::shared_ptr<Boosting>> boosting_;
    std::unique_ptr<ObjectiveFunction> objective_fun_;
  };

} // namespace LightGBM

#endif // APPLICATION_H_
