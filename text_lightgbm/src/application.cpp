#include <LightGBM/dataset_loader.h>
#include <LightGBM/network.h>

#include <string>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>
#include <utility>
#include <filesystem>

#include "application.h"
#include "predictor.h"

namespace LightGBM
{

    ApplicationLightGBM::ApplicationLightGBM()
    {
    }

    ApplicationLightGBM::~ApplicationLightGBM()
    {
        if (first_config_.is_parallel)
        {
            Network::Dispose();
        }
    }

    void ApplicationLightGBM::LoadModel(const std::string models_path)
    {
        std::vector<std::unordered_map<std::string, std::string>> configs_params;
        for (const auto &model_name : std::filesystem::directory_iterator(models_path))
        {
            if (!std::filesystem::is_directory(model_name.path()))
            {
                std::unordered_map<std::string, std::string> params;

                params.insert({"task", "predict"});
                params.insert({"data", "../data/test_for_c++_inference.csv"});
                params.insert({"input_model", model_name.path()});
                configs_params.push_back(params);
            }
        }

        for (auto i = configs_params.begin(); i != configs_params.end(); ++i)
        {
            Config config;
            config.Set(*i);
            config_.push_back(config);
        }

        vectorizer_.load_model(models_path + "/tfidf/");

        first_config_ = config_.front();
        first_config_.output_result = "lightgbm_model_output_";
        if (first_config_.num_threads > 0)
        {
            omp_set_num_threads(first_config_.num_threads);
        }
        if (first_config_.data.size() == 0 && first_config_.task != TaskType::kConvertModel)
        {
            Log::Fatal("prediction data, application quit");
        }
    }

    void ApplicationLightGBM::Predict()
    {
        if (first_config_.data.size() == 0 && first_config_.task != TaskType::kConvertModel)
        {
            Log::Fatal("No prediction data, ApplicationLightGBM quit");
        }
        for (auto i = 0; i != boosting_.size(); ++i)
        {
            // create predictor
            Predictor predictor(boosting_[i].get(), first_config_.start_iteration_predict, first_config_.num_iteration_predict, first_config_.predict_raw_score,
                                first_config_.predict_leaf_index, first_config_.predict_contrib,
                                first_config_.pred_early_stop, first_config_.pred_early_stop_freq,
                                first_config_.pred_early_stop_margin);
            predictor.Predict(first_config_.data.c_str(),
                              (first_config_.output_result + std::to_string(i) + ".txt").c_str(), vectorizer_, first_config_.header, first_config_.predict_disable_shape_check);
        }
        Log::Info("Finished prediction");
    }

    void ApplicationLightGBM::InitPredict()
    {
        for (auto i = config_.begin(); i != config_.end(); ++i)
        {
            std::shared_ptr<Boosting> boosting;
            boosting.reset(Boosting::CreateBoosting("gbdt", i->input_model.c_str()));
            boosting_.push_back(boosting);
        }
        Log::Info("Finished initializing prediction, total used %d iterations", boosting_.front()->GetCurrentIteration());
    }

} // namespace LightGBM