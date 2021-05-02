#include <LightGBM/dataset_loader.h>
#include <LightGBM/network.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/prediction_early_stop.h>

#include <string>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>
#include <utility>
#include <filesystem>

#include "application.h"

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

    void ApplicationLightGBM::LoadModel(const std::string &models_path)
    {
        std::vector<std::unordered_map<std::string, std::string>> configs_params;
        for (const auto &model_name : std::filesystem::directory_iterator(models_path))
        {
            if (!std::filesystem::is_directory(model_name.path()))
            {
                std::unordered_map<std::string, std::string> params;

                params.insert({"task", "predict"});
                params.insert({"input_model", model_name.path()});
                configs_params.push_back(params);
            }
        }

        for (auto i = 0; i != configs_params.size(); ++i)
        {
            Config config;
            config.Set(configs_params[i]);
            config_.push_back(config);
        }

        vectorizer_.load_model(models_path + "/tfidf/");

        first_config_ = config_.front();
        first_config_.output_result = "lightgbm_model_output_";
        if (first_config_.num_threads > 0)
        {
            omp_set_num_threads(first_config_.num_threads);
        }
    }

    std::vector<std::pair<float, std::string>> ApplicationLightGBM::Predict(const std::string &line)
    {
        PredictionEarlyStopConfig pred_early_stop_config;
        pred_early_stop_config.margin_threshold = first_config_.pred_early_stop_margin;
        pred_early_stop_config.round_period = first_config_.pred_early_stop_freq;
        PredictionEarlyStopInstance early_stop = CreatePredictionEarlyStopInstance("binary", pred_early_stop_config);

        std::vector<std::pair<float, std::string>> result;
        for (auto i = 0; i != boosting_.size(); ++i)
        {
            double *probality;
            std::string label = std::to_string(i);

            std::vector<double> features = vectorizer_.transform_line(line);

            boosting_[i]->Predict(features.data(), probality, &early_stop);
            result.push_back({static_cast<float>(*probality), label});
        }
        return result;
    }

    void ApplicationLightGBM::InitPredict()
    {
        boosting_.clear();
        for (auto i = 0; i != config_.size(); ++i)
        {
            std::shared_ptr<Boosting> boosting;
            boosting.reset(Boosting::CreateBoosting("gbdt", config_[i].input_model.c_str()));
            boosting->InitPredict(first_config_.start_iteration_predict, first_config_.num_iteration_predict, first_config_.predict_contrib);
            boosting_.push_back(boosting);
        }
        Log::Info("Finished initializing prediction, total used %d iterations", boosting_.front()->GetCurrentIteration());
    }

} // namespace LightGBM
