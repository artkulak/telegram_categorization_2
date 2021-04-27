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
        first_config_.output_result = "lightgbm_model_output_";
        std::vector<std::unordered_map<std::string, std::string>> configs_params;
        for (const auto &model_name : std::filesystem::directory_iterator(models_path))
        {
            std::unordered_map<std::string, std::string> params;

            params.insert({"task", "predict"});
            params.insert({"data", "../data/test_for_c++_inference.csv"});
            params.insert({"input_model", model_name.path()});
            configs_params.push_back(params);
        }

        for (auto i = configs_params.begin(); i != configs_params.end(); ++i)
        {
            Config config;
            config.Set(*i);
            config_.push_back(config);
        }
    }

    void ApplicationLightGBM::LoadData()
    {
        Config first_config_ = config_.front();
        auto start_time = std::chrono::high_resolution_clock::now();
        std::unique_ptr<Predictor> predictor;
        // prediction is needed if using input initial model(continued train)
        PredictFunction predict_fun = nullptr;
        // need to continue training
        if (boosting_.front()->NumberOfTotalModel() > 0 && first_config_.task != TaskType::KRefitTree)
        {
            predictor.reset(new Predictor(boosting_.front().get(), 0, -1, true, false, false, false, -1, -1));
            predict_fun = predictor->GetPredictFunction();
        }

        // sync up random seed for data partition
        if (first_config_.is_data_based_parallel)
        {
            first_config_.data_random_seed = Network::GlobalSyncUpByMin(first_config_.data_random_seed);
        }

        Log::Debug("Loading train file...");
        DatasetLoader dataset_loader(first_config_, predict_fun,
                                     first_config_.num_class, first_config_.data.c_str());
        // load Training data
        if (first_config_.is_data_based_parallel)
        {
            // load data for distributed training
            train_data_.reset(dataset_loader.LoadFromFile(first_config_.data.c_str(),
                                                          Network::rank(), Network::num_machines()));
        }
        else
        {
            // load data for single machine
            train_data_.reset(dataset_loader.LoadFromFile(first_config_.data.c_str(), 0, 1));
        }
        // need save binary file
        if (first_config_.save_binary)
        {
            train_data_->SaveBinaryFile(nullptr);
        }
        // create training metric
        if (first_config_.is_provide_training_metric)
        {
            for (auto metric_type : first_config_.metric)
            {
                auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, first_config_));
                if (metric == nullptr)
                {
                    continue;
                }
                metric->Init(train_data_->metadata(), train_data_->num_data());
                train_metric_.push_back(std::move(metric));
            }
        }
        train_metric_.shrink_to_fit();

        if (!first_config_.metric.empty())
        {
            // only when have metrics then need to construct validation data

            // Add validation data, if it exists
            for (size_t i = 0; i < first_config_.valid.size(); ++i)
            {
                Log::Debug("Loading validation file #%zu...", (i + 1));
                // add
                auto new_dataset = std::unique_ptr<Dataset>(
                    dataset_loader.LoadFromFileAlignWithOtherDataset(
                        first_config_.valid[i].c_str(),
                        train_data_.get()));
                valid_datas_.push_back(std::move(new_dataset));
                // need save binary file
                if (first_config_.save_binary)
                {
                    valid_datas_.back()->SaveBinaryFile(nullptr);
                }

                // add metric for validation data
                valid_metrics_.emplace_back();
                for (auto metric_type : first_config_.metric)
                {
                    auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, first_config_));
                    if (metric == nullptr)
                    {
                        continue;
                    }
                    metric->Init(valid_datas_.back()->metadata(),
                                 valid_datas_.back()->num_data());
                    valid_metrics_.back().push_back(std::move(metric));
                }
                valid_metrics_.back().shrink_to_fit();
            }
            valid_datas_.shrink_to_fit();
            valid_metrics_.shrink_to_fit();
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        // output used time on each iteration
        Log::Info("Finished loading data in %f seconds",
                  std::chrono::duration<double, std::milli>(end_time - start_time) * 1e-3);
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
                              (first_config_.output_result + std::to_string(i) + ".txt").c_str(), first_config_.header, first_config_.predict_disable_shape_check);
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