#include <LightGBM/dataset_loader.h>
#include <LightGBM/network.h>
#include <LightGBM/utils/common.h>

#include <string>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>
#include <utility>

#include "application.h"
#include "predictor.h"

namespace LightGBM
{

    ApplicationLightGBM::ApplicationLightGBM()
    {
        if (config_.num_threads > 0)
        {
            omp_set_num_threads(config_.num_threads);
        }
        if (config_.data.size() == 0 && config_.task != TaskType::kConvertModel)
        {
            Log::Fatal("No prediction data, ApplicationLightGBM quit");
        }
        InitPredict();
        Predict();
    }

    ApplicationLightGBM::~ApplicationLightGBM()
    {
        if (config_.is_parallel)
        {
            Network::Dispose();
        }
    }

    void ApplicationLightGBM::LoadModel(const std::string model_folder)
    {
        //need redefenition
        //config_.Set(params);
    }


    void ApplicationLightGBM::LoadData()
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::unique_ptr<Predictor> predictor;
        // prediction is needed if using input initial model(continued train)
        PredictFunction predict_fun = nullptr;
        // need to continue training
        if (boosting_->NumberOfTotalModel() > 0 && config_.task != TaskType::KRefitTree)
        {
            predictor.reset(new Predictor(boosting_.get(), 0, -1, true, false, false, false, -1, -1));
            predict_fun = predictor->GetPredictFunction();
        }

        // sync up random seed for data partition
        if (config_.is_data_based_parallel)
        {
            config_.data_random_seed = Network::GlobalSyncUpByMin(config_.data_random_seed);
        }

        Log::Debug("Loading train file...");
        DatasetLoader dataset_loader(config_, predict_fun,
                                     config_.num_class, config_.data.c_str());
        // load Training data
        if (config_.is_data_based_parallel)
        {
            // load data for distributed training
            train_data_.reset(dataset_loader.LoadFromFile(config_.data.c_str(),
                                                          Network::rank(), Network::num_machines()));
        }
        else
        {
            // load data for single machine
            train_data_.reset(dataset_loader.LoadFromFile(config_.data.c_str(), 0, 1));
        }
        // need save binary file
        if (config_.save_binary)
        {
            train_data_->SaveBinaryFile(nullptr);
        }
        // create training metric
        if (config_.is_provide_training_metric)
        {
            for (auto metric_type : config_.metric)
            {
                auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, config_));
                if (metric == nullptr)
                {
                    continue;
                }
                metric->Init(train_data_->metadata(), train_data_->num_data());
                train_metric_.push_back(std::move(metric));
            }
        }
        train_metric_.shrink_to_fit();

        if (!config_.metric.empty())
        {
            // only when have metrics then need to construct validation data

            // Add validation data, if it exists
            for (size_t i = 0; i < config_.valid.size(); ++i)
            {
                Log::Debug("Loading validation file #%zu...", (i + 1));
                // add
                auto new_dataset = std::unique_ptr<Dataset>(
                    dataset_loader.LoadFromFileAlignWithOtherDataset(
                        config_.valid[i].c_str(),
                        train_data_.get()));
                valid_datas_.push_back(std::move(new_dataset));
                // need save binary file
                if (config_.save_binary)
                {
                    valid_datas_.back()->SaveBinaryFile(nullptr);
                }

                // add metric for validation data
                valid_metrics_.emplace_back();
                for (auto metric_type : config_.metric)
                {
                    auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, config_));
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
        if (config_.task == TaskType::KRefitTree)
        {
            // create predictor
            Predictor predictor(boosting_.get(), 0, -1, false, true, false, false, 1, 1);
            predictor.Predict(config_.data.c_str(), config_.output_result.c_str(), config_.header, config_.predict_disable_shape_check);
            TextReader<int> result_reader(config_.output_result.c_str(), false);
            result_reader.ReadAllLines();
            std::vector<std::vector<int>> pred_leaf(result_reader.Lines().size());
#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(result_reader.Lines().size()); ++i)
            {
                pred_leaf[i] = Common::StringToArray<int>(result_reader.Lines()[i], '\t');
                // Free memory
                result_reader.Lines()[i].clear();
            }
            DatasetLoader dataset_loader(config_, nullptr,
                                         config_.num_class, config_.data.c_str());
            train_data_.reset(dataset_loader.LoadFromFile(config_.data.c_str(), 0, 1));
            train_metric_.clear();
            objective_fun_.reset(ObjectiveFunction::CreateObjectiveFunction(config_.objective,
                                                                            config_));
            objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
            boosting_->Init(&config_, train_data_.get(), objective_fun_.get(),
                            Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
            boosting_->RefitTree(pred_leaf);
            boosting_->SaveModelToFile(0, -1, config_.saved_feature_importance_type,
                                       config_.output_model.c_str());
            Log::Info("Finished RefitTree");
        }
        else
        {
            // create predictor
            Predictor predictor(boosting_.get(), config_.start_iteration_predict, config_.num_iteration_predict, config_.predict_raw_score,
                                config_.predict_leaf_index, config_.predict_contrib,
                                config_.pred_early_stop, config_.pred_early_stop_freq,
                                config_.pred_early_stop_margin);
            predictor.Predict(config_.data.c_str(),
                              config_.output_result.c_str(), config_.header, config_.predict_disable_shape_check);
            Log::Info("Finished prediction");
        }
    }

    void ApplicationLightGBM::InitPredict()
    {
        Log::Info("For DEBUg");
        boosting_.reset(Boosting::CreateBoosting("gbdt", config_.input_model.c_str()));
        Log::Info("Finished initializing prediction, total used %d iterations", boosting_->GetCurrentIteration());
    }

    void ApplicationLightGBM::ConvertModel()
    {
        boosting_.reset(
            Boosting::CreateBoosting(config_.boosting, config_.input_model.c_str()));
        boosting_->SaveModelToIfElse(-1, config_.convert_model.c_str());
    }

} // namespace LightGBM