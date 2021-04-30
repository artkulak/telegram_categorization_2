#ifndef PREDICTOR_HPP_
#define PREDICTOR_HPP_

#include <LightGBM/boosting.h>
#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/text_reader.h>
#include <LightGBM/prediction_early_stop.h>

#include <string>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tfidf_vectorizer.h"

namespace LightGBM
{

    class Predictor
    {
    public:
        Predictor(Boosting *boosting, int start_iteration, int num_iteration, bool is_raw_score,
                  bool predict_leaf_index, bool predict_contrib, bool early_stop,
                  int early_stop_freq, double early_stop_margin)
        {
            early_stop_ = CreatePredictionEarlyStopInstance(
                "none", LightGBM::PredictionEarlyStopConfig());
            if (early_stop && !boosting->NeedAccuratePrediction())
            {
                PredictionEarlyStopConfig pred_early_stop_config;
                CHECK_GT(early_stop_freq, 0);
                CHECK_GE(early_stop_margin, 0);
                pred_early_stop_config.margin_threshold = early_stop_margin;
                pred_early_stop_config.round_period = early_stop_freq;
                if (boosting->NumberOfClasses() == 1)
                {
                    early_stop_ =
                        CreatePredictionEarlyStopInstance("binary", pred_early_stop_config);
                }
                else
                {
                    early_stop_ = CreatePredictionEarlyStopInstance("multiclass",
                                                                    pred_early_stop_config);
                }
            }

            boosting->InitPredict(start_iteration, num_iteration, predict_contrib);
            boosting_ = boosting;
            num_pred_one_row_ = boosting_->NumPredictOneRow(start_iteration,
                                                            num_iteration, predict_leaf_index, predict_contrib);
            num_feature_ = boosting_->MaxFeatureIdx() + 1;
            predict_buf_.resize(
                OMP_NUM_THREADS(),
                std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>(
                    num_feature_, 0.0f));
            const int kFeatureThreshold = 100000;
            const size_t KSparseThreshold = static_cast<size_t>(0.01 * num_feature_);
            if (predict_leaf_index)
            {
                predict_fun_ = [=](const std::vector<std::pair<int, double>> &features,
                                   double *output) {
                    int tid = omp_get_thread_num();
                    if (num_feature_ > kFeatureThreshold &&
                        features.size() < KSparseThreshold)
                    {
                        auto buf = CopyToPredictMap(features);
                        boosting_->PredictLeafIndexByMap(buf, output);
                    }
                    else
                    {
                        CopyToPredictBuffer(predict_buf_[tid].data(), features);
                        // get result for leaf index
                        boosting_->PredictLeafIndex(predict_buf_[tid].data(), output);
                        ClearPredictBuffer(predict_buf_[tid].data(), predict_buf_[tid].size(),
                                           features);
                    }
                };
            }
            else if (predict_contrib)
            {
                if (boosting_->IsLinear())
                {
                    Log::Fatal("Predicting SHAP feature contributions is not implemented for linear trees.");
                }
                predict_fun_ = [=](const std::vector<std::pair<int, double>> &features,
                                   double *output) {
                    int tid = omp_get_thread_num();
                    CopyToPredictBuffer(predict_buf_[tid].data(), features);
                    // get feature importances
                    boosting_->PredictContrib(predict_buf_[tid].data(), output);
                    ClearPredictBuffer(predict_buf_[tid].data(), predict_buf_[tid].size(),
                                       features);
                };
                predict_sparse_fun_ = [=](const std::vector<std::pair<int, double>> &features,
                                          std::vector<std::unordered_map<int, double>> *output) {
                    auto buf = CopyToPredictMap(features);
                    // get sparse feature importances
                    boosting_->PredictContribByMap(buf, output);
                };
            }
            else
            {
                if (is_raw_score)
                {
                    predict_fun_ = [=](const std::vector<std::pair<int, double>> &features,
                                       double *output) {
                        int tid = omp_get_thread_num();
                        if (num_feature_ > kFeatureThreshold &&
                            features.size() < KSparseThreshold)
                        {
                            auto buf = CopyToPredictMap(features);
                            boosting_->PredictRawByMap(buf, output, &early_stop_);
                        }
                        else
                        {
                            CopyToPredictBuffer(predict_buf_[tid].data(), features);
                            boosting_->PredictRaw(predict_buf_[tid].data(), output,
                                                  &early_stop_);
                            ClearPredictBuffer(predict_buf_[tid].data(),
                                               predict_buf_[tid].size(), features);
                        }
                    };
                }
                else
                {
                    predict_fun_ = [=](const std::vector<std::pair<int, double>> &features,
                                       double *output) {
                        int tid = omp_get_thread_num();
                        if (num_feature_ > kFeatureThreshold &&
                            features.size() < KSparseThreshold)
                        {
                            auto buf = CopyToPredictMap(features);
                            boosting_->PredictByMap(buf, output, &early_stop_);
                        }
                        else
                        {
                            CopyToPredictBuffer(predict_buf_[tid].data(), features);
                            boosting_->Predict(predict_buf_[tid].data(), output, &early_stop_);
                            ClearPredictBuffer(predict_buf_[tid].data(),
                                               predict_buf_[tid].size(), features);
                        }
                    };
                }
            }
        }

        ~Predictor()
        {
        }

        inline const PredictFunction &GetPredictFunction() const
        {
            return predict_fun_;
        }

        inline const PredictSparseFunction &GetPredictSparseFunction() const
        {
            return predict_sparse_fun_;
        }

        float Predict(const char *line, TfIdfVectorizer &vectorizer)
        {
            std::vector<std::pair<int, double>> feature = vectorizer.transform_line(line);
            std::vector<double> result(1);
            predict_fun_(feature, result.data());
            return static_cast<float>(result.front());
        }

    private:
        void
        CopyToPredictBuffer(double *pred_buf, const std::vector<std::pair<int, double>> &features)
        {
            for (const auto &feature : features)
            {
                if (feature.first < num_feature_)
                {
                    pred_buf[feature.first] = feature.second;
                }
            }
        }

        void ClearPredictBuffer(double *pred_buf, size_t buf_size, const std::vector<std::pair<int, double>> &features)
        {
            if (features.size() > static_cast<size_t>(buf_size / 2))
            {
                std::memset(pred_buf, 0, sizeof(double) * (buf_size));
            }
            else
            {
                for (const auto &feature : features)
                {
                    if (feature.first < num_feature_)
                    {
                        pred_buf[feature.first] = 0.0f;
                    }
                }
            }
        }

        std::unordered_map<int, double> CopyToPredictMap(const std::vector<std::pair<int, double>> &features)
        {
            std::unordered_map<int, double> buf;
            for (const auto &feature : features)
            {
                if (feature.first < num_feature_)
                {
                    buf[feature.first] = feature.second;
                }
            }
            return buf;
        }

        const Boosting *boosting_;
        PredictFunction predict_fun_;
        PredictSparseFunction predict_sparse_fun_;
        PredictionEarlyStopInstance early_stop_;
        int num_feature_;
        int num_pred_one_row_;
        std::vector<std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>> predict_buf_;
    };

} // namespace LightGBM

#endif // PREDICTOR_HPP_