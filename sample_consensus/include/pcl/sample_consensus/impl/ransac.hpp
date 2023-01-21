/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_SAMPLE_CONSENSUS_IMPL_RANSAC_H_
#define PCL_SAMPLE_CONSENSUS_IMPL_RANSAC_H_

#include <pcl/sample_consensus/ransac.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined _OPENMP && _OPENMP >= 201107 // We need OpenMP 3.1 for the atomic constructs
#define OPENMP_AVAILABLE_RANSAC true
#else
#define OPENMP_AVAILABLE_RANSAC false
#endif

//////////////////////////////////////////////////////////////////////////
template <typename PointT> bool
pcl::RandomSampleConsensus<PointT>::computeModel (int)
{
  // Warn and exit if no threshold was set
  if (threshold_ == std::numeric_limits<double>::max())
  {
    PCL_ERROR ("[pcl::RandomSampleConsensus::computeModel] No threshold set!\n");
    return (false);
  }

  iterations_ = 0;
  std::size_t n_best_inliers_count = 0;
  double k = std::numeric_limits<double>::max();

  Indices selection;
  Indices selection_0;
  Indices selection_1;
  Indices selection_2;
  Indices selection_3;
  Indices selection_4;
  Indices selection_5;
  Indices selection_6;
  Eigen::VectorXf model_coefficients (sac_model_->getModelSize ());

  const double log_probability  = std::log (1.0 - probability_);
  const double one_over_indices = 1.0 / static_cast<double> (sac_model_->getIndices ()->size ());
  Eigen::VectorXf n_best_inliers_count_vector = Eigen::VectorXf::Zero(sac_model_->getModelSize ());
  //float n_best_inliers_count_vector8;
  unsigned skipped_count = 0;

  // suppress infinite loops by just allowing 10 x maximum allowed iterations for invalid model parameters!
  const unsigned max_skip = max_iterations_ * 10;

  int threads = threads_;
  if (threads >= 0)
  {
#if OPENMP_AVAILABLE_RANSAC
    if (threads == 0)
    {
      threads = omp_get_num_procs();
      PCL_DEBUG ("[pcl::RandomSampleConsensus::computeModel] Automatic number of threads requested, choosing %i threads.\n", threads);
    }
#else
    // Parallelization desired, but not available
    PCL_WARN ("[pcl::RandomSampleConsensus::computeModel] Parallelization is requested, but OpenMP 3.1 is not available! Continuing without parallelization.\n");
    threads = -1;
#endif
  }

#if OPENMP_AVAILABLE_RANSAC
#pragma omp parallel if(threads > 0) num_threads(threads) shared(k, skipped_count, n_best_inliers_count) firstprivate(selection, model_coefficients) // would be nice to have a default(none)-clause here, but then some compilers complain about the shared const variables
#endif
  {
#if OPENMP_AVAILABLE_RANSAC
    if (omp_in_parallel())
#pragma omp master
      PCL_DEBUG ("[pcl::RandomSampleConsensus::computeModel] Computing in parallel with up to %i threads.\n", omp_get_num_threads());
    else
#endif
      PCL_DEBUG ("[pcl::RandomSampleConsensus::computeModel] Computing not parallel.\n");

    // Iterate
    while (true) // infinite loop with four possible breaks
    {
      // Get X samples which satisfy the model criteria
#if OPENMP_AVAILABLE_RANSAC
#pragma omp critical(samples)
#endif
      {
        sac_model_->getSamples (iterations_, selection); // The random number generator used when choosing the samples should not be called in parallel
      }

      if (selection.empty ())
      {
        PCL_ERROR ("[pcl::RandomSampleConsensus::computeModel] No samples could be selected!\n");
        break;
      }

      // Search for inliers in the point cloud for the current plane model M
      if (!sac_model_->computeModelCoefficients (selection, model_coefficients)) // This function has to be thread-safe
      {
        //++iterations_;
        unsigned skipped_count_tmp;
#if OPENMP_AVAILABLE_RANSAC
#pragma omp atomic capture
#endif
        skipped_count_tmp = ++skipped_count;
        if (skipped_count_tmp < max_skip)
          continue;
        else
          break;
      }

      // Select the inliers that are within threshold_ from the model
      //sac_model_->selectWithinDistance (model_coefficients, threshold_, inliers);
      //if (inliers.empty () && k > 1.0)
      //  continue;

      std::size_t n_inliers_count = sac_model_->countWithinDistance (model_coefficients, threshold_); // This functions has to be thread-safe. Most work is done here

      std::size_t n_best_inliers_count_tmp;
#if OPENMP_AVAILABLE_RANSAC
#pragma omp atomic read
#endif
      //n_best_inliers_count_tmp = n_best_inliers_count;
      n_best_inliers_count_tmp = n_best_inliers_count_vector(0);
      if (sac_model_->getModelSize () == 7)
        n_best_inliers_count_tmp = n_best_inliers_count_vector(6);

      if (n_inliers_count > n_best_inliers_count_tmp) // This condition is false most of the time, and the critical region is not entered, hopefully leading to more efficient concurrency
      {
        int model_similar = 10;
        int model_not_frees = 100;
        bool model_not_free_bool;
        if (sac_model_->getModelSize () == 7) //only cylinder segmentation 
          {
            points_in_current_selection_ = sac_model_->Return_points_in_selection (); //return values or must it be different function since          
            for (auto& point: points_in_current_selection_) { 
              for (auto& point_0: point_cloud0_) {
                if (sqrt((point.x-point_0.x)*(point.x-point_0.x)+(point.y-point_0.y)*(point.y-point_0.y)+(point.z-point_0.z)*(point.z-point_0.z)) < 0.6) {                  
                  model_similar = 0;
                }
                if (sqrt((point.x-point_0.x)*(point.x-point_0.x)+(point.y-point_0.y)*(point.y-point_0.y)) > 0.3) {                  
                  model_not_frees = 0;
                }
              }
              if (model_similar == 0)
                  break;
              for (auto& point_1: point_cloud1_) {
                if (sqrt((point.x-point_1.x)*(point.x-point_1.x)+(point.y-point_1.y)*(point.y-point_1.y)+(point.z-point_1.z)*(point.z-point_1.z)) < 0.6) {                  
                  model_similar = 1;
                }
                if (sqrt((point.x-point_1.x)*(point.x-point_1.x)+(point.y-point_1.y)*(point.y-point_1.y)) > 0.3) {                  
                  model_not_frees = 1;
                }
              } 
              if (model_similar == 1)
                break;              
              for (auto& point_2: point_cloud2_) {
                if (sqrt((point.x-point_2.x)*(point.x-point_2.x)+(point.y-point_2.y)*(point.y-point_2.y)+(point.z-point_2.z)*(point.z-point_2.z)) < 0.6) {                  
                  model_similar = 2;
                }
                if (sqrt((point.x-point_2.x)*(point.x-point_2.x)+(point.y-point_2.y)*(point.y-point_2.y)) > 0.3) {                  
                  model_not_frees = 2;
                }
              } 
              if (model_similar == 2)
                break;                 
              for (auto& point_3: point_cloud3_) {
                if (sqrt((point.x-point_3.x)*(point.x-point_3.x)+(point.y-point_3.y)*(point.y-point_3.y)+(point.z-point_3.z)*(point.z-point_3.z)) < 0.6) {                  
                  model_similar = 3;
                }
                if (sqrt((point.x-point_3.x)*(point.x-point_3.x)+(point.y-point_3.y)*(point.y-point_3.y)) > 0.3) {                  
                  model_not_frees = 3;
                }
              }
              if (model_similar == 3)
                break;
              for (auto& point_4: point_cloud4_) {
                if (sqrt((point.x-point_4.x)*(point.x-point_4.x)+(point.y-point_4.y)*(point.y-point_4.y)+(point.z-point_4.z)*(point.z-point_4.z)) < 0.6) {                  
                  model_similar = 4;
                }
                if (sqrt((point.x-point_4.x)*(point.x-point_4.x)+(point.y-point_4.y)*(point.y-point_4.y)) > 0.3) {                  
                  model_not_frees = 4;
                }
              }
              if (model_similar == 4)
                break;
              for (auto& point_5: point_cloud5_) {
                if (sqrt((point.x-point_5.x)*(point.x-point_5.x)+(point.y-point_5.y)*(point.y-point_5.y)+(point.z-point_5.z)*(point.z-point_5.z)) < 0.6) {                  
                  model_similar = 5;
                }
                if (sqrt((point.x-point_5.x)*(point.x-point_5.x)+(point.y-point_5.y)*(point.y-point_5.y)) > 0.3) {                  
                  model_not_frees = 5;
                }               
              }
              if (model_similar == 5)
                break;
              for (auto& point_6: point_cloud6_) {
                if (sqrt((point.x-point_6.x)*(point.x-point_6.x)+(point.y-point_6.y)*(point.y-point_6.y)+(point.z-point_6.z)*(point.z-point_6.z)) < 0.6) {                  
                  model_similar = 6;
                }
                if (sqrt((point.x-point_6.x)*(point.x-point_6.x)+(point.y-point_6.y)*(point.y-point_6.y)) > 0.3) {                  
                  model_not_frees = 6;
                }
              }
              if (model_similar == 6)
                break;
            } 
          /*  if (sqrt((points_in_current_selection_[1].x-1.5)*(points_in_current_selection_[1].x-1.5)+(points_in_current_selection_[1].y)*(points_in_current_selection_[1].y)) < 0.4) {
              std::cerr << model_similar << " ";      
              // Compute the k parameter (k=std::log(z)/std::log(1-w^n))
              const double w = static_cast<double> (n_inliers_count) * one_over_indices;
              double p_outliers = 1.0 - std::pow (w, static_cast<double> (selection.size ()));      // Probability that selection is contaminated by at least one outlier
              p_outliers = (std::max) (std::numeric_limits<double>::epsilon (), p_outliers);        // Avoid division by -Inf
              p_outliers = (std::min) (1.0 - std::numeric_limits<double>::epsilon (), p_outliers);  // Avoid division by 0.
              k = log_probability / std::log (p_outliers);
              std::cerr << "n_inliers_count: " << n_inliers_count;
              std::cerr << "k_tmp: " << k;
              std::cerr << "iterations_tmp: " << iterations_ + 1;
            } */


            //std::cerr << model_not_frees << " " << model_similar << std::endl;
          }
#if OPENMP_AVAILABLE_RANSAC
#pragma omp critical(update) // n_best_inliers_count, model_, model_coefficients_, k are shared and read/write must be protected
#endif
        {
          // Better match ?
        /*  if (model_similar == model_not_frees) {
            model_not_free_bool = 1;
            point_cloud_ = point_cloud_not_frees;
            point_cloud_not_frees = points_in_current_selection_ + point_cloud_;
            //std::cerr << "point_cloud_not_frees: " << point_cloud_not_frees.size() << " ";
          }
          else if (model_not_frees == 11) {
            model_not_free_bool = 1;
          }
          if (model_not_free_bool == 1 && model_similar == 0) {
            point_cloud0_ = point_cloud8_;
            n_best_inliers_count_vector(0) = n_best_inliers_count_vector8;
            model_coefficients0_ = model_coefficients8_;
            model0_ = model8_;
          }
          if (model_not_free_bool == 1 && model_similar == 1) {
            point_cloud1_ = point_cloud8_;
            n_best_inliers_count_vector(1) = n_best_inliers_count_vector8;
            model_coefficients1_ = model_coefficients8_;
            model1_ = model8_;
          }
          if (model_not_free_bool == 1 && model_similar == 2) {
            point_cloud2_ = point_cloud8_;
            n_best_inliers_count_vector(2) = n_best_inliers_count_vector8;
            model_coefficients2_ = model_coefficients8_;
            model2_ = model8_;
          }
          if (model_not_free_bool == 1 && model_similar == 3) {
            point_cloud3_ = point_cloud8_;
            n_best_inliers_count_vector(3) = n_best_inliers_count_vector8;
            model_coefficients3_ = model_coefficients8_;
            model3_ = model8_;
          }
          if (model_not_free_bool == 1 && model_similar == 4) {
            point_cloud4_ = point_cloud8_;
            n_best_inliers_count_vector(4) = n_best_inliers_count_vector8;
            model_coefficients4_ = model_coefficients8_;
            model4_ = model8_;
          }
          if (model_not_free_bool == 1 && model_similar == 5) {
            point_cloud5_ = point_cloud8_;
            n_best_inliers_count_vector(5) = n_best_inliers_count_vector8;
            model_coefficients5_ = model_coefficients8_;
            model5_ = model8_;
          }
          if (model_not_free_bool == 1 && model_similar == 6) {
            point_cloud6_ = point_cloud8_;
            n_best_inliers_count_vector(6) = n_best_inliers_count_vector8;
            model_coefficients6_ = model_coefficients8_;
            model6_ = model8_;
            n_best_inliers_count_vector8 = 0;
          } 

          // Compute the k parameter (k=std::log(z)/std::log(1-w^n))
          const double w = static_cast<double> (n_inliers_count) * one_over_indices;
          double p_outliers = 1.0 - std::pow (w, static_cast<double> (selection.size ()));      // Probability that selection is contaminated by at least one outlier
          p_outliers = (std::max) (std::numeric_limits<double>::epsilon (), p_outliers);        // Avoid division by -Inf
          p_outliers = (std::min) (1.0 - std::numeric_limits<double>::epsilon (), p_outliers);  // Avoid division by 0.
          k = log_probability / std::log (p_outliers);
          std::cerr << "n_inliers_count: " << n_inliers_count;
          std::cerr << "k_tmp: " << k;
          std::cerr << "iterations_tmp: " << iterations_ + 1 << std::endl; */

          if (sac_model_->getModelSize () != 7 && n_inliers_count > n_best_inliers_count_tmp) {
            n_best_inliers_count_vector(0) = n_inliers_count; // This write and the previous read of n_best_inliers_count must be consecutive and must not be interrupted!
            n_best_inliers_count_tmp = n_best_inliers_count;
            n_best_inliers_count = n_inliers_count;
            
            inliers0_ = inliers_;
            model0_ = selection;
            model_coefficients0_ = model_coefficients;
            //std::cerr << iterations_ << " ";
            //Compute the k parameter (k=std::log(z)/std::log(1-w^n))
            const double w = static_cast<double> (n_best_inliers_count) * one_over_indices;
            double p_outliers = 1.0 - std::pow (w, static_cast<double> (selection.size ()));      // Probability that selection is contaminated by at least one outlier
            p_outliers = (std::max) (std::numeric_limits<double>::epsilon (), p_outliers);        // Avoid division by -Inf
            p_outliers = (std::min) (1.0 - std::numeric_limits<double>::epsilon (), p_outliers);  // Avoid division by 0.
            k = log_probability / std::log (p_outliers);
            //std::cerr << "k_tmp: " << k << std::endl;
            //std::cerr << "iterations_tmp: " << iterations_ + 1 << std::endl;
            model_similar = 20;
         }
          
          //Fill Matrix and then always kick worst out 
          //Current selection similar to a previous found selection
          else if (model_similar == 0 && n_inliers_count > n_best_inliers_count_vector(0)) {
            point_cloud0_ = points_in_current_selection_;
            n_best_inliers_count_vector(0) = n_inliers_count;
            model_coefficients0_ = model_coefficients;
            model0_ = selection;
          }
          else if (model_similar == 1 && n_inliers_count > n_best_inliers_count_vector(1)) {
            point_cloud1_ = points_in_current_selection_;
            n_best_inliers_count_vector(1) = n_best_inliers_count;
            model_coefficients1_ = model_coefficients;
            model1_ = selection;
          }
          else if (model_similar == 2 && n_inliers_count > n_best_inliers_count_vector(2)) {
            point_cloud2_ = points_in_current_selection_;
            n_best_inliers_count_vector(2) = n_inliers_count;
            model_coefficients2_ = model_coefficients;
            model2_ = selection;
          }
          else if (model_similar == 3 && n_inliers_count > n_best_inliers_count_vector(3)) {
            point_cloud3_ = points_in_current_selection_;
            n_best_inliers_count_vector(3) = n_inliers_count;
            model_coefficients3_ = model_coefficients;
            model3_ = selection;
          }
          else if (model_similar == 4 && n_inliers_count > n_best_inliers_count_vector(4)) {
            point_cloud4_ = points_in_current_selection_;
            n_best_inliers_count_vector(4) = n_inliers_count;
            model_coefficients4_ = model_coefficients;
            model4_ = selection;
          }
          else if (model_similar == 5 && n_inliers_count > n_best_inliers_count_vector(5)) {
            point_cloud5_ = points_in_current_selection_;
            n_best_inliers_count_vector(5) = n_inliers_count;
            model_coefficients5_ = model_coefficients;
            model5_ = selection;
          }
          else if ((model_similar == 6 || model_similar == 10) && n_inliers_count > n_best_inliers_count_vector(6)) {
            point_cloud6_ = points_in_current_selection_;
            n_best_inliers_count_vector(6) = n_inliers_count;
            model_coefficients6_ = model_coefficients;
            model6_ = selection;
          }

          while (!(n_best_inliers_count_vector(0) >= n_best_inliers_count_vector(1) && n_best_inliers_count_vector(1) >= n_best_inliers_count_vector(2) && n_best_inliers_count_vector(2) >= n_best_inliers_count_vector(3) && n_best_inliers_count_vector(3) >= n_best_inliers_count_vector(4) && n_best_inliers_count_vector(4) >= n_best_inliers_count_vector(5) && n_best_inliers_count_vector(5) >= n_best_inliers_count_vector(6))) {
            if (sac_model_->getModelSize () != 7) {
              break;
            }
            if (n_best_inliers_count_vector(1) > n_best_inliers_count_vector(0)) {
              point_cloud_ = point_cloud1_;
              point_cloud1_ = point_cloud0_;
              point_cloud0_ = point_cloud_;
              int inliers = n_best_inliers_count_vector(1);
              n_best_inliers_count_vector(1) = n_best_inliers_count_vector(0);          
              n_best_inliers_count_vector(0) = inliers;
              model_coefficients_ = model_coefficients1_;
              model_coefficients1_ = model_coefficients0_;
              model_coefficients0_ = model_coefficients_;
              model_ = model1_;
              model1_ = model0_;
              model0_ = model_;
            }
            if (n_best_inliers_count_vector(2) > n_best_inliers_count_vector(1)) {
              point_cloud_ = point_cloud1_;
              point_cloud1_ = point_cloud2_;
              point_cloud2_ = point_cloud_;
              int inliers = n_best_inliers_count_vector(1);
              n_best_inliers_count_vector(1) = n_best_inliers_count_vector(2);          
              n_best_inliers_count_vector(2) = inliers;
              model_coefficients_ = model_coefficients1_;
              model_coefficients1_ = model_coefficients2_;
              model_coefficients2_ = model_coefficients_;
              model_ = model1_;
              model1_ = model2_;
              model2_ = model_;
            }
            if (n_best_inliers_count_vector(3) > n_best_inliers_count_vector(2)) {
              point_cloud_ = point_cloud3_;
              point_cloud3_ = point_cloud2_;
              point_cloud2_ = point_cloud_;
              int inliers = n_best_inliers_count_vector(3);
              n_best_inliers_count_vector(3) = n_best_inliers_count_vector(2);          
              n_best_inliers_count_vector(2) = inliers;
              model_coefficients_ = model_coefficients3_;
              model_coefficients3_ = model_coefficients2_;
              model_coefficients2_ = model_coefficients_;
              model_ = model3_;
              model3_ = model2_;
              model2_ = model_;
            }
            if (n_best_inliers_count_vector(4) > n_best_inliers_count_vector(3)) {
              point_cloud_ = point_cloud3_;
              point_cloud3_ = point_cloud4_;
              point_cloud4_ = point_cloud_;
              int inliers = n_best_inliers_count_vector(3);
              n_best_inliers_count_vector(3) = n_best_inliers_count_vector(4);          
              n_best_inliers_count_vector(4) = inliers;
              model_coefficients_ = model_coefficients3_;
              model_coefficients3_ = model_coefficients4_;
              model_coefficients4_ = model_coefficients_;
              model_ = model3_;
              model3_ = model4_;
              model4_ = model_;
            }
            if (n_best_inliers_count_vector(5) > n_best_inliers_count_vector(4)) {
              point_cloud_ = point_cloud5_;
              point_cloud5_ = point_cloud4_;
              point_cloud4_ = point_cloud_;
              int inliers = n_best_inliers_count_vector(5);
              n_best_inliers_count_vector(5) = n_best_inliers_count_vector(4);          
              n_best_inliers_count_vector(4) = inliers;
              model_coefficients_ = model_coefficients5_;
              model_coefficients5_ = model_coefficients4_;
              model_coefficients4_ = model_coefficients_;
              model_ = model5_;
              model5_ = model4_;
              model4_ = model_;
            }
            if (n_best_inliers_count_vector(6) > n_best_inliers_count_vector(5)) {
              point_cloud_ = point_cloud5_;
              point_cloud5_ = point_cloud6_;
              point_cloud6_ = point_cloud5_;
              int inliers = n_best_inliers_count_vector(5);
              n_best_inliers_count_vector(5) = n_best_inliers_count_vector(6);          
              n_best_inliers_count_vector(6) = inliers;
              model_coefficients_ = model_coefficients5_;
              model_coefficients5_ = model_coefficients6_;
              model_coefficients6_ = model_coefficients_;
              model_ = model5_;
              model5_ = model6_;
              model6_ = model_;
            }
          }
          //if (model_similar != 10)
            //std::cerr << "n_best_inliers_count_vector(model_similar): " << n_best_inliers_count_vector(model_similar) << " ";
          //if (sac_model_->getModelSize () == 7)
            //std::cerr << "n_best_inliers_count_vector: " << n_best_inliers_count_vector(0) << " " << n_best_inliers_count_vector(1) << " "<< n_best_inliers_count_vector(2) << " "<< n_best_inliers_count_vector(3) << " "<< n_best_inliers_count_vector(4) << " "<< n_best_inliers_count_vector(5) << " "<< n_best_inliers_count_vector(6) << std::endl;

/*
            // Compute the k parameter (k=std::log(z)/std::log(1-w^n))
            const double w = static_cast<double> (n_best_inliers_count) * one_over_indices;
            double p_outliers = 1.0 - std::pow (w, static_cast<double> (selection.size ()));      // Probability that selection is contaminated by at least one outlier
            p_outliers = (std::max) (std::numeric_limits<double>::epsilon (), p_outliers);        // Avoid division by -Inf
            p_outliers = (std::min) (1.0 - std::numeric_limits<double>::epsilon (), p_outliers);  // Avoid division by 0.
            k = log_probability / std::log (p_outliers); */

        } // omp critical
      }

      int iterations_tmp;
      double k_tmp;
#if OPENMP_AVAILABLE_RANSAC
#pragma omp atomic capture
#endif
      iterations_tmp = ++iterations_;
#if OPENMP_AVAILABLE_RANSAC
#pragma omp atomic read
#endif
      k_tmp = k;
#if OPENMP_AVAILABLE_RANSAC
      PCL_DEBUG ("[pcl::RandomSampleConsensus::computeModel] Trial %d out of %f: %u inliers (best is: %u so far) (thread %d).\n", iterations_tmp, k_tmp, n_inliers_count, n_best_inliers_count_tmp, omp_get_thread_num());
#else
      PCL_DEBUG ("[pcl::RandomSampleConsensus::computeModel] Trial %d out of %f: %u inliers (best is: %u so far).\n", iterations_tmp, k_tmp, n_inliers_count, n_best_inliers_count_tmp);
#endif
      if (iterations_tmp > k_tmp && sac_model_->getModelSize () != 7)
        break;
      if (iterations_tmp > max_iterations_)
      {
        PCL_DEBUG ("[pcl::RandomSampleConsensus::computeModel] RANSAC reached the maximum number of trials.\n");
        break;
      }
    } // while
  } // omp parallel

  PCL_DEBUG ("[pcl::RandomSampleConsensus::computeModel] Model: %lu size, %u inliers.\n", model_.size (), n_best_inliers_count);

  // Get the set of inliers that correspond to the found models
  if (model0_.empty ())
  {
    PCL_ERROR ("[pcl::RandomSampleConsensus::computeModel] RANSAC found no model 0.\n");
    inliers0_.clear ();
    return (false);
  }
  else {
    sac_model_->selectWithinDistance (model_coefficients0_, threshold_, inliers0_);    
  }

  if (!model1_.empty () && sac_model_->getModelSize () == 7)
  {
    sac_model_->selectWithinDistance (model_coefficients1_, threshold_, inliers1_);    
  }
  else {
    inliers1_.clear ();
  }

  if (!model2_.empty () && sac_model_->getModelSize () == 7)
  {
    sac_model_->selectWithinDistance (model_coefficients2_, threshold_, inliers2_);    
  }
  else {
    inliers2_.clear ();
  }
  
  if (!model3_.empty () && sac_model_->getModelSize () == 7)
  {
    sac_model_->selectWithinDistance (model_coefficients3_, threshold_, inliers3_);    
  }
  else {
    inliers3_.clear ();
  }
  
  if (!model4_.empty () && sac_model_->getModelSize () == 7)
  {
    sac_model_->selectWithinDistance (model_coefficients4_, threshold_, inliers4_);    
  }
  else {
    inliers4_.clear ();
  }

  if (!model5_.empty () && sac_model_->getModelSize () == 7)
  {
    sac_model_->selectWithinDistance (model_coefficients5_, threshold_, inliers5_);    
  }
  else {
    inliers5_.clear ();
  }
  
  if (!model6_.empty () && sac_model_->getModelSize () == 7)
  {
    sac_model_->selectWithinDistance (model_coefficients6_, threshold_, inliers6_);    
  }
  else {
    inliers6_.clear ();
  }


  return (true);
}

#define PCL_INSTANTIATE_RandomSampleConsensus(T) template class PCL_EXPORTS pcl::RandomSampleConsensus<T>;

#endif    // PCL_SAMPLE_CONSENSUS_IMPL_RANSAC_H_

