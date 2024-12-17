/*
Copyright 2024 Massimo Fioravanti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once
#include "memory"

namespace mlir::rlc
{

	void run();
	class PPOConfig
	{
		public:
		bool gae = false;
		bool shuffle_per_batch_per_epoch = true;
		int train_batch_size = 4000;
		int num_epochs = 32;
		int minibatch_size = 128;
		float lambda = 1.0;
		bool use_kl_loss = true;
		float lr = 0.00005;
		float kl_coeff = 0.2;
		float kl_target = 0.01;
		float vf_loss_coeff = 1.0;
		float entropy_coeff = 0.0;
		float clip_param = 0.3;
		float vf_clip_param = 10.0;
		float grad_clip = -1.0;
	};

	class PPOImpl;
	class PPO
	{
		public:
		PPO();
		~PPO();

		void run();

		private:
		std::unique_ptr<PPOImpl> impl;
	};
	void trainingStep();
}	 // namespace mlir::rlc
