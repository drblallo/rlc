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

#include "rlc/ml/PPO.hpp"

#include "torch/torch.h"

namespace
{
	struct PPONet: torch::nn::Module
	{
		PPONet(int64_t N, int64_t M)
		{
			W = register_parameter("W", torch::randn({ N, M }));
			b = register_parameter("b", torch::randn(M));
		}
		torch::Tensor forward(torch::Tensor input)
		{
			return torch::addmm(b, input, W);
		}
		torch::Tensor W, b;
	};
}	 // namespace

namespace mlir::rlc
{
	class PPOImpl
	{
		public:
		bool init()
		{
			if (not torch::cuda::is_available())
			{
				std::cerr << "Cuda not available\n";
				return false;
			}
			torch::Device d("cuda:0");
			learnerTensorOptions = learnerTensorOptions.device(d);

			return true;
		}

		void run()
		{
			torch::Tensor tensor = torch::eye(3, learnerTensorOptions);
			std::cout << tensor.device() << std::endl;
		}

		private:
		torch::TensorOptions learnerTensorOptions;
	};

	void PPO::run() { impl->run(); }

	PPO::PPO() { impl = std::make_unique<PPOImpl>(); }
	PPO::~PPO() { impl.reset(); }

}	 // namespace mlir::rlc
