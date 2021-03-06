#pragma once

#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <string>

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "rlc/ast/SymbolTable.hpp"
#include "rlc/ast/Type.hpp"
namespace rlc
{
	class FunctionTypeUse;
	class SingleTypeUse
	{
		public:
		friend class FunctionTypeUse;
		[[nodiscard]] const std::string& getName() const { return name; }
		[[nodiscard]] bool isArray() const { return array; }
		[[nodiscard]] size_t getDimension() const { return dim; }
		[[nodiscard]] bool isScalar() const { return !array; }
		[[nodiscard]] bool isFunctionType() const { return fType != nullptr; }
		[[nodiscard]] const FunctionTypeUse& getFunctionType() const;
		[[nodiscard]] FunctionTypeUse& getFunctionType();

		static SingleTypeUse scalarType(std::string name)
		{
			return SingleTypeUse(std::move(name), false);
		}
		static SingleTypeUse arrayType(std::string name, size_t dim = 1)
		{
			return SingleTypeUse(std::move(name), true, dim);
		}
		void print(llvm::raw_ostream& OS) const;
		void dump() const;

		SingleTypeUse(SingleTypeUse&& other) = default;
		SingleTypeUse& operator=(SingleTypeUse&& other) = default;
		SingleTypeUse(const SingleTypeUse& other);
		SingleTypeUse& operator=(const SingleTypeUse& other);
		~SingleTypeUse() = default;
		[[nodiscard]] std::string toString() const;
		[[nodiscard]] Type* getType() const { return type; }
		llvm::Error deduceType(const SymbolTable& tb, TypeDB& db);

		private:
		explicit SingleTypeUse(std::string name, bool array, size_t dim = 1)
				: name(std::move(name)), array(array), dim(dim)
		{
		}
		explicit SingleTypeUse(std::unique_ptr<FunctionTypeUse> f);
		std::string name{ "" };
		bool array{ false };
		size_t dim{ 1 };
		std::unique_ptr<FunctionTypeUse> fType{ nullptr };
		Type* type{ nullptr };
	};

	/**
	 * A type use is a position in the source code where a type appears,
	 * such as in argument declarations.
	 */
	class FunctionTypeUse
	{
		public:
		friend class SingleTypeUse;
		[[nodiscard]] auto begin() const { return types.begin(); }
		[[nodiscard]] auto begin() { return types.begin(); }
		[[nodiscard]] auto end() const { return types.end(); }
		[[nodiscard]] auto end() { return types.end(); }
		[[nodiscard]] auto argsRange()
		{
			return llvm::make_range(begin(), end() - 1);
		}
		[[nodiscard]] auto argsRange() const
		{
			return llvm::make_range(begin(), end() - 1);
		}
		[[nodiscard]] const SingleTypeUse& operator[](size_t index) const
		{
			return types[index];
		}

		[[nodiscard]] const SingleTypeUse& getArgType(size_t index) const
		{
			assert(index < argCount());
			return types[index];
		}

		[[nodiscard]] const SingleTypeUse& getReturnType() const
		{
			return types.back();
		}

		[[nodiscard]] size_t argCount() const { return types.size() - 1; }
		static SingleTypeUse functionType(llvm::SmallVector<SingleTypeUse, 2> types)
		{
			return SingleTypeUse(std::make_unique<FunctionTypeUse>(std::move(types)));
		}

		void print(llvm::raw_ostream& OS) const;
		void dump() const;

		explicit FunctionTypeUse(llvm::SmallVector<SingleTypeUse, 2> types)
				: types(std::move(types))
		{
		}
		llvm::Error deduceType(const SymbolTable& tb, TypeDB& db);
		[[nodiscard]] Type* getType() const { return type; }

		private:
		llvm::SmallVector<SingleTypeUse, 2> types;
		Type* type{ nullptr };
	};

}	 // namespace rlc
