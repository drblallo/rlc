#pragma once

#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include "rlc/utils/SimpleIterator.hpp"

namespace rlc
{
	enum class BuiltinType
	{
		VOID,
		LONG,
		DOUBLE,
		BOOL
	};

	llvm::StringRef builtinTypeToString(BuiltinType t);

	class Type;
	class ArrayType
	{
		public:
		ArrayType(Type* baseType, size_t size): innerType({ baseType }), siz(size)
		{
		}
		[[nodiscard]] size_t size() const { return siz; }
		[[nodiscard]] Type* getBaseType() const { return innerType[0]; }
		[[nodiscard]] std::string getName() const;
		[[nodiscard]] auto begin() const { return innerType.begin(); }
		[[nodiscard]] auto end() const { return innerType.end(); }

		[[nodiscard]] bool operator==(const ArrayType& other) const
		{
			return siz == other.siz and innerType == other.innerType;
		}

		[[nodiscard]] bool operator!=(const ArrayType& other) const
		{
			return !(*this == other);
		}

		[[nodiscard]] std::string mangledName() const;

		private:
		const std::array<Type*, 1> innerType;
		const size_t siz;
	};

	class UserDefinedType
	{
		public:
		UserDefinedType(llvm::StringRef name, llvm::ArrayRef<Type*> memberTypes)
				: name(name), memberTypes(memberTypes.begin(), memberTypes.end())
		{
		}
		[[nodiscard]] llvm::StringRef getName() const { return name; }

		[[nodiscard]] auto begin() const { return memberTypes.begin(); }
		[[nodiscard]] auto end() const { return memberTypes.end(); }
		[[nodiscard]] size_t childrenCount() const { return memberTypes.size(); }
		[[nodiscard]] Type* getContainedType(size_t index) const
		{
			return memberTypes[index];
		}

		[[nodiscard]] bool operator==(const UserDefinedType& other) const
		{
			return name == other.name && memberTypes == other.memberTypes;
		}

		[[nodiscard]] bool operator!=(const UserDefinedType& other) const
		{
			return !(*this == other);
		}

		[[nodiscard]] std::string mangledName() const;

		void addSubType(Type* subType)
		{
			assert(subType != nullptr);
			memberTypes.emplace_back(subType);
		}

		private:
		llvm::StringRef name;
		llvm::SmallVector<Type*, 3> memberTypes;
	};

	class FunctionType
	{
		public:
		FunctionType(Type* returnType, llvm::ArrayRef<Type*> argTypes)
				: types(argTypes.begin(), argTypes.end())
		{
			types.push_back(returnType);
		}

		template<typename... Args>
		FunctionType(Type* type, Args... args): types({ type, args... })
		{
		}

		[[nodiscard]] auto begin() const { return types.begin(); }
		[[nodiscard]] auto end() const { return types.end(); }
		[[nodiscard]] Type* getContainedType(size_t index) const
		{
			assert(index < types.size());
			return types[index];
		}

		[[nodiscard]] Type* getArgumentType(size_t index) const
		{
			assert(index < argumentsCount());
			return types[index];
		}

		[[nodiscard]] auto argumentsRange() const
		{
			return llvm::make_range(begin(), end() - 1);
		}

		[[nodiscard]] bool operator==(const FunctionType& other) const
		{
			return types == other.types;
		}

		[[nodiscard]] std::string mangledName() const;

		[[nodiscard]] bool operator!=(const FunctionType& other) const
		{
			return !(*this == other);
		}

		[[nodiscard]] std::string getName() const;

		[[nodiscard]] size_t argumentsCount() const { return types.size() - 1; }
		[[nodiscard]] size_t containedTypesCount() const { return types.size(); }

		[[nodiscard]] Type* getReturnType() const { return types.back(); }

		private:
		llvm::SmallVector<Type*, 4> types;
	};

	class Type
	{
		public:
		using iterator = SimpleIterator<const Type*, const Type*, const Type*>;
		template<typename T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<T>(content);
		}

		[[nodiscard]] size_t arrayLenght() const { return get<ArrayType>().size(); }

		[[nodiscard]] Type* getBaseType() const
		{
			return get<ArrayType>().getBaseType();
		}
		[[nodiscard]] size_t getArgCount() const
		{
			return get<FunctionType>().argumentsCount();
		}

		template<typename T>
		[[nodiscard]] const T& get() const
		{
			assert(isA<T>());
			return std::get<T>(content);
		}

		[[nodiscard]] bool isBuiltin() const { return isA<BuiltinType>(); }

		[[nodiscard]] bool isArray() const { return isA<ArrayType>(); }

		[[nodiscard]] bool isUserDefined() const { return isA<UserDefinedType>(); }

		template<class Visitor>
		auto visit(Visitor&& vis) const
		{
			return std::visit(std::forward<Visitor>(vis), content);
		}

		[[nodiscard]] size_t containedTypesCount() const;
		[[nodiscard]] Type* getContainedType(size_t index) const;

		[[nodiscard]] auto argumentsRange() const
		{
			assert(isFunctionType());
			return get<FunctionType>().argumentsRange();
		}

		[[nodiscard]] std::string getName() const;

		[[nodiscard]] iterator begin() const { return iterator(this); }
		[[nodiscard]] iterator end() const
		{
			return iterator(this, containedTypesCount());
		}

		[[nodiscard]] bool isVoid() const
		{
			return isBuiltin() and get<BuiltinType>() == BuiltinType::VOID;
		}

		[[nodiscard]] bool isLong() const
		{
			return isBuiltin() and get<BuiltinType>() == BuiltinType::LONG;
		}

		[[nodiscard]] bool isDouble() const
		{
			return isBuiltin() and get<BuiltinType>() == BuiltinType::DOUBLE;
		}

		[[nodiscard]] bool isBool() const
		{
			return isBuiltin() and get<BuiltinType>() == BuiltinType::BOOL;
		}

		[[nodiscard]] bool isFunctionType() const { return isA<FunctionType>(); }

		[[nodiscard]] Type* getReturnType() const
		{
			assert(isFunctionType());
			return get<FunctionType>().getReturnType();
		}

		[[nodiscard]] Type* getArgumentType(size_t index) const
		{
			assert(isFunctionType());
			return get<FunctionType>().getArgumentType(index);
		}

		[[nodiscard]] bool operator==(const Type& other) const
		{
			return content == other.content;
		}

		[[nodiscard]] bool operator!=(const Type& other) const
		{
			return !(*this == other);
		}

		[[nodiscard]] BuiltinType getBuiltinType() const
		{
			assert(isBuiltin());
			return get<BuiltinType>();
		}

		[[nodiscard]] size_t getArraySize() const
		{
			assert(isArray());
			return get<ArrayType>().size();
		}

		void addSubType(Type* subType)
		{
			assert(isUserDefined());
			assert(subType != nullptr);

			get<UserDefinedType>().addSubType(subType);
		}

		[[nodiscard]] std::string mangledName() const;

		void print(llvm::raw_ostream& out);
		void dump();
		Type(BuiltinType b): content(b) {}
		Type(Type* t, size_t s): content(ArrayType(t, s)) {}
		Type(Type* t, llvm::ArrayRef<Type*> args): content(FunctionType(t, args)) {}
		Type(llvm::StringRef name, llvm::ArrayRef<Type*> members)
				: content(UserDefinedType(name, members))
		{
		}

		private:
		template<typename T>
		[[nodiscard]] T& get()
		{
			assert(isA<T>());
			return std::get<T>(content);
		}

		std::variant<BuiltinType, ArrayType, UserDefinedType, FunctionType> content;
	};

	class TypeDB
	{
		public:
		[[nodiscard]] Type* getLongType() { return getBuiltin(BuiltinType::LONG); }
		[[nodiscard]] Type* getFloatType()
		{
			return getBuiltin(BuiltinType::DOUBLE);
		}
		[[nodiscard]] Type* getBoolType() { return getBuiltin(BuiltinType::BOOL); }
		[[nodiscard]] Type* getDoubleType()
		{
			return getBuiltin(BuiltinType::DOUBLE);
		}
		[[nodiscard]] Type* getVoidType() { return getBuiltin(BuiltinType::VOID); }

		template<BuiltinType b>
		[[nodiscard]] Type* getBuiltin()
		{
			return getBuiltin(b);
		}

		[[nodiscard]] Type* getUserDefined(llvm::StringRef name);

		[[nodiscard]] Type* getBuiltin(BuiltinType b);

		[[nodiscard]] Type* createUserDefinedType(
				std::string name, llvm::ArrayRef<Type*> members);

		template<typename... Members>
		[[nodiscard]] Type* createUserDefinedType(
				std::string name, Members&&... members)
		{
			return createUserDefinedType(
					std::move(name), { std::forward<Members>(members)... });
		}

		[[nodiscard]] Type* getUserDefinedType(llvm::StringRef name)
		{
			if (auto f = userDefinedTypes.find(name); f != userDefinedTypes.end())
				return f->second.get();
			return nullptr;
		}

		[[nodiscard]] Type* getArrayType(Type* tp, size_t size);
		[[nodiscard]] Type* getFunctionType(
				Type* returnType, llvm::ArrayRef<Type*> args);

		template<typename... Args>
		[[nodiscard]] Type* getFunctionType(Type* retType, Args*... args)
		{
			return getFunctionType(retType, { args... });
		}
		Type* nameToBuiltinType(llvm::StringRef name);

		private:
		llvm::BumpPtrAllocator allocator;
		llvm::StringMap<std::unique_ptr<Type>> userDefinedTypes;
		std::map<BuiltinType, std::unique_ptr<Type>> builtinTypes;
		std::map<std::pair<Type*, size_t>, std::unique_ptr<Type>> arrayTypes;
		std::map<llvm::SmallVector<Type*, 3>, std::unique_ptr<Type>> functionTypes;
	};
}	 // namespace rlc
