#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "rlc/conversions/RLCToC.hpp"
#include "rlc/dialect/Dialect.h"
#include "rlc/dialect/Passes.hpp"
#include "rlc/parser/MultiFileParser.hpp"
#include "rlc/python/Interfaces.hpp"
#include "rlc/python/Passes.hpp"
#include "rlc/utils/Error.hpp"

using namespace rlc;
using namespace llvm;
using namespace std;

static llvm::codegen::RegisterCodeGenFlags Flags;
static cl::OptionCategory astDumperCategory("rlc options");
static cl::opt<string> InputFilePath(
		cl::Positional,
		cl::desc("<input-file>"),
		cl::init("-"),
		cl::cat(astDumperCategory));

cl::list<std::string> ExtraObjectFiles(
		cl::Positional, cl::desc("<extra-object-files>"));
cl::list<std::string> IncludeDirs("i", cl::desc("<include dirs>"));

static cl::opt<bool> dumpTokens(
		"token",
		cl::desc("dumps the tokens and exits"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<bool> dumpUncheckedAST(
		"unchecked",
		cl::desc("dumps the unchcked ast and exits"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<bool> dumpCheckedAST(
		"type-checked",
		cl::desc("dumps the type checked ast before template expansion and exits"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<bool> compileOnly(
		"compile",
		cl::desc("compile but do not link"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<std::string> clangPath(
		"clang",
		cl::desc("clang to used as a linker"),
		cl::init("clang"),
		cl::cat(astDumperCategory));

static cl::opt<bool> dumpPythonAST(
		"python-ast",
		cl::desc("dumps the ast of python-ast and exits"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<bool> dumpGodotWrapper(
		"godot",
		cl::desc("dumps the godot wrapper and exits"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<bool> dumpCWrapper(
		"header",
		cl::desc("dumps the c wrapper and exits"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<bool> dumpPythonWrapper(
		"python",
		cl::desc("dumps the ast of python and exits"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<bool> dumpAfterImplicit(
		"after-implicit",
		cl::desc("dumps the ast after implicit function expansions and exits"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<bool> dumpRLC(
		"rlc",
		cl::desc("dumps the ast and exits"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<bool> hidePosition(
		"hide-position",
		cl::desc("does not print source file position in ast"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<bool> dumpMLIR(
		"mlir",
		cl::desc("dumps the mlir and exits"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<bool> shared(
		"shared",
		cl::desc("compile as shared lib"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<bool> Optimize(
		"O2", cl::desc("Optimize"), cl::init(false), cl::cat(astDumperCategory));

static cl::opt<bool> dumpIR(
		"ir",
		cl::desc("dumps the llvm-ir and exits"),
		cl::init(false),
		cl::cat(astDumperCategory));

static cl::opt<string> outputFile(
		"o", cl::desc("<output-file>"), cl::init("-"), cl::cat(astDumperCategory));

static void initLLVM()
{
	using namespace llvm;
	InitializeAllTargets();
	InitializeAllTargetMCs();
	InitializeAllAsmPrinters();
	InitializeAllAsmParsers();

	PassRegistry &Registry = *llvm::PassRegistry::getPassRegistry();
	initializeCore(Registry);

	initializeScalarOpts(Registry);
	initializeVectorization(Registry);
	initializeIPO(Registry);
	initializeAnalysis(Registry);
	initializeTransformUtils(Registry);
	initializeInstCombine(Registry);
	initializeTarget(Registry);
	// For codegen passes, only passes that do IR to IR transformation are
	// supported.
	initializeExpandMemCmpPassPass(Registry);
	initializeCodeGenPreparePass(Registry);
	initializeAtomicExpandPass(Registry);
	initializeRewriteSymbolsLegacyPassPass(Registry);
	initializeWinEHPreparePass(Registry);
	initializeSafeStackLegacyPassPass(Registry);
	initializeSjLjEHPreparePass(Registry);
	initializePreISelIntrinsicLoweringLegacyPassPass(Registry);
	initializeGlobalMergePass(Registry);
	initializeIndirectBrExpandPassPass(Registry);
	initializeInterleavedLoadCombinePass(Registry);
	initializeInterleavedAccessPass(Registry);
	initializeUnreachableBlockElimLegacyPassPass(Registry);
	initializeExpandReductionsPass(Registry);
	initializeWasmEHPreparePass(Registry);
	initializeWriteBitcodePassPass(Registry);
	initializeHardwareLoopsPass(Registry);
}

static void optimize(llvm::Module &M)
{
	// Create the analysis managers.
	LoopAnalysisManager LAM;
	FunctionAnalysisManager FAM;
	CGSCCAnalysisManager CGAM;
	ModuleAnalysisManager MAM;

	// Create the new pass manager builder.
	// Take a look at the PassBuilder constructor parameters for more
	// customization, e.g. specifying a TargetMachine or various debugging
	// options.
	PassBuilder PB;

	// Register all the basic analyses with the managers.
	PB.registerModuleAnalyses(MAM);
	PB.registerCGSCCAnalyses(CGAM);
	PB.registerFunctionAnalyses(FAM);
	PB.registerLoopAnalyses(LAM);
	PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

	// Create the pass manager.
	// This one corresponds to a typical -O2 optimization pipeline.
	if (Optimize)
	{
		PB.buildModuleSimplificationPipeline(
					OptimizationLevel::O2, llvm::ThinOrFullLTOPhase::ThinLTOPreLink)
				.run(M, MAM);
		PB.buildModuleInlinerPipeline(
					OptimizationLevel::O2, llvm::ThinOrFullLTOPhase::ThinLTOPreLink)
				.run(M, MAM);
		PB.buildModuleSimplificationPipeline(
					OptimizationLevel::O2, llvm::ThinOrFullLTOPhase::ThinLTOPreLink)
				.run(M, MAM);
		PB.buildModuleOptimizationPipeline(
					OptimizationLevel::O2, llvm::ThinOrFullLTOPhase::ThinLTOPreLink)
				.run(M, MAM);
	}
	else
	{
		ModulePassManager MPM =
				PB.buildO0DefaultPipeline(OptimizationLevel::O0, true);
		MPM.run(M, MAM);
	}
}

static void compile(
		std::unique_ptr<llvm::Module> M, llvm::raw_pwrite_stream &OS)
{
	std::string Error;
	llvm::Triple triple(M->getTargetTriple());
	const auto *TheTarget = llvm::TargetRegistry::lookupTarget("", triple, Error);
	TargetOptions options =
			llvm::codegen::InitTargetOptionsFromCodeGenFlags(triple);

	CodeGenOpt::Level OLvl =
			Optimize ? CodeGenOpt::Aggressive : CodeGenOpt::Default;
	auto *Ptr = TheTarget->createTargetMachine(
			triple.getTriple(),
			"",
			"",
			options,
			llvm::codegen::getRelocModel(),
			M->getCodeModel(),
			OLvl);
	unique_ptr<TargetMachine> Target(Ptr);

	M->setDataLayout(Target->createDataLayout());
	llvm::UpgradeDebugInfo(*M);

	auto &LLVMTM = static_cast<LLVMTargetMachine &>(*Target);
	auto *MMIWP = new MachineModuleInfoWrapperPass(&LLVMTM);

	llvm::legacy::PassManager manager;
	llvm::TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));

	manager.add(new TargetLibraryInfoWrapperPass(TLII));
	manager.add(new TargetLibraryInfoWrapperPass(TLII));

	bool Err = Target->addPassesToEmitFile(
			manager, OS, nullptr, CGFT_ObjectFile, true, MMIWP);
	assert(not Err);
	manager.run(*M);
}

class RlcExitOnError
{
	public:
	RlcExitOnError(mlir::SourceMgrDiagnosticHandler &handler): handler(&handler)
	{
	}

	void operator()(llvm::Error error)
	{
		auto otherErrors =
				llvm::handleErrors(std::move(error), [this](const rlc::RlcError &e) {
					handler->emitDiagnostic(
							e.getPosition(), e.getText(), mlir::DiagnosticSeverity::Error);
					exit(-1);
				});

		exitOnErrBase(std::move(otherErrors));
	}

	template<typename T>
	T operator()(llvm::Expected<T> maybeObj)
	{
		if (not maybeObj)
			(*this)(maybeObj.takeError());

		return std::move(*maybeObj);
	}

	private:
	static ExitOnError exitOnErrBase;
	mlir::SourceMgrDiagnosticHandler *handler = nullptr;
};

ExitOnError RlcExitOnError::exitOnErrBase;

int main(int argc, char *argv[])
{
	llvm::cl::HideUnrelatedOptions(astDumperCategory);
	cl::ParseCommandLineOptions(argc, argv);
	InitLLVM X(argc, argv);
	initLLVM();
	mlir::registerAllTranslations();

	mlir::MLIRContext context;
	mlir::DialectRegistry Registry;
	Registry.insert<
			mlir::BuiltinDialect,
			mlir::memref::MemRefDialect,
			mlir::rlc::RLCDialect,
			mlir::index::IndexDialect>();
	mlir::registerLLVMDialectTranslation(Registry);
	context.appendDialectRegistry(Registry);
	context.loadAllAvailableDialects();
	MultiFileParser parser(&context, IncludeDirs);

	RlcExitOnError exitOnErr(parser.getDiagnostic());

	error_code error;
	raw_fd_ostream OS(outputFile, error);
	if (error)
	{
		errs() << error.message();
		return -1;
	}

	const auto inputFileName = llvm::sys::path::filename(InputFilePath);
	if (dumpTokens)
	{
		std::string fullPath;
		parser.getSourceMgr().AddIncludeFile(InputFilePath, SMLoc(), fullPath);
		Lexer lexer(parser.getSourceMgr().getMemoryBuffer(1)->getBuffer().data());
		lexer.print(OS);
		return 0;
	}

	auto ast = exitOnErr(parser.parse(InputFilePath));

	if (dumpUncheckedAST)
	{
		mlir::OpPrintingFlags flags;
		if (not hidePosition)
			flags.enableDebugInfo(true);
		ast->print(OS, flags);
		if (mlir::verify(ast).failed())
			return -1;
		return 0;
	}

	mlir::PassManager typeChecker(&context);
	typeChecker.addPass(mlir::rlc::createTypeCheckEntitiesPass());
	typeChecker.addPass(mlir::rlc::createTypeCheckPass());
	if (typeChecker.run(ast).failed())
	{
		mlir::OpPrintingFlags flags;
		if (not hidePosition)
			flags.enableDebugInfo(true);
		ast.print(OS, flags);

		return -1;
	}

	if (dumpCheckedAST)
	{
		mlir::OpPrintingFlags flags;
		if (not hidePosition)
			flags.enableDebugInfo(true);
		ast->print(OS, flags);
		if (mlir::verify(ast).failed())
			return -1;
		return 0;
	}

	mlir::PassManager templateInstantiator(&context);
	templateInstantiator.addPass(
			mlir::rlc::createEmitImplicitDestructorInvocationsPass());
	templateInstantiator.addPass(mlir::rlc::createEmitImplicitDestructorsPass());
	templateInstantiator.addPass(mlir::rlc::createLowerForFieldOpPass());
	templateInstantiator.addPass(mlir::rlc::createLowerIsOperationsPass());
	templateInstantiator.addPass(mlir::rlc::createLowerAssignPass());
	templateInstantiator.addPass(mlir::rlc::createLowerConstructOpPass());
	templateInstantiator.addPass(mlir::rlc::createLowerDestructorsPass());
	templateInstantiator.addPass(mlir::rlc::createInstantiateTemplatesPass());
	if (templateInstantiator.run(ast).failed())
	{
		mlir::OpPrintingFlags flags;
		if (not hidePosition)
			flags.enableDebugInfo(true);
		ast.print(OS, flags);

		return -1;
	}
	if (dumpCWrapper)
	{
		rlc::rlcToCHeader(ast, OS);
		return 0;
	}

	if (dumpGodotWrapper)
	{
		rlc::rlcToGodot(ast, OS);
		return 0;
	}

	if (dumpPythonWrapper or dumpPythonAST)
	{
		mlir::PassManager manager(&context);
		manager.addPass(mlir::python::createRLCToPythonPass());
		auto res = manager.run(ast);

		mlir::OpPrintingFlags flags;
		if (not hidePosition)
			flags.enableDebugInfo(true);
		auto verified = mlir::verify(ast);
		if (verified.failed() or res.failed() or dumpPythonAST)
		{
			ast.print(OS, flags);
			return verified.failed() or res.failed();
		}

		return mlir::rlc::python::serializePythonModule(OS, *ast.getOperation())
				.failed();
	}

	if (dumpRLC)
	{
		mlir::OpPrintingFlags flags;
		if (not hidePosition)
			flags.enableDebugInfo(true);
		ast->print(OS, flags);

		if (mlir::verify(ast).failed())
			return -1;
		return 0;
	}

	mlir::PassManager implictExpansionManager(&context);
	implictExpansionManager.addPass(mlir::rlc::createLowerActionPass());
	implictExpansionManager.addPass(mlir::rlc::createLowerConstructOpPass());
	implictExpansionManager.addPass(mlir::rlc::createLowerAssignPass());
	implictExpansionManager.addPass(
			mlir::rlc::createEmitImplicitDestructorsPass());
	implictExpansionManager.addPass(mlir::rlc::createLowerDestructorsPass());

	implictExpansionManager.addPass(mlir::rlc::createEmitImplicitAssignPass());
	implictExpansionManager.addPass(mlir::rlc::createEmitImplicitInitPass());
	implictExpansionManager.addPass(mlir::rlc::createLowerArrayCallsPass());

	if (implictExpansionManager.run(ast).failed())
	{
		mlir::OpPrintingFlags flags;
		if (not hidePosition)
			flags.enableDebugInfo(true);
		ast.print(OS, flags);

		return -1;
	}

	if (dumpAfterImplicit)
	{
		mlir::OpPrintingFlags flags;
		if (not hidePosition)
			flags.enableDebugInfo(true);
		ast.print(OS, flags);
		if (mlir::verify(ast).failed())
			return -1;
		return 0;
	}

	mlir::PassManager manager(&context);
	manager.addPass(mlir::rlc::createLowerToCfPass());
	manager.addPass(mlir::rlc::createActionStatementsToCoroPass());
	manager.addPass(mlir::rlc::createLowerToLLVMPass());
	manager.addPass(mlir::rlc::createRespectCReturnTypeCallingConventions());
	manager.addPass(mlir::rlc::createEmitTypeTypeAccessorsPass());
	if (not compileOnly)
		manager.addPass(mlir::rlc::createEmitMainPass());
	if (manager.run(ast).failed())
	{
		mlir::OpPrintingFlags flags;
		if (not hidePosition)
			flags.enableDebugInfo(true);
		ast.print(OS, flags);

		return -1;
	}

	if (dumpMLIR)
	{
		mlir::OpPrintingFlags flags;
		if (not hidePosition)
			flags.enableDebugInfo(true);
		ast.print(OS, flags);
		if (mlir::verify(ast).failed())
			return -1;
		return 0;
	}

	LLVMContext LLVMcontext;
	auto Module = mlir::translateModuleToLLVMIR(ast, LLVMcontext, inputFileName);
	Module->setTargetTriple(llvm::sys::getDefaultTargetTriple());

	assert(Module);
	optimize(*Module);
	if (dumpIR)
	{
		Module->print(OS, nullptr);
		return 0;
	}

	error_code errorCompile;
	std::string realOut = outputFile;
	if (realOut != "-")
		realOut = compileOnly ? outputFile : outputFile + ".o";
	llvm::ToolOutputFile library(
			realOut, errorCompile, llvm::sys::fs::OpenFlags::OF_None);

	if (errorCompile)
	{
		errs() << error.message();
		return -1;
	}

	compile(std::move(Module), library.os());

	if (compileOnly)
	{
		library.keep();
		return 0;
	}

	auto realPath = exitOnErr(
			llvm::errorOrToExpected(llvm::sys::findProgramByName(clangPath)));
	std::string Errors;
	llvm::SmallVector<std::string, 4> argSource;
	argSource.push_back("clang");
	argSource.push_back(library.getFilename().str());
	argSource.push_back("-o");
	argSource.push_back(outputFile);
	argSource.push_back("-lm");
	argSource.push_back(shared ? "--shared" : "");
	for (auto extraObject : ExtraObjectFiles)
		argSource.push_back(extraObject);

	llvm::SmallVector<llvm::StringRef, 4> args(
			argSource.begin(), argSource.end());

	auto res = llvm::sys::ExecuteAndWait(
			realPath, args, std::nullopt, {}, 0, 0, &Errors);
	llvm::errs() << Errors;

	Errors.clear();
	auto perms = exitOnErr(
			llvm::errorOrToExpected(llvm::sys::fs::getPermissions(outputFile)));
	llvm::sys::fs::setPermissions(
			outputFile, llvm::sys::fs::perms::owner_exe | perms);

	return res;
}
