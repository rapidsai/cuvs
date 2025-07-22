/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

#include <cstdlib>
#include <fstream>
#include <future>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

static llvm::cl::OptionCategory ToolCategory("Generic JSON Generator");
static llvm::cl::opt<std::string> OutputFile("output",
                                             llvm::cl::desc("Output file"),
                                             llvm::cl::Required,
                                             llvm::cl::cat(ToolCategory));

static llvm::cl::opt<std::string> BuildPath(
  "p",
  llvm::cl::desc("Build path containing compile_commands.json"),
  llvm::cl::init("."),
  llvm::cl::cat(ToolCategory));

static llvm::cl::opt<unsigned> NumThreads(
  "j",
  llvm::cl::desc("Number of parallel threads (0 = auto-detect)"),
  llvm::cl::init(0),
  llvm::cl::cat(ToolCategory));

struct GenericInfo {
  std::string class_name;
  std::string qualified_name;
  std::string header_path;
  std::vector<std::pair<std::string, std::string>> fields;  // name, type
};

// Thread-safe container for collecting generic classes
class GenericClassesCollector {
 public:
  void addClass(const GenericInfo& info)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Use qualified name as key to avoid duplicates from multiple compilation units
    classes_[info.qualified_name] = info;
  }

  std::vector<GenericInfo> getClasses() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<GenericInfo> result;
    result.reserve(classes_.size());
    for (const auto& pair : classes_) {
      result.push_back(pair.second);
    }
    return result;
  }

  size_t size() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return classes_.size();
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, GenericInfo> classes_;
};

// Function to get optimal number of threads
unsigned getOptimalThreadCount()
{
  if (NumThreads.getValue() > 0) { return NumThreads.getValue(); }

  // Try to get from environment (set by CMake or ninja)
  if (const char* cmake_threads = std::getenv("CMAKE_BUILD_PARALLEL_LEVEL")) {
    unsigned threads = std::stoi(cmake_threads);
    if (threads > 0) return threads;
  }

  // Fall back to hardware concurrency
  unsigned hw_threads = std::thread::hardware_concurrency();
  return hw_threads > 0 ? hw_threads : 16;  // Default to 16 if detection fails
}

// Visitor that traverses types to detect CRTP patterns
class GenericTypeTraversalVisitor : public RecursiveASTVisitor<GenericTypeTraversalVisitor> {
 public:
  explicit GenericTypeTraversalVisitor(ASTContext* context,
                                       std::shared_ptr<GenericClassesCollector> collector)
    : context_(context), collector_(collector)
  {
  }

  // This template override is essential for finding the CRTP instances.
  bool shouldVisitTemplateInstantiations() const { return true; }

  bool VisitClassTemplateSpecializationDecl(ClassTemplateSpecializationDecl* decl)
  {
    // Find the generic template if we haven't yet
    if (!generic_template_) { FindGenericTemplate(); }

    if (!generic_template_) { return true; }

    // Check if this is a specialization of our generic template
    if (decl->getSpecializedTemplate() == generic_template_) {
      // Get the template arguments (should be the CRTP class)
      const auto& args = decl->getTemplateArgs();
      if (args.size() == 1 && args[0].getKind() == TemplateArgument::Type) {
        QualType arg_type = args[0].getAsType();
        llvm::outs() << "  Template argument: " << arg_type.getAsString() << "\n";

        if (auto* record_type = arg_type->getAs<RecordType>()) {
          if (auto* record_decl = dyn_cast<CXXRecordDecl>(record_type->getDecl())) {
            ProcessGenericClass(record_decl);
          }
        }
      }
    }

    return true;
  }

 private:
  void FindGenericTemplate()
  {
    auto* tu = context_->getTranslationUnitDecl();

    // Search for cuvs::core::generic template
    for (auto* decl : tu->decls()) {
      if (auto* ns = dyn_cast<NamespaceDecl>(decl)) {
        if (ns->getName() == "cuvs") {
          // Look for core namespace inside cuvs
          for (auto* inner_decl : ns->decls()) {
            if (auto* core_ns = dyn_cast<NamespaceDecl>(inner_decl)) {
              if (core_ns->getName() == "core") {
                // Look for generic template inside core
                for (auto* core_decl : core_ns->decls()) {
                  if (auto* template_decl = dyn_cast<ClassTemplateDecl>(core_decl)) {
                    if (template_decl->getName() == "generic") {
                      generic_template_ = template_decl;
                      llvm::outs() << "Found cuvs::core::generic template!\n";
                      return;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    llvm::outs() << "Could not find cuvs::core::generic template\n";
  }

  void ProcessGenericClass(CXXRecordDecl* decl)
  {
    std::string class_name = decl->getQualifiedNameAsString();

    // Only process cuvs namespace classes
    if (class_name.find("cuvs::") != 0) { return; }

    llvm::outs() << "Processing generic class: " << class_name << "\n";

    GenericInfo info;
    info.class_name     = decl->getNameAsString();
    info.qualified_name = decl->getQualifiedNameAsString();

    // Extract header file path from source location
    SourceManager& sm  = context_->getSourceManager();
    SourceLocation loc = decl->getLocation();
    if (loc.isValid()) {
      // Get the presumed location (accounts for #line directives)
      PresumedLoc presumed_loc = sm.getPresumedLoc(loc);
      if (presumed_loc.isValid()) {
        std::string full_path = presumed_loc.getFilename();

        // Convert to relative path from workspace root if possible
        // Look for "include/" in the path and extract from there
        std::string include_str = "include/";
        size_t include_pos      = full_path.find(include_str);
        if (include_pos != std::string::npos) {
          info.header_path = full_path.substr(include_pos + include_str.size());
        } else {
          // Fall back to just the filename
          size_t last_slash = full_path.find_last_of('/');
          if (last_slash != std::string::npos) {
            info.header_path = full_path.substr(last_slash + 1);
          } else {
            info.header_path = full_path;
          }
        }

        llvm::outs() << "  Header path: " << info.header_path << "\n";
      }
    }

    // Extract fields if the definition is complete
    if (decl->isCompleteDefinition()) {
      for (auto* field : decl->fields()) {
        std::string field_name = field->getNameAsString();
        std::string field_type = field->getType().getAsString();
        info.fields.emplace_back(field_name, field_type);
        llvm::outs() << "  Field: " << field_name << " : " << field_type << "\n";
      }

      // Also extract fields from base classes
      for (const auto& base : decl->bases()) {
        if (auto* base_record = base.getType()->getAsCXXRecordDecl()) {
          if (base_record->isCompleteDefinition()) {
            for (auto* field : base_record->fields()) {
              std::string field_name = field->getNameAsString();
              std::string field_type = field->getType().getAsString();
              info.fields.emplace_back(field_name, field_type);
              llvm::outs() << "  Inherited field: " << field_name << " : " << field_type << "\n";
            }
          }
        }
      }
    }

    collector_->addClass(info);
  }

  ASTContext* context_;
  std::shared_ptr<GenericClassesCollector> collector_;
  ClassTemplateDecl* generic_template_ = nullptr;
};

class GenericConsumer : public ASTConsumer {
 public:
  explicit GenericConsumer(ASTContext* context, std::shared_ptr<GenericClassesCollector> collector)
    : visitor_(context, collector), context_(context)
  {
  }

  void HandleTranslationUnit(ASTContext& context) override { visitor_.TraverseAST(context); }

 private:
  GenericTypeTraversalVisitor visitor_;
  ASTContext* context_;
};

class GenericAction : public ASTFrontendAction {
 public:
  explicit GenericAction(std::shared_ptr<GenericClassesCollector> collector) : collector_(collector)
  {
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& compiler,
                                                 llvm::StringRef) override
  {
    return std::make_unique<GenericConsumer>(&compiler.getASTContext(), collector_);
  }

 private:
  std::shared_ptr<GenericClassesCollector> collector_;
};

class CollectingGenericAction : public GenericAction {
 public:
  explicit CollectingGenericAction(std::shared_ptr<GenericClassesCollector> collector)
    : GenericAction(collector)
  {
  }

  void ExecuteAction() override
  {
    auto f = getCurrentFile().str();
    llvm::outs() << "Processing " << f << "\n";
    llvm::outs().flush();
    try {
      // This will now use syntactic analysis to detect CRTP inheritance
      GenericAction::ExecuteAction();
    } catch (const std::exception& e) {
      llvm::outs() << "Exception in ExecuteAction for " << f << ": " << e.what() << "\n";
      llvm::outs().flush();
    } catch (...) {
      llvm::outs() << "Unknown exception in ExecuteAction for " << f << "\n";
      llvm::outs().flush();
    }
  }
};

class CollectingActionFactory : public FrontendActionFactory {
 public:
  explicit CollectingActionFactory(std::shared_ptr<GenericClassesCollector> collector)
    : collector_(collector)
  {
  }

  std::unique_ptr<FrontendAction> create() override
  {
    return std::make_unique<CollectingGenericAction>(collector_);
  }

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager* Files,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer* DiagConsumer) override
  {
    try {
      if (!Invocation || !Files) {
        return false;  // Skip invalid invocations
      }

      // Create a null diagnostic consumer to suppress all diagnostics
      class NullDiagnosticConsumer : public DiagnosticConsumer {
       public:
        void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel, const Diagnostic& Info) override
        {
          // Ignore all diagnostics including fatal errors
        }
      };

      NullDiagnosticConsumer nullConsumer;

      return FrontendActionFactory::runInvocation(
        Invocation, Files, PCHContainerOps, &nullConsumer);
    } catch (const std::exception& e) {
      llvm::outs() << "Exception in runInvocation: " << e.what() << "\n";
      llvm::outs().flush();
      return true;  // Pretend success to continue with other files
    } catch (...) {
      // Ignore all compilation failures - common with CUDA files
      return true;  // Pretend success to continue with other files
    }
  }

 private:
  std::shared_ptr<GenericClassesCollector> collector_;
};

// Function to process a batch of files in parallel
void processFilesBatch(const CompilationDatabase& db,
                       std::vector<std::string> files,
                       std::shared_ptr<GenericClassesCollector> collector)
{
  for (const auto& file : files) {
    try {
      ClangTool tool(db, {file});

      // Add arguments to make clang more permissive with CUDA code
      tool.appendArgumentsAdjuster([file](const CommandLineArguments& Args, StringRef /*unused*/) {
        llvm::outs() << "Adjusting arguments for file: " << file << "\n";
        CommandLineArguments AdjustedArgs;

        // Be very conservative - only filter out the most problematic CUDA compiler arguments
        // Preserve all include paths and most compilation flags that CMake set up
        for (size_t i = 0; i < Args.size(); ++i) {
          const auto& arg = Args[i];

          // Only skip the most problematic CUDA compiler arguments
          if (arg.find("--generate-code") != std::string::npos ||
              arg.find("-gencode") != std::string::npos ||
              arg.find("--gpu-architecture") != std::string::npos ||
              arg.find("-ccbin") != std::string::npos ||
              arg.find("-maxrregcount") != std::string::npos ||
              arg.find("-lineinfo") != std::string::npos ||
              arg.find("-Xptxas") != std::string::npos ||
              arg.find("-Xfatbin") != std::string::npos || arg.find("-rdc=") != std::string::npos) {
            continue;  // Skip this argument
          }

          // Handle -x cu arguments (convert to -x c++)
          if (arg == "-x" && i + 1 < Args.size() && Args[i + 1] == "cu") {
            AdjustedArgs.push_back("-x");
            AdjustedArgs.push_back("c++");
            i++;  // Skip the next argument
            continue;
          }

          // Skip standalone "cu" if it follows -x
          if (i > 0 && Args[i - 1] == "-x" && arg == "cu") { continue; }

          // Keep everything else, including all include paths and compilation flags
          AdjustedArgs.push_back(arg);
        }

        // Add minimal flags to make clang more permissive for our analysis
        AdjustedArgs.push_back("-Wno-everything");              // Suppress all warnings
        AdjustedArgs.push_back("-fmax-errors=0");               // Don't limit the number of errors
        AdjustedArgs.push_back("-ferror-limit=0");              // Don't stop on errors
        AdjustedArgs.push_back("-fsyntax-only");                // Only check syntax, don't compile
        AdjustedArgs.push_back("-fno-caret-diagnostics");       // Reduce diagnostic output
        AdjustedArgs.push_back("-fno-diagnostics-fixit-info");  // Don't show fix-it information
        AdjustedArgs.push_back(
          "-frelaxed-template-template-args");   // Be more relaxed about templates
        AdjustedArgs.push_back("-fno-builtin");  // Don't assume builtin functions
        AdjustedArgs.push_back("-x");            // Specify language
        AdjustedArgs.push_back("c++");           // Treat as C++ instead of CUDA

        return AdjustedArgs;
      });

      auto factory = std::make_unique<CollectingActionFactory>(collector);
      tool.run(factory.get());  // Ignore return code, continue with other files
    } catch (...) {
      llvm::outs() << "Exception processing file: " << file << "\n";
      llvm::outs().flush();
      // Silently ignore failures for individual files
    }
  }
}

std::string GenerateImplementations(const std::vector<GenericInfo>& classes)
{
  std::string code = R"(// AUTO-GENERATED FILE - DO NOT EDIT
// Generated by generate_generic_json tool

#include <cuvs/core/generic.hpp>
)";

  // Collect unique header files from all classes
  std::set<std::string> unique_headers;
  for (const auto& cls : classes) {
    if (!cls.header_path.empty()) { unique_headers.insert(cls.header_path); }
  }

  // Add include statements for all discovered headers
  for (const auto& header : unique_headers) {
    code += "#include <" + header + ">\n";
  }
  code += R"(

#include <nlohmann/json.hpp>

namespace cuvs::core {

)";

  // Generate specializations
  for (const auto& cls : classes) {
    code += "template<>\n";
    code += "auto generic<" + cls.qualified_name + ">::to_json() const -> nlohmann::json {\n";
    code += "  nlohmann::json j;\n";
    code += "  const auto& self = this->crtp();\n";

    for (const auto& field : cls.fields) {
      code += "  j[\"" + field.first + "\"] = self." + field.first + ";\n";
    }
    code += R"(  return j;
}

)";
  }

  code += "}  // namespace cuvs::core\n\n";

  return code;
}

int main(int argc, const char** argv)
{
  // Parse command line options manually since we don't need source files from command line
  llvm::cl::ParseCommandLineOptions(argc, argv, "Generic JSON Generator\n");

  llvm::outs() << "Loading compilation database from: " << BuildPath << "\n";

  // Load the compilation database from the build directory
  std::string error_message;
  auto compilation_db = CompilationDatabase::loadFromDirectory(BuildPath, error_message);
  if (!compilation_db) {
    llvm::errs() << "Error loading compilation database from " << BuildPath << ": " << error_message
                 << "\n";
    // If we can't load compilation database, generate empty implementation
    llvm::outs() << "Generating empty implementation due to compilation database load failure\n";
    std::ofstream output(OutputFile);
    if (!output) {
      llvm::errs() << "Error: Could not open output file " << OutputFile << "\n";
      return 1;
    }
    output << GenerateImplementations({});
    return 0;
  }

  // Get all files from the compilation database
  auto all_files = compilation_db->getAllFiles();

  // Filter out external dependencies and only keep relevant source files
  std::vector<std::string> filtered_files;
  for (const auto& file : all_files) {
    // Skip external dependencies and build artifacts
    if (file.find("_deps/") != std::string::npos || file.find("build/") != std::string::npos) {
      continue;
    }

    filtered_files.push_back(file);
  }

  llvm::outs() << "Processing " << filtered_files.size() << " files out of " << all_files.size()
               << " total files\n";

  if (filtered_files.empty()) {
    llvm::outs() << "No source files found, generating empty implementation\n";
    std::ofstream output(OutputFile);
    if (!output) {
      llvm::errs() << "Error: Could not open output file " << OutputFile << "\n";
      return 1;
    }
    output << GenerateImplementations({});
    return 0;
  }

  // Global shared collector
  auto g_collector = std::make_shared<GenericClassesCollector>();

  // Get optimal number of threads
  unsigned num_threads = getOptimalThreadCount();
  llvm::outs() << "Using " << num_threads << " threads for parallel processing.\n";
  llvm::outs().flush();

  // Process files in parallel
  std::vector<std::future<void>> futures;
  size_t total_size = filtered_files.size();
  size_t batch_size = (total_size + num_threads - 1) / num_threads;

  for (unsigned start_idx = 0; start_idx < total_size; start_idx += batch_size) {
    size_t end_idx = std::min(start_idx + batch_size, total_size);

    futures.push_back(std::async(std::launch::async,
                                 processFilesBatch,
                                 std::ref(*compilation_db),
                                 std::vector<std::string>(filtered_files.begin() + start_idx,
                                                          filtered_files.begin() + end_idx),
                                 g_collector));
  }

  // Wait for all batches to complete
  for (auto& f : futures) {
    f.wait();
  }

  const auto& classes = g_collector->getClasses();

  llvm::outs() << "Found " << classes.size() << " classes inheriting from cuvs::core::generic:\n";
  for (const auto& cls : classes) {
    llvm::outs() << "  " << cls.qualified_name << " (" << cls.fields.size() << " fields)\n";
  }

  std::string generated_code = GenerateImplementations(classes);

  std::ofstream output(OutputFile);
  if (!output) {
    llvm::errs() << "Error: Could not open output file " << OutputFile << "\n";
    return 1;
  }

  output << generated_code;
  llvm::outs() << "Generated implementations written to " << OutputFile << "\n";

  return 0;
}
