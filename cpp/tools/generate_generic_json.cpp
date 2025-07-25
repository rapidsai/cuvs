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
#include "llvm/Support/Path.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

static llvm::cl::OptionCategory ToolCategory("Generic JSON Generator");

static llvm::cl::opt<std::string> OutputDir("output-dir",
                                            llvm::cl::desc("Output directory for generated files"),
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

static llvm::cl::opt<std::string> ManifestFile(
  "manifest",
  llvm::cl::desc("Output file containing list of generated files"),
  llvm::cl::Required,
  llvm::cl::cat(ToolCategory));

struct GenericInfo {
  std::string class_name;
  std::string qualified_name;
  std::string namespace_path;
  std::string source_file;  // The source or header file where this instance was found
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

  std::unordered_map<std::string, GenericInfo> getClasses() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return classes_;
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

// Helper function to get fully qualified type name without CV qualifiers
std::string getQualifiedTypeName(QualType qual_type)
{
  // Get the fully qualified type name without CV qualifiers
  QualType unqualified_type = qual_type.getUnqualifiedType();
  QualType canonical_type   = unqualified_type.getCanonicalType();

  // Create a printing policy that suppresses class/struct/union keywords
  PrintingPolicy policy(LangOptions{});
  policy.SuppressTagKeyword     = true;   // This removes class/struct/union prefixes
  policy.SuppressScope          = false;  // Keep namespace qualifications
  policy.SuppressUnwrittenScope = true;   // Don't print implicit scopes

  // Use the custom printing policy to get clean type names
  std::string type_str = canonical_type.getAsString(policy);

  // Clean up some common type name issues for better readability
  // Replace _Bool with bool
  if (type_str == "_Bool") { type_str = "bool"; }

  return type_str;
}

// Helper function to get canonical path with fallbacks
std::string getCanonicalPath(const std::string& path)
{
  try {
    return std::filesystem::canonical(path).string();
  } catch (const std::filesystem::filesystem_error&) {
    // If canonical fails, fall back to absolute
    try {
      return std::filesystem::absolute(path).string();
    } catch (const std::filesystem::filesystem_error&) {
      // Final fallback to the original path
      return path;
    }
  }
}

// Function to get relative path from generated file to source header
std::string getRelativeSourcePath(const std::string& source_header_path,
                                  const std::string& generated_file_path)
{
  if (generated_file_path.empty()) {
    // Fall back to extracting from common patterns
    std::string include_str = "include/";
    size_t include_pos      = source_header_path.find(include_str);
    if (include_pos != std::string::npos) {
      return source_header_path.substr(include_pos + include_str.size());
    }

    std::string src_str = "src/";
    size_t src_pos      = source_header_path.find(src_str);
    if (src_pos != std::string::npos) {
      return source_header_path.substr(src_pos + src_str.size());
    }

    std::string tests_str = "tests/";
    size_t tests_pos      = source_header_path.find(tests_str);
    if (tests_pos != std::string::npos) {
      return source_header_path.substr(tests_pos + tests_str.size());
    }

    // Return just the filename
    return llvm::sys::path::filename(source_header_path).str();
  }

  // Try to make relative path from generated file to source header
  std::filesystem::path source_path(source_header_path);
  std::filesystem::path generated_path(generated_file_path);
  std::filesystem::path generated_dir = generated_path.parent_path();

  try {
    auto rel = std::filesystem::relative(source_path, generated_dir);
    return rel.string();
  } catch (...) {
    // Fall back to just the filename
    return llvm::sys::path::filename(source_header_path).str();
  }
}

// Visitor that traverses types to detect CRTP patterns
class GenericTypeTraversalVisitor : public RecursiveASTVisitor<GenericTypeTraversalVisitor> {
 public:
  explicit GenericTypeTraversalVisitor(ASTContext* context,
                                       std::shared_ptr<GenericClassesCollector> collector,
                                       const std::string& current_file)
    : context_(context), collector_(collector), current_file_(current_file)
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
  }

  void ProcessGenericClass(CXXRecordDecl* decl)
  {
    std::string class_name = decl->getQualifiedNameAsString();

    // Only process cuvs namespace classes
    if (class_name.find("cuvs::") != 0) { return; }

    GenericInfo info;
    info.class_name     = decl->getNameAsString();
    info.qualified_name = decl->getQualifiedNameAsString();

    // Get the enclosing namespace
    DeclContext* ctx = decl->getDeclContext();
    while (ctx && !ctx->isNamespace() && !ctx->isTranslationUnit()) {
      ctx = ctx->getParent();
    }

    if (ctx && ctx->isNamespace()) {
      if (auto* ns = dyn_cast<NamespaceDecl>(ctx)) {
        info.namespace_path = ns->getQualifiedNameAsString();
      } else {
        info.namespace_path = "::";
      }
    } else {
      info.namespace_path = "::";
    }

    // Extract header file path from source location
    SourceManager& sm  = context_->getSourceManager();
    SourceLocation loc = decl->getLocation();
    if (loc.isValid()) {
      FileID file_id = sm.getFileID(loc);
      if (const FileEntry* file_entry = sm.getFileEntryForID(file_id)) {
        std::string file_path;

        // Try to get the real path name first
        StringRef real_path = file_entry->tryGetRealPathName();
        if (!real_path.empty()) {
          file_path = real_path.str();
        } else {
          // Fall back to the filename from source manager
          file_path = sm.getFilename(loc).str();
        }

        info.source_file = getCanonicalPath(file_path);
      } else {
        // Fall back to current compilation unit
        info.source_file = getCanonicalPath(current_file_);
      }
    } else {
      // Fall back to current compilation unit
      info.source_file = getCanonicalPath(current_file_);
    }

    // Extract fields if the definition is complete
    if (decl->isCompleteDefinition()) {
      for (auto* field : decl->fields()) {
        std::string field_name = field->getNameAsString();
        std::string field_type = getQualifiedTypeName(field->getType());
        info.fields.emplace_back(field_name, field_type);
      }

      // Also extract fields from base classes
      for (const auto& base : decl->bases()) {
        if (auto* base_record = base.getType()->getAsCXXRecordDecl()) {
          if (base_record->isCompleteDefinition()) {
            for (auto* field : base_record->fields()) {
              std::string field_name = field->getNameAsString();
              std::string field_type = getQualifiedTypeName(field->getType());
              info.fields.emplace_back(field_name, field_type);
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
  std::string current_file_;
};

class GenericConsumer : public ASTConsumer {
 public:
  explicit GenericConsumer(ASTContext* context,
                           std::shared_ptr<GenericClassesCollector> collector,
                           const std::string& current_file)
    : visitor_(context, collector, current_file), context_(context)
  {
  }

  void HandleTranslationUnit(ASTContext& context) override { visitor_.TraverseAST(context); }

 private:
  GenericTypeTraversalVisitor visitor_;
  ASTContext* context_;
};

class GenericAction : public ASTFrontendAction {
 public:
  explicit GenericAction(std::shared_ptr<GenericClassesCollector> collector,
                         const std::string& current_file)
    : collector_(collector), current_file_(current_file)
  {
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& compiler,
                                                 llvm::StringRef) override
  {
    return std::make_unique<GenericConsumer>(&compiler.getASTContext(), collector_, current_file_);
  }

 private:
  std::shared_ptr<GenericClassesCollector> collector_;
  std::string current_file_;
};

class CollectingGenericAction : public GenericAction {
 public:
  explicit CollectingGenericAction(std::shared_ptr<GenericClassesCollector> collector,
                                   const std::string& current_file)
    : GenericAction(collector, current_file)
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
  explicit CollectingActionFactory(std::shared_ptr<GenericClassesCollector> collector,
                                   const std::string& current_file)
    : collector_(collector), current_file_(current_file)
  {
  }

  std::unique_ptr<FrontendAction> create() override
  {
    return std::make_unique<CollectingGenericAction>(collector_, current_file_);
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
      return true;  // Pretend success to continue with other files
    } catch (...) {
      // Ignore all compilation failures - common with CUDA files
      return true;  // Pretend success to continue with other files
    }
  }

 private:
  std::shared_ptr<GenericClassesCollector> collector_;
  std::string current_file_;
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
        CommandLineArguments AdjustedArgs;

        // Filter out all CUDA-specific compiler arguments that clang doesn't understand
        for (size_t i = 0; i < Args.size(); ++i) {
          const auto& arg = Args[i];

          // Skip all CUDA-specific arguments
          if (arg.find("--generate-code") != std::string::npos ||
              arg.find("-gencode") != std::string::npos ||
              arg.find("--gpu-architecture") != std::string::npos ||
              arg.find("-ccbin") != std::string::npos ||
              arg.find("-maxrregcount") != std::string::npos ||
              arg.find("-lineinfo") != std::string::npos ||
              arg.find("-Xptxas") != std::string::npos ||
              arg.find("-Xfatbin") != std::string::npos || arg.find("-rdc=") != std::string::npos ||
              arg.find("-Xcompiler=") != std::string::npos ||
              arg.find("--expt-") != std::string::npos ||
              arg.find("-static-global-template-stub=") != std::string::npos ||
              arg.find("-G") != std::string::npos ||
              arg.find("-forward-unknown-to-host-compiler") != std::string::npos ||
              arg.find("--suppress-stack-size-warning") != std::string::npos) {
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

      auto factory = std::make_unique<CollectingActionFactory>(collector, file);
      tool.run(factory.get());  // Ignore return code, continue with other files
    } catch (...) {
      llvm::outs() << "Exception processing file: " << file << "\n";
      llvm::outs().flush();
      // Silently ignore failures for individual files
    }
  }
}

std::string GenerateImplementations(const std::vector<GenericInfo>& classes,
                                    const std::string& generated_file_path)
{
  if (classes.empty()) { return ""; }

  std::string code = R"(// AUTO-GENERATED FILE - DO NOT EDIT
// Generated by generate_generic_json tool

)";

  // Collect unique header files from all classes
  std::set<std::string> unique_headers;
  for (const auto& cls : classes) {
    if (!cls.source_file.empty()) {
      std::string relative_path = getRelativeSourcePath(cls.source_file, generated_file_path);
      unique_headers.insert(relative_path);
    }
  }

  // Add include statements for all discovered headers
  for (const auto& header : unique_headers) {
    code += "#include \"" + header + "\"\n";
  }
  code += R"(

#include <cuvs/core/generic.hpp>

#include <nlohmann/json.hpp>

#include <optional>
#include <variant>
#include <type_traits>

using json = nlohmann::json;

template<typename Variant, std::size_t... Is>
void deserialize_at_index_impl(Variant& variant, const json& j, std::size_t index, std::index_sequence<Is...>) {
    ((index == Is ? [&]() {
        using T = std::variant_alternative_t<Is, Variant>;
        if constexpr (std::is_convertible_v<json, T>) {
            variant = j.get<T>();
        }
    }() : void()), ...);
}

template<typename Variant>
void deserialize_variant_by_index(Variant& variant, const json& j, std::size_t index) {
    constexpr auto variant_size = std::variant_size_v<Variant>;
    deserialize_at_index_impl(variant, j, index, std::make_index_sequence<variant_size>{});
}

)";

  // Generate forward declarations
  code += "// Forward declarations help with dependencies\n";
  for (const auto& cls : classes) {
    auto qname = cls.qualified_name;
    code += "namespace " + cls.namespace_path + " {\n";
    code += "void to_json(json&, const " + qname + "&);\n";
    code += "void from_json(const json&, " + qname + "&);\n";
    code += "}\n";
  }
  code += "\n\n";

  // Generate specializations
  for (const auto& cls : classes) {
    auto qname = cls.qualified_name;
    // nlohmann::json
    code += "namespace " + cls.namespace_path + " {\n";
    code += "void to_json(json& j, const " + qname + "& obj) {\n";
    for (const auto& [field_name, field_type] : cls.fields) {
      if (field_type.find("std::optional") == 0) {
        code += "  if (obj." + field_name + ".has_value()) j[\"" + field_name + "\"] = obj." +
                field_name + ".value();\n";
      } else if (field_type.find("std::variant") == 0) {
        code += "  j[\"" + field_name + "_variant\"] = obj." + field_name + ".index();\n";
        code += "  std::visit([&](const auto& value) {\n";
        code +=
          "    if constexpr (!std::is_same_v<std::decay_t<decltype(value)>, std::monostate>) {\n";
        code += "      j[\"" + field_name + "\"] = value;\n";
        code += "    }\n";
        code += "  }, obj." + field_name + ");\n";
      } else {
        code += "  j[\"" + field_name + "\"] = obj." + field_name + ";\n";
      }
    }
    code += "}\n";

    code += "void from_json(const json& j, " + qname + "& obj) {\n";
    code += "  obj = " + qname + "{};\n";
    for (const auto& [field_name, field_type] : cls.fields) {
      if (field_type.find("std::optional") == 0) {
        code += "  if (j.contains(\"" + field_name + "\")) {\n";
        code += "    obj." + field_name + " = j[\"" + field_name + "\"];\n";
        code += "  }\n";
      } else if (field_type.find("std::variant") == 0) {
        code += "  if (j.contains(\"" + field_name + "_variant\") && j.contains(\"" + field_name +
                "\")) {\n";
        code += "    auto variant_index = j[\"" + field_name + "_variant\"].get<std::size_t>();\n";
        code += "    deserialize_variant_by_index(obj." + field_name + ", j[\"" + field_name +
                "\"], variant_index);\n";
        code += "  }\n";
      } else {
        code +=
          "  obj." + field_name + " = j.value(\"" + field_name + "\", obj." + field_name + ");\n";
      }
    }
    code += "}\n";
    code += "}\n";

    // generic<T>::to_json
    code += "template<>\n";
    code += "auto cuvs::core::generic<" + qname + ">::to_json(const " + qname +
            "& obj) -> nlohmann::json {\n";
    code += "  return obj;\n";
    code += "}\n";
    code += "template<>\n";
    code += "auto cuvs::core::generic<" + qname + ">::from_json(const nlohmann::json& j) -> " +
            qname + " {\n";
    code += "  return j.get<" + qname + ">();\n";
    code += "}\n\n";
  }

  return code;
}

// Generate output filename for a given source file
std::string generateOutputFilename(const std::string& source_file)
{
  std::string relative_path = getRelativeSourcePath(source_file, "");

  // Replace path separators with underscores and remove problematic characters
  std::string safe_name = relative_path;
  std::replace(safe_name.begin(), safe_name.end(), '/', '_');
  std::replace(safe_name.begin(), safe_name.end(), '\\', '_');
  std::replace(safe_name.begin(), safe_name.end(), '.', '_');
  std::replace(safe_name.begin(), safe_name.end(), ':', '_');
  std::replace(safe_name.begin(), safe_name.end(), '-', '_');

  // Remove any double underscores
  while (safe_name.find("__") != std::string::npos) {
    size_t pos = safe_name.find("__");
    safe_name.replace(pos, 2, "_");
  }

  // Remove leading/trailing underscores
  while (!safe_name.empty() && safe_name[0] == '_') {
    safe_name = safe_name.substr(1);
  }
  while (!safe_name.empty() && safe_name.back() == '_') {
    safe_name.pop_back();
  }

  if (safe_name.empty()) { safe_name = "unknown"; }

  safe_name += "_generic";

  // Determine extension based on original file
  std::string ext = ".cpp";
  if (source_file.find(".cu") != std::string::npos ||
      source_file.find(".cuh") != std::string::npos) {
    ext = ".cu";
  }

  return safe_name + ext;
}

int main(int argc, const char** argv)
{
  // Parse command line options
  llvm::cl::ParseCommandLineOptions(argc, argv, "Generic JSON Generator\n");

  llvm::outs() << "Loading compilation database from: " << BuildPath << "\n";

  // Load the compilation database from the build directory
  std::string error_message;
  auto compilation_db = CompilationDatabase::loadFromDirectory(BuildPath, error_message);
  if (!compilation_db) {
    llvm::errs() << "Error loading compilation database from " << BuildPath << ": " << error_message
                 << "\n";
    return 1;
  }

  // Get all files from compilation database
  auto all_files = compilation_db->getAllFiles();

  // Filter out external dependencies and only keep relevant source files
  std::vector<std::string> filtered_files;
  for (const auto& file : all_files) {
    // Skip external dependencies and build artifacts
    if (file.find("_deps/") != std::string::npos || file.find("build/") != std::string::npos ||
        file.find("third") != std::string::npos) {
      continue;
    }

    filtered_files.push_back(file);
  }

  llvm::outs() << "Processing " << filtered_files.size() << " source files\n";

  if (filtered_files.empty()) {
    llvm::outs() << "No source files found\n";

    // Create empty manifest file
    std::ofstream manifest(ManifestFile);
    if (!manifest) {
      llvm::errs() << "Error: Could not open manifest file " << ManifestFile << "\n";
      return 1;
    }
    manifest.close();

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

  // Group classes by source file late in the process
  std::unordered_map<std::string, std::vector<GenericInfo>> classes_by_source;
  for (const auto& [_, info] : classes) {
    classes_by_source[info.source_file].push_back(info);
  }

  llvm::outs() << "Found " << classes.size() << " classes inheriting from cuvs::core::generic in "
               << classes_by_source.size() << " source files\n";

  // Create output directory
  std::filesystem::create_directories(OutputDir.getValue());

  // Generate files for each source file that has generic instances
  std::vector<std::string> generated_files;

  for (const auto& [source_file, classes] : classes_by_source) {
    if (classes.empty()) continue;

    std::string output_filename = generateOutputFilename(source_file);
    std::string output_path     = OutputDir.getValue() + "/" + output_filename;

    llvm::outs() << "Generating " << output_filename << " for " << classes.size()
                 << " classes from " << source_file << "\n";

    std::string generated_code = GenerateImplementations(classes, output_path);

    std::ofstream output(output_path);
    if (!output) {
      llvm::errs() << "Error: Could not open output file " << output_path << "\n";
      return 1;
    }

    output << generated_code;
    output.close();

    generated_files.push_back(output_path);
  }

  // Write manifest file
  std::ofstream manifest(ManifestFile);
  if (!manifest) {
    llvm::errs() << "Error: Could not open manifest file " << ManifestFile << "\n";
    return 1;
  }

  for (const auto& file : generated_files) {
    manifest << file << "\n";
  }
  manifest.close();

  llvm::outs() << "Generated " << generated_files.size() << " files\n";
  llvm::outs() << "Manifest written to " << ManifestFile << "\n";

  return 0;
}
