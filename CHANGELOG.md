# cuvs 25.02.00 (13 Feb 2025)

## 🚨 Breaking Changes

- update pip devcontainers to UCX 1.18, small update-version.sh fixes ([#604](https://github.com/rapidsai/cuvs/pull/604)) [@jameslamb](https://github.com/jameslamb)
- Improve the performance of CAGRA new vector addition with the default params ([#569](https://github.com/rapidsai/cuvs/pull/569)) [@enp1s0](https://github.com/enp1s0)
- Update for raft logger changes ([#540](https://github.com/rapidsai/cuvs/pull/540)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Fix ann-bench dataset blob integer overflow leading to incorrect data copy beyond 4B elems ([#671](https://github.com/rapidsai/cuvs/pull/671)) [@achirkin](https://github.com/achirkin)
- Fix ann-bench deadlocking on HNSW destruction due to task locks ([#667](https://github.com/rapidsai/cuvs/pull/667)) [@achirkin](https://github.com/achirkin)
- cuvs-bench Fixes ([#654](https://github.com/rapidsai/cuvs/pull/654)) [@tarang-jain](https://github.com/tarang-jain)
- Fix std::lock_guard use for gcc 14 support ([#639](https://github.com/rapidsai/cuvs/pull/639)) [@enp1s0](https://github.com/enp1s0)
- Fix indexing bug when using parallelism to build CPU hierarchy in HNSW ([#620](https://github.com/rapidsai/cuvs/pull/620)) [@divyegala](https://github.com/divyegala)
- add runtime dependency on libcuvs in cuvs wheels ([#615](https://github.com/rapidsai/cuvs/pull/615)) [@jameslamb](https://github.com/jameslamb)
- Temporarily skip CUDA 11 wheel CI ([#599](https://github.com/rapidsai/cuvs/pull/599)) [@bdice](https://github.com/bdice)
- [Fix] l2_exp random fail in half-float32 mixed precision on self-neighboring ([#596](https://github.com/rapidsai/cuvs/pull/596)) [@rhdong](https://github.com/rhdong)
- Add CAGRA InnerProduct test and fix a bug ([#595](https://github.com/rapidsai/cuvs/pull/595)) [@enp1s0](https://github.com/enp1s0)
- fix cuvs_bench.run --groups options ([#592](https://github.com/rapidsai/cuvs/pull/592)) [@jiangyinzuo](https://github.com/jiangyinzuo)
- Fix cagra_hnsw serialization when dataset is not part of index ([#591](https://github.com/rapidsai/cuvs/pull/591)) [@tfeher](https://github.com/tfeher)
- fix create_pointset for throughput mode ([#589](https://github.com/rapidsai/cuvs/pull/589)) [@jiangyinzuo](https://github.com/jiangyinzuo)
- Fix the use of constexpr in the dynamic batching header ([#582](https://github.com/rapidsai/cuvs/pull/582)) [@achirkin](https://github.com/achirkin)
- Reduce the recall threshold for IVF-PQ low-precision LUT inner product  tests ([#573](https://github.com/rapidsai/cuvs/pull/573)) [@achirkin](https://github.com/achirkin)
- Small fixes to docs and pairwise distances ([#570](https://github.com/rapidsai/cuvs/pull/570)) [@cjnolet](https://github.com/cjnolet)
- [BUG] Fix CAGRA graph optimization bug ([#565](https://github.com/rapidsai/cuvs/pull/565)) [@enp1s0](https://github.com/enp1s0)
- Fix broken link to python doc ([#564](https://github.com/rapidsai/cuvs/pull/564)) [@lowener](https://github.com/lowener)
- Fix cagra::extend error message ([#532](https://github.com/rapidsai/cuvs/pull/532)) [@enp1s0](https://github.com/enp1s0)
- Fix Grace-specific issues in CAGRA ([#527](https://github.com/rapidsai/cuvs/pull/527)) [@achirkin](https://github.com/achirkin)

## 📖 Documentation

- add docs for nn_descent ([#668](https://github.com/rapidsai/cuvs/pull/668)) [@Intron7](https://github.com/Intron7)
- Fixing small typo in cuvs bench docs ([#586](https://github.com/rapidsai/cuvs/pull/586)) [@cjnolet](https://github.com/cjnolet)
- Fix typos in README ([#543](https://github.com/rapidsai/cuvs/pull/543)) [@nvanbenschoten](https://github.com/nvanbenschoten)
- Use nvidia-sphinx-theme for docs ([#528](https://github.com/rapidsai/cuvs/pull/528)) [@benfred](https://github.com/benfred)

## 🚀 New Features

- Add deep-100M to datasets.yaml for cuvs-bench ([#670](https://github.com/rapidsai/cuvs/pull/670)) [@tarang-jain](https://github.com/tarang-jain)
- Expose configuration singleton as a global context for ann-bench algos ([#647](https://github.com/rapidsai/cuvs/pull/647)) [@achirkin](https://github.com/achirkin)
- ANN_BENCH enhanced dataset support ([#624](https://github.com/rapidsai/cuvs/pull/624)) [@achirkin](https://github.com/achirkin)
- [Feat] Add Support for Index `merge` in CAGRA ([#618](https://github.com/rapidsai/cuvs/pull/618)) [@rhdong](https://github.com/rhdong)
- HNSW GPU hierarchy ([#616](https://github.com/rapidsai/cuvs/pull/616)) [@divyegala](https://github.com/divyegala)
- CAGRA binary Hamming distance support ([#610](https://github.com/rapidsai/cuvs/pull/610)) [@enp1s0](https://github.com/enp1s0)
- Add cuda 12.8 support ([#605](https://github.com/rapidsai/cuvs/pull/605)) [@robertmaynard](https://github.com/robertmaynard)
- Add support for refinement with `uint32_t` index type ([#563](https://github.com/rapidsai/cuvs/pull/563)) [@lowener](https://github.com/lowener)
- [Feat] Support `bitset` filter for Brute Force ([#560](https://github.com/rapidsai/cuvs/pull/560)) [@rhdong](https://github.com/rhdong)
- Remove upper bounds on cuda-python to allow 12.6.2 and 11.8.5 ([#508](https://github.com/rapidsai/cuvs/pull/508)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Add filtering to python for ivf_flat ([#664](https://github.com/rapidsai/cuvs/pull/664)) [@benfred](https://github.com/benfred)
- Expose binary quantizer to C and Python ([#660](https://github.com/rapidsai/cuvs/pull/660)) [@benfred](https://github.com/benfred)
- Add telemetry ([#652](https://github.com/rapidsai/cuvs/pull/652)) [@gforsyth](https://github.com/gforsyth)
- Revert docs builds to CI latest tag. ([#643](https://github.com/rapidsai/cuvs/pull/643)) [@bdice](https://github.com/bdice)
- Add float16 support in python for cagra/brute_force/ivf_pq and scalar quantizer ([#637](https://github.com/rapidsai/cuvs/pull/637)) [@benfred](https://github.com/benfred)
- Expose NN-Descent to C and Python ([#635](https://github.com/rapidsai/cuvs/pull/635)) [@benfred](https://github.com/benfred)
- Revert CUDA 12.8 shared workflow branch changes ([#630](https://github.com/rapidsai/cuvs/pull/630)) [@vyasr](https://github.com/vyasr)
- cuvs-java: Rework the api to be Java 21 friendly ([#628](https://github.com/rapidsai/cuvs/pull/628)) [@ChrisHegarty](https://github.com/ChrisHegarty)
- Build and test with CUDA 12.8.0 ([#621](https://github.com/rapidsai/cuvs/pull/621)) [@bdice](https://github.com/bdice)
- Add Scalar Quantization to the c and python apis ([#617](https://github.com/rapidsai/cuvs/pull/617)) [@benfred](https://github.com/benfred)
- Iteratively build graph index ([#612](https://github.com/rapidsai/cuvs/pull/612)) [@anaruse](https://github.com/anaruse)
- update pip devcontainers to UCX 1.18, small update-version.sh fixes ([#604](https://github.com/rapidsai/cuvs/pull/604)) [@jameslamb](https://github.com/jameslamb)
- Reduce CAGRA test runtime ([#602](https://github.com/rapidsai/cuvs/pull/602)) [@bdice](https://github.com/bdice)
- Revert &quot;Temporarily skip CUDA 11 wheel CI&quot; ([#601](https://github.com/rapidsai/cuvs/pull/601)) [@bdice](https://github.com/bdice)
- introduce libcuvs wheels ([#594](https://github.com/rapidsai/cuvs/pull/594)) [@jameslamb](https://github.com/jameslamb)
- Normalize whitespace ([#593](https://github.com/rapidsai/cuvs/pull/593)) [@bdice](https://github.com/bdice)
- Rename test to tests. ([#590](https://github.com/rapidsai/cuvs/pull/590)) [@bdice](https://github.com/bdice)
- Use cuda.bindings layout. ([#588](https://github.com/rapidsai/cuvs/pull/588)) [@bdice](https://github.com/bdice)
- run_cuvs_pytests.sh uses proper test dir ([#584](https://github.com/rapidsai/cuvs/pull/584)) [@robertmaynard](https://github.com/robertmaynard)
- expose col-major bfknn to python ([#575](https://github.com/rapidsai/cuvs/pull/575)) [@benfred](https://github.com/benfred)
- Run cuvs-bench pytests and end-to-end tests in CI ([#574](https://github.com/rapidsai/cuvs/pull/574)) [@dantegd](https://github.com/dantegd)
- Expose col-major pairwise distances to python ([#572](https://github.com/rapidsai/cuvs/pull/572)) [@benfred](https://github.com/benfred)
- Improve the performance of CAGRA new vector addition with the default params ([#569](https://github.com/rapidsai/cuvs/pull/569)) [@enp1s0](https://github.com/enp1s0)
- Improve filtering documentation ([#568](https://github.com/rapidsai/cuvs/pull/568)) [@lowener](https://github.com/lowener)
- Use GCC 13 in CUDA 12 conda builds. ([#567](https://github.com/rapidsai/cuvs/pull/567)) [@bdice](https://github.com/bdice)
- Allow brute_force::build to work on host matrix dataset ([#562](https://github.com/rapidsai/cuvs/pull/562)) [@benfred](https://github.com/benfred)
- FAISS with cuVS enabled in cuvs-bench ([#561](https://github.com/rapidsai/cuvs/pull/561)) [@tarang-jain](https://github.com/tarang-jain)
- Vamana build improvement and added docs ([#558](https://github.com/rapidsai/cuvs/pull/558)) [@bkarsin](https://github.com/bkarsin)
- Support raft&#39;s logger targets ([#557](https://github.com/rapidsai/cuvs/pull/557)) [@vyasr](https://github.com/vyasr)
- Get Breathe from conda again ([#554](https://github.com/rapidsai/cuvs/pull/554)) [@vyasr](https://github.com/vyasr)
- Check if nightlies have succeeded recently enough ([#548](https://github.com/rapidsai/cuvs/pull/548)) [@vyasr](https://github.com/vyasr)
- Add support for float16 to the python pairwise distance api ([#547](https://github.com/rapidsai/cuvs/pull/547)) [@benfred](https://github.com/benfred)
- Additional Distances for CAGRA C and Python API ([#546](https://github.com/rapidsai/cuvs/pull/546)) [@tarang-jain](https://github.com/tarang-jain)
- remove setup.cfg files, other packaging cleanup ([#544](https://github.com/rapidsai/cuvs/pull/544)) [@jameslamb](https://github.com/jameslamb)
- Fix CI for python cuvs_bench ([#541](https://github.com/rapidsai/cuvs/pull/541)) [@benfred](https://github.com/benfred)
- Update for raft logger changes ([#540](https://github.com/rapidsai/cuvs/pull/540)) [@vyasr](https://github.com/vyasr)
- Change brute_force api to match ivf*/cagra ([#536](https://github.com/rapidsai/cuvs/pull/536)) [@benfred](https://github.com/benfred)
- Branch 25.02 merge 24.12 ([#526](https://github.com/rapidsai/cuvs/pull/526)) [@benfred](https://github.com/benfred)
- Update cuda-python lower bounds to 12.6.2 / 11.8.5 ([#524](https://github.com/rapidsai/cuvs/pull/524)) [@bdice](https://github.com/bdice)
- Automatic adjustment of itopk size according to filtering rate ([#509](https://github.com/rapidsai/cuvs/pull/509)) [@anaruse](https://github.com/anaruse)
- prefer system install of UCX in devcontainers ([#501](https://github.com/rapidsai/cuvs/pull/501)) [@jameslamb](https://github.com/jameslamb)
- Adapt to rmm logger changes ([#499](https://github.com/rapidsai/cuvs/pull/499)) [@vyasr](https://github.com/vyasr)
- Require approval to run CI on draft PRs ([#498](https://github.com/rapidsai/cuvs/pull/498)) [@bdice](https://github.com/bdice)
- Remove RAFT BUILD_ANN_BENCH option ([#497](https://github.com/rapidsai/cuvs/pull/497)) [@bdice](https://github.com/bdice)
- Update example code fetching rapids-cmake to use CUVS instead of RAFT ([#493](https://github.com/rapidsai/cuvs/pull/493)) [@bdice](https://github.com/bdice)
- Improve multi-CTA algorithm ([#492](https://github.com/rapidsai/cuvs/pull/492)) [@anaruse](https://github.com/anaruse)
- Add filtering for CAGRA to C API ([#452](https://github.com/rapidsai/cuvs/pull/452)) [@ajit283](https://github.com/ajit283)
- Initial cut for a cuVS Java API ([#450](https://github.com/rapidsai/cuvs/pull/450)) [@chatman](https://github.com/chatman)
- Add breaking change workflow trigger ([#442](https://github.com/rapidsai/cuvs/pull/442)) [@AyodeAwe](https://github.com/AyodeAwe)
- Expose `extend()` in C API ([#276](https://github.com/rapidsai/cuvs/pull/276)) [@ajit283](https://github.com/ajit283)
- Go API - [WIP] ([#212](https://github.com/rapidsai/cuvs/pull/212)) [@ajit283](https://github.com/ajit283)

# cuvs 24.12.00 (11 Dec 2024)

## 🚨 Breaking Changes

- HNSW CPU Hierarchy ([#465](https://github.com/rapidsai/cuvs/pull/465)) [@divyegala](https://github.com/divyegala)
- Use dashes in cuvs-bench package name. ([#417](https://github.com/rapidsai/cuvs/pull/417)) [@bdice](https://github.com/bdice)

## 🐛 Bug Fixes

- Skip IVF-PQ packing test for lists with not enough data ([#512](https://github.com/rapidsai/cuvs/pull/512)) [@achirkin](https://github.com/achirkin)
- [BUG] Fix CAGRA filter ([#489](https://github.com/rapidsai/cuvs/pull/489)) [@enp1s0](https://github.com/enp1s0)
- Add `kIsSingleSource` to `PairwiseDistanceEpilogueElementwise` ([#485](https://github.com/rapidsai/cuvs/pull/485)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix include errors, header, and unsafe locks in iface.hpp ([#467](https://github.com/rapidsai/cuvs/pull/467)) [@achirkin](https://github.com/achirkin)
- Fix an OOB error in device-side cuvs::neighbors::refine and CAGRA kern_prune ([#460](https://github.com/rapidsai/cuvs/pull/460)) [@achirkin](https://github.com/achirkin)
- Put a ceiling on cuda-python ([#445](https://github.com/rapidsai/cuvs/pull/445)) [@bdice](https://github.com/bdice)
- Enable NVTX in cuvs-cagra-search component ([#439](https://github.com/rapidsai/cuvs/pull/439)) [@achirkin](https://github.com/achirkin)
- BUG: CAGRA multi-cta illegal access with bad queries ([#438](https://github.com/rapidsai/cuvs/pull/438)) [@achirkin](https://github.com/achirkin)
- Fix index overflow in edge cases of CAGRA graph optimize ([#435](https://github.com/rapidsai/cuvs/pull/435)) [@achirkin](https://github.com/achirkin)
- Fix correct call to brute force in generate groundtruth of cuvs-bench ([#427](https://github.com/rapidsai/cuvs/pull/427)) [@dantegd](https://github.com/dantegd)
- Use Python for sccache hit rate computation. ([#420](https://github.com/rapidsai/cuvs/pull/420)) [@bdice](https://github.com/bdice)
- Add `click` package to `cuvs-bench` conda recipe ([#408](https://github.com/rapidsai/cuvs/pull/408)) [@divyegala](https://github.com/divyegala)
- Fix NVTX annotations ([#400](https://github.com/rapidsai/cuvs/pull/400)) [@achirkin](https://github.com/achirkin)

## 📖 Documentation

- [Doc] Fix CAGRA search sample code ([#484](https://github.com/rapidsai/cuvs/pull/484)) [@enp1s0](https://github.com/enp1s0)
- Fix broken link in README.md references ([#473](https://github.com/rapidsai/cuvs/pull/473)) [@Azurethi](https://github.com/Azurethi)
- Adding tech stack to docs ([#448](https://github.com/rapidsai/cuvs/pull/448)) [@cjnolet](https://github.com/cjnolet)
- Fix Question Retrieval notebook ([#352](https://github.com/rapidsai/cuvs/pull/352)) [@lowener](https://github.com/lowener)

## 🚀 New Features

- Add C++ API scalar quantization ([#494](https://github.com/rapidsai/cuvs/pull/494)) [@mfoerste4](https://github.com/mfoerste4)
- HNSW CPU Hierarchy ([#465](https://github.com/rapidsai/cuvs/pull/465)) [@divyegala](https://github.com/divyegala)
- Add serialization API to brute-force ([#461](https://github.com/rapidsai/cuvs/pull/461)) [@lowener](https://github.com/lowener)
- Add Question Retrieval notebook using Milvus ([#451](https://github.com/rapidsai/cuvs/pull/451)) [@lowener](https://github.com/lowener)
- Migrate feature diff for NN Descent from RAFT to cuVS ([#421](https://github.com/rapidsai/cuvs/pull/421)) [@divyegala](https://github.com/divyegala)
- Add --no-lap-sync cmd option to ann-bench ([#405](https://github.com/rapidsai/cuvs/pull/405)) [@achirkin](https://github.com/achirkin)
- Add `InnerProduct` and `CosineExpanded` metric support in NN Descent ([#177](https://github.com/rapidsai/cuvs/pull/177)) [@divyegala](https://github.com/divyegala)

## 🛠️ Improvements

- Update cuvs to match raft&#39;s cutlass changes ([#516](https://github.com/rapidsai/cuvs/pull/516)) [@vyasr](https://github.com/vyasr)
- add a README for wheels ([#504](https://github.com/rapidsai/cuvs/pull/504)) [@jameslamb](https://github.com/jameslamb)
- Move check_input_array from pylibraft ([#474](https://github.com/rapidsai/cuvs/pull/474)) [@benfred](https://github.com/benfred)
- use different wheel-size thresholds based on CUDA version ([#469](https://github.com/rapidsai/cuvs/pull/469)) [@jameslamb](https://github.com/jameslamb)
- Modify cuvs-bench to be able to generate ground truth in CPU systems ([#466](https://github.com/rapidsai/cuvs/pull/466)) [@dantegd](https://github.com/dantegd)
- enforce wheel size limits, README formatting in CI ([#464](https://github.com/rapidsai/cuvs/pull/464)) [@jameslamb](https://github.com/jameslamb)
- Moving spectral embedding and kernel gramm APIs to cuVS ([#463](https://github.com/rapidsai/cuvs/pull/463)) [@cjnolet](https://github.com/cjnolet)
- Migrate sparse knn and distances code from raft ([#457](https://github.com/rapidsai/cuvs/pull/457)) [@benfred](https://github.com/benfred)
- Don&#39;t presume pointers location infers usability. ([#441](https://github.com/rapidsai/cuvs/pull/441)) [@robertmaynard](https://github.com/robertmaynard)
- call `enable_testing` in root CMakeLists.txt ([#437](https://github.com/rapidsai/cuvs/pull/437)) [@robertmaynard](https://github.com/robertmaynard)
- CAGRA tech debt: distance descriptor and workspace memory ([#436](https://github.com/rapidsai/cuvs/pull/436)) [@achirkin](https://github.com/achirkin)
- Add ci run_ scripts needed for build infra ([#434](https://github.com/rapidsai/cuvs/pull/434)) [@robertmaynard](https://github.com/robertmaynard)
- Use environment variables in cache hit rate computation. ([#422](https://github.com/rapidsai/cuvs/pull/422)) [@bdice](https://github.com/bdice)
- Use dashes in cuvs-bench package name. ([#417](https://github.com/rapidsai/cuvs/pull/417)) [@bdice](https://github.com/bdice)
- We need to enable the c_api by default ([#416](https://github.com/rapidsai/cuvs/pull/416)) [@robertmaynard](https://github.com/robertmaynard)
- print sccache stats in builds ([#413](https://github.com/rapidsai/cuvs/pull/413)) [@jameslamb](https://github.com/jameslamb)
- make conda installs in CI stricter ([#406](https://github.com/rapidsai/cuvs/pull/406)) [@jameslamb](https://github.com/jameslamb)
- Ivf c example ([#404](https://github.com/rapidsai/cuvs/pull/404)) [@abner-ma](https://github.com/abner-ma)
- Prune workflows based on changed files ([#392](https://github.com/rapidsai/cuvs/pull/392)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- [WIP] Add pinned memory resource to C API ([#311](https://github.com/rapidsai/cuvs/pull/311)) [@ajit283](https://github.com/ajit283)
- Dynamic Batching ([#261](https://github.com/rapidsai/cuvs/pull/261)) [@achirkin](https://github.com/achirkin)

# cuvs 24.10.00 (9 Oct 2024)

## 🐛 Bug Fixes

- Use 64 bit types for dataset size calculation in CAGRA graph optimizer ([#380](https://github.com/rapidsai/cuvs/pull/380)) [@tfeher](https://github.com/tfeher)
- Remove EXPLICIT_INSTANTIATE_ONLY macros ([#358](https://github.com/rapidsai/cuvs/pull/358)) [@achirkin](https://github.com/achirkin)
- Fix order of operations for cosine IVF Flat ([#329](https://github.com/rapidsai/cuvs/pull/329)) [@lowener](https://github.com/lowener)
- Exclude any kernel symbol that uses cutlass ([#314](https://github.com/rapidsai/cuvs/pull/314)) [@benfred](https://github.com/benfred)
- [Fix] pin raft dependent to rapidsai ([#299](https://github.com/rapidsai/cuvs/pull/299)) [@rhdong](https://github.com/rhdong)
- Fix dataset dimension in IVF-PQ C wrappers ([#292](https://github.com/rapidsai/cuvs/pull/292)) [@tfeher](https://github.com/tfeher)
- Fix python ivf-pq for int8/uint8 dtypes ([#271](https://github.com/rapidsai/cuvs/pull/271)) [@benfred](https://github.com/benfred)
- FP16 API for CAGRA and IVF-PQ ([#264](https://github.com/rapidsai/cuvs/pull/264)) [@tfeher](https://github.com/tfeher)

## 📖 Documentation

- More doc updates for 24.10 ([#396](https://github.com/rapidsai/cuvs/pull/396)) [@cjnolet](https://github.com/cjnolet)
- fix 404 in documentation link in readme ([#395](https://github.com/rapidsai/cuvs/pull/395)) [@benfred](https://github.com/benfred)
- Improving getting started materials ([#342](https://github.com/rapidsai/cuvs/pull/342)) [@cjnolet](https://github.com/cjnolet)
- Fix broken examples link in README. ([#326](https://github.com/rapidsai/cuvs/pull/326)) [@bdice](https://github.com/bdice)
- Recommend `miniforge` for conda install ([#325](https://github.com/rapidsai/cuvs/pull/325)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- Port remaining scripts to `cuvs_bench` ([#368](https://github.com/rapidsai/cuvs/pull/368)) [@divyegala](https://github.com/divyegala)
- [Feat] Relative change with `bitset` API feature #2439 in raft ([#350](https://github.com/rapidsai/cuvs/pull/350)) [@rhdong](https://github.com/rhdong)
- cuvs_bench plotting functions ([#347](https://github.com/rapidsai/cuvs/pull/347)) [@dantegd](https://github.com/dantegd)
- CosineExpanded Metric for IVF-PQ (normalize inputs) ([#346](https://github.com/rapidsai/cuvs/pull/346)) [@tarang-jain](https://github.com/tarang-jain)
- Python API for CAGRA+HNSW ([#246](https://github.com/rapidsai/cuvs/pull/246)) [@divyegala](https://github.com/divyegala)
- C API for CAGRA+HNSW ([#240](https://github.com/rapidsai/cuvs/pull/240)) [@divyegala](https://github.com/divyegala)
- SNMG ANN ([#231](https://github.com/rapidsai/cuvs/pull/231)) [@viclafargue](https://github.com/viclafargue)
- [FEA] Support for half-float mixed precise in brute-force ([#225](https://github.com/rapidsai/cuvs/pull/225)) [@rhdong](https://github.com/rhdong)

## 🛠️ Improvements

- Remove cuvs-cagra-search from cuvs_static link ([#388](https://github.com/rapidsai/cuvs/pull/388)) [@benfred](https://github.com/benfred)
- Add a static library for cuvs ([#382](https://github.com/rapidsai/cuvs/pull/382)) [@benfred](https://github.com/benfred)
- Put the ann-bench large_workspace_resource in managed memory ([#372](https://github.com/rapidsai/cuvs/pull/372)) [@achirkin](https://github.com/achirkin)
- Add multigpu kmeans fit function ([#348](https://github.com/rapidsai/cuvs/pull/348)) [@benfred](https://github.com/benfred)
- Update update-version.sh to use packaging lib ([#344](https://github.com/rapidsai/cuvs/pull/344)) [@AyodeAwe](https://github.com/AyodeAwe)
- remove NCCL pins in build and test environments ([#341](https://github.com/rapidsai/cuvs/pull/341)) [@jameslamb](https://github.com/jameslamb)
- Vamana/DiskANN index build ([#339](https://github.com/rapidsai/cuvs/pull/339)) [@bkarsin](https://github.com/bkarsin)
- Use CI workflow branch &#39;branch-24.10&#39; again ([#331](https://github.com/rapidsai/cuvs/pull/331)) [@jameslamb](https://github.com/jameslamb)
- fix style checks on Python 3.12 ([#328](https://github.com/rapidsai/cuvs/pull/328)) [@jameslamb](https://github.com/jameslamb)
- Update flake8 to 7.1.1. ([#327](https://github.com/rapidsai/cuvs/pull/327)) [@bdice](https://github.com/bdice)
- Add function for calculating the mutual_reachability_graph ([#323](https://github.com/rapidsai/cuvs/pull/323)) [@benfred](https://github.com/benfred)
- Simplify libcuvs conda recipe. ([#322](https://github.com/rapidsai/cuvs/pull/322)) [@bdice](https://github.com/bdice)
- Refactor dependencies.yaml to use depends-on pattern. ([#321](https://github.com/rapidsai/cuvs/pull/321)) [@bdice](https://github.com/bdice)
- Update Python versions in cuvs_bench pyproject.toml. ([#318](https://github.com/rapidsai/cuvs/pull/318)) [@bdice](https://github.com/bdice)
- Brute force knn tile size heuristic ([#316](https://github.com/rapidsai/cuvs/pull/316)) [@mfoerste4](https://github.com/mfoerste4)
- Euclidean distance example ([#315](https://github.com/rapidsai/cuvs/pull/315)) [@abner-ma](https://github.com/abner-ma)
- Migrate trustworthiness and silhouette_score stats from RAFT ([#313](https://github.com/rapidsai/cuvs/pull/313)) [@benfred](https://github.com/benfred)
- Add support for Python 3.12 ([#312](https://github.com/rapidsai/cuvs/pull/312)) [@jameslamb](https://github.com/jameslamb)
- Add `managed` option for RMM Pool memory resource to C API ([#305](https://github.com/rapidsai/cuvs/pull/305)) [@ajit283](https://github.com/ajit283)
- Update rapidsai/pre-commit-hooks ([#303](https://github.com/rapidsai/cuvs/pull/303)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Expose search function with pre-filter for ANN ([#302](https://github.com/rapidsai/cuvs/pull/302)) [@lowener](https://github.com/lowener)
- Drop Python 3.9 support ([#301](https://github.com/rapidsai/cuvs/pull/301)) [@jameslamb](https://github.com/jameslamb)
- Use CUDA math wheels ([#298](https://github.com/rapidsai/cuvs/pull/298)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Remove NumPy &lt;2 pin ([#297](https://github.com/rapidsai/cuvs/pull/297)) [@seberg](https://github.com/seberg)
- CAGRA - separable compilation for distance computation ([#296](https://github.com/rapidsai/cuvs/pull/296)) [@achirkin](https://github.com/achirkin)
- Updating example notebooks ([#294](https://github.com/rapidsai/cuvs/pull/294)) [@cjnolet](https://github.com/cjnolet)
- Add RMM Pool memory resource to C API ([#285](https://github.com/rapidsai/cuvs/pull/285)) [@ajit283](https://github.com/ajit283)
- Update pre-commit hooks ([#283](https://github.com/rapidsai/cuvs/pull/283)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Improve update-version.sh ([#282](https://github.com/rapidsai/cuvs/pull/282)) [@bdice](https://github.com/bdice)
- Use tool.scikit-build.cmake.version, set scikit-build-core minimum-version ([#280](https://github.com/rapidsai/cuvs/pull/280)) [@jameslamb](https://github.com/jameslamb)
- Add cuvs_bench.run python code and build ([#279](https://github.com/rapidsai/cuvs/pull/279)) [@dantegd](https://github.com/dantegd)
- Add cuvs-bench to dependencies and conda environments ([#275](https://github.com/rapidsai/cuvs/pull/275)) [@dantegd](https://github.com/dantegd)
- Update pip devcontainers to UCX v1.17.0 ([#262](https://github.com/rapidsai/cuvs/pull/262)) [@jameslamb](https://github.com/jameslamb)
- Adding example for tuning build and search params using Optuna ([#257](https://github.com/rapidsai/cuvs/pull/257)) [@dpadmanabhan03](https://github.com/dpadmanabhan03)
- Fixed link to build docs and corrected ivf_flat_example ([#255](https://github.com/rapidsai/cuvs/pull/255)) [@mmccarty](https://github.com/mmccarty)
- Merge branch-24.08 into branch-24.10 ([#254](https://github.com/rapidsai/cuvs/pull/254)) [@jameslamb](https://github.com/jameslamb)
- Persistent CAGRA kernel ([#215](https://github.com/rapidsai/cuvs/pull/215)) [@achirkin](https://github.com/achirkin)
- [FEA] Support for Cosine distance in IVF-Flat ([#179](https://github.com/rapidsai/cuvs/pull/179)) [@lowener](https://github.com/lowener)

# cuvs 24.08.00 (7 Aug 2024)

## 🚨 Breaking Changes

- Allow serialization on streams ([#173](https://github.com/rapidsai/cuvs/pull/173)) [@benfred](https://github.com/benfred)

## 🐛 Bug Fixes

- Remove fp16 kernels that have no public entry point ([#268](https://github.com/rapidsai/cuvs/pull/268)) [@tfeher](https://github.com/tfeher)
- Use `raft::util::popc(...)` public API ([#249](https://github.com/rapidsai/cuvs/pull/249)) [@divyegala](https://github.com/divyegala)
- Enable building FAISS main statically ([#241](https://github.com/rapidsai/cuvs/pull/241)) [@tarang-jain](https://github.com/tarang-jain)
- CAGRA bench: use device-side refinement when the data is on device ([#228](https://github.com/rapidsai/cuvs/pull/228)) [@achirkin](https://github.com/achirkin)
- Rename `.devcontainer`s for CUDA 12.5 ([#224](https://github.com/rapidsai/cuvs/pull/224)) [@jakirkham](https://github.com/jakirkham)
- Fix a CAGRA graph opt bug ([#192](https://github.com/rapidsai/cuvs/pull/192)) [@enp1s0](https://github.com/enp1s0)

## 📖 Documentation

- fix library name in docs (&#39;cuvs&#39; not &#39;pycuvs&#39;) ([#193](https://github.com/rapidsai/cuvs/pull/193)) [@jameslamb](https://github.com/jameslamb)

## 🚀 New Features

- Add cuvs_bench python folder, config files and constraints ([#244](https://github.com/rapidsai/cuvs/pull/244)) [@dantegd](https://github.com/dantegd)
- Add MST optimization to guarantee the connectivity of CAGRA graphs ([#237](https://github.com/rapidsai/cuvs/pull/237)) [@anaruse](https://github.com/anaruse)
- Moving over C++ API of CAGRA+hnswlib from RAFT ([#229](https://github.com/rapidsai/cuvs/pull/229)) [@divyegala](https://github.com/divyegala)
- [FEA] expose python &amp; C API for prefiltered brute force ([#174](https://github.com/rapidsai/cuvs/pull/174)) [@rhdong](https://github.com/rhdong)
- CAGRA new vector addition ([#151](https://github.com/rapidsai/cuvs/pull/151)) [@enp1s0](https://github.com/enp1s0)

## 🛠️ Improvements

- [Opt] introduce the `masked_matmul` to prefiltered brute force. ([#251](https://github.com/rapidsai/cuvs/pull/251)) [@rhdong](https://github.com/rhdong)
- Add more info to ANN_BENCH context ([#248](https://github.com/rapidsai/cuvs/pull/248)) [@achirkin](https://github.com/achirkin)
- split up CUDA-suffixed dependencies in dependencies.yaml ([#247](https://github.com/rapidsai/cuvs/pull/247)) [@jameslamb](https://github.com/jameslamb)
- Fix pinning to a different RAFT tag ([#235](https://github.com/rapidsai/cuvs/pull/235)) [@benfred](https://github.com/benfred)
- Use workflow branch 24.08 again ([#234](https://github.com/rapidsai/cuvs/pull/234)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- chore: update search_plan.cuh ([#232](https://github.com/rapidsai/cuvs/pull/232)) [@eltociear](https://github.com/eltociear)
- Enable kernel &amp; memcpy overlapping in IVF index building ([#230](https://github.com/rapidsai/cuvs/pull/230)) [@abc99lr](https://github.com/abc99lr)
- CAGRA: reduce argument count in select_and_run() kernel wrappers ([#227](https://github.com/rapidsai/cuvs/pull/227)) [@achirkin](https://github.com/achirkin)
- Mark the rust brute force unittest as flaky ([#226](https://github.com/rapidsai/cuvs/pull/226)) [@benfred](https://github.com/benfred)
- Add python bindings for ivf-* extend functions ([#220](https://github.com/rapidsai/cuvs/pull/220)) [@benfred](https://github.com/benfred)
- Build and test with CUDA 12.5.1 ([#219](https://github.com/rapidsai/cuvs/pull/219)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add col-major support for brute force knn ([#217](https://github.com/rapidsai/cuvs/pull/217)) [@benfred](https://github.com/benfred)
- Add CUDA_STATIC_MATH_LIBRARIES ([#216](https://github.com/rapidsai/cuvs/pull/216)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- skip CMake 3.30.0 ([#214](https://github.com/rapidsai/cuvs/pull/214)) [@jameslamb](https://github.com/jameslamb)
- Complete Migration of IVF Helpers / Features from RAFT ([#213](https://github.com/rapidsai/cuvs/pull/213)) [@tarang-jain](https://github.com/tarang-jain)
- Use verify-alpha-spec hook ([#209](https://github.com/rapidsai/cuvs/pull/209)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fixes for publishing rust package to crates.io ([#207](https://github.com/rapidsai/cuvs/pull/207)) [@benfred](https://github.com/benfred)
- Add rust example ([#206](https://github.com/rapidsai/cuvs/pull/206)) [@benfred](https://github.com/benfred)
- Adding IVF examples ([#203](https://github.com/rapidsai/cuvs/pull/203)) [@cjnolet](https://github.com/cjnolet)
- Fix compilation error when _CLK_BREAKDOWN is defined in cagra. ([#202](https://github.com/rapidsai/cuvs/pull/202)) [@jiangyinzuo](https://github.com/jiangyinzuo)
- DOC: update notebook link ([#191](https://github.com/rapidsai/cuvs/pull/191)) [@raybellwaves](https://github.com/raybellwaves)
- Change cagra.build_index to cagra.build ([#187](https://github.com/rapidsai/cuvs/pull/187)) [@benfred](https://github.com/benfred)
- Add python serialization API&#39;s for ivf-pq and ivf_flat ([#186](https://github.com/rapidsai/cuvs/pull/186)) [@benfred](https://github.com/benfred)
- resolve dependency-file-generator warning, rapids-build-backend followup ([#185](https://github.com/rapidsai/cuvs/pull/185)) [@jameslamb](https://github.com/jameslamb)
- Adopt CI/packaging codeowners ([#183](https://github.com/rapidsai/cuvs/pull/183)) [@raydouglass](https://github.com/raydouglass)
- Scaling workspace resources ([#181](https://github.com/rapidsai/cuvs/pull/181)) [@achirkin](https://github.com/achirkin)
- Remove text builds of documentation ([#180](https://github.com/rapidsai/cuvs/pull/180)) [@vyasr](https://github.com/vyasr)
- Add refine to the Python and C api&#39;s ([#175](https://github.com/rapidsai/cuvs/pull/175)) [@benfred](https://github.com/benfred)
- Allow serialization on streams ([#173](https://github.com/rapidsai/cuvs/pull/173)) [@benfred](https://github.com/benfred)
- Forward-merge branch-24.06 into branch-24.08 ([#169](https://github.com/rapidsai/cuvs/pull/169)) [@benfred](https://github.com/benfred)
- Use rapids-build-backend ([#145](https://github.com/rapidsai/cuvs/pull/145)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- ANN_BENCH ([#130](https://github.com/rapidsai/cuvs/pull/130)) [@achirkin](https://github.com/achirkin)
- Enable random subsampling ([#122](https://github.com/rapidsai/cuvs/pull/122)) [@tfeher](https://github.com/tfeher)

# cuvs 24.06.00 (5 Jun 2024)

## 🐛 Bug Fixes

- Fix CAGRA OOM handling ([#167](https://github.com/rapidsai/cuvs/pull/167)) [@tfeher](https://github.com/tfeher)
- Pass through raft static CMake var ([#111](https://github.com/rapidsai/cuvs/pull/111)) [@vyasr](https://github.com/vyasr)
- Fix rust docs build ([#84](https://github.com/rapidsai/cuvs/pull/84)) [@benfred](https://github.com/benfred)

## 📖 Documentation

- chore: update Doxyfile ([#162](https://github.com/rapidsai/cuvs/pull/162)) [@eltociear](https://github.com/eltociear)
- cuVS docs updates for release ([#161](https://github.com/rapidsai/cuvs/pull/161)) [@cjnolet](https://github.com/cjnolet)
- update: fix RAFT URL in README ([#91](https://github.com/rapidsai/cuvs/pull/91)) [@hurutoriya](https://github.com/hurutoriya)
- Update the developer&#39;s guide with new copyright hook ([#81](https://github.com/rapidsai/cuvs/pull/81)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add `lucene-cuvs` to integrations section of docs ([#73](https://github.com/rapidsai/cuvs/pull/73)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- Add `refine` to public API ([#154](https://github.com/rapidsai/cuvs/pull/154)) [@lowener](https://github.com/lowener)
- [FEA] support of prefiltered brute force ([#146](https://github.com/rapidsai/cuvs/pull/146)) [@rhdong](https://github.com/rhdong)
- Migrate IVF-Flat from RAFT ([#94](https://github.com/rapidsai/cuvs/pull/94)) [@divyegala](https://github.com/divyegala)
- Migrate IVF-PQ from RAFT to cuVS ([#86](https://github.com/rapidsai/cuvs/pull/86)) [@lowener](https://github.com/lowener)

## 🛠️ Improvements

- Expose serialization to the python / c-api ([#164](https://github.com/rapidsai/cuvs/pull/164)) [@benfred](https://github.com/benfred)
- Select k instantiations ([#159](https://github.com/rapidsai/cuvs/pull/159)) [@benfred](https://github.com/benfred)
- fix devcontainer name for codespaces ([#153](https://github.com/rapidsai/cuvs/pull/153)) [@trxcllnt](https://github.com/trxcllnt)
- Accept host_mdspan for IVF-PQ build and extend ([#148](https://github.com/rapidsai/cuvs/pull/148)) [@tfeher](https://github.com/tfeher)
- Add pairwise_distance api&#39;s for C, Python and Rust ([#142](https://github.com/rapidsai/cuvs/pull/142)) [@benfred](https://github.com/benfred)
- Changing RAFT_EXPLICT_* to CUVS_EXPLITI_* ([#141](https://github.com/rapidsai/cuvs/pull/141)) [@cjnolet](https://github.com/cjnolet)
- Speed-up rust build ([#138](https://github.com/rapidsai/cuvs/pull/138)) [@benfred](https://github.com/benfred)
- Removing `libraft.so` from libcuvs dependencies ([#132](https://github.com/rapidsai/cuvs/pull/132)) [@cjnolet](https://github.com/cjnolet)
- CAGRA API update and allow async host refinement ([#131](https://github.com/rapidsai/cuvs/pull/131)) [@mfoerste4](https://github.com/mfoerste4)
- Fix rust api docs ([#119](https://github.com/rapidsai/cuvs/pull/119)) [@benfred](https://github.com/benfred)
- Migrate BFKNN from raft ([#118](https://github.com/rapidsai/cuvs/pull/118)) [@benfred](https://github.com/benfred)
- Fix IVF-PQ helper functions ([#116](https://github.com/rapidsai/cuvs/pull/116)) [@lowener](https://github.com/lowener)
- Migrate `raft::cluster` to `cuvs::cluster` ([#115](https://github.com/rapidsai/cuvs/pull/115)) [@cjnolet](https://github.com/cjnolet)
- hide RAFT #pragma deprecation warnings ([#114](https://github.com/rapidsai/cuvs/pull/114)) [@trxcllnt](https://github.com/trxcllnt)
- Enable Warnings as errors in Python tests ([#102](https://github.com/rapidsai/cuvs/pull/102)) [@mroeschke](https://github.com/mroeschke)
- Remove libnvjitlink dependency. ([#97](https://github.com/rapidsai/cuvs/pull/97)) [@bdice](https://github.com/bdice)
- Migrate to `{{ stdlib(&quot;c&quot;) }}` ([#93](https://github.com/rapidsai/cuvs/pull/93)) [@hcho3](https://github.com/hcho3)
- update: replace to cuvs from RAFT in PULL_REQUEST_TEMPLATE ([#92](https://github.com/rapidsai/cuvs/pull/92)) [@hurutoriya](https://github.com/hurutoriya)
- Add python and rust bindings for Ivf-Pq ([#90](https://github.com/rapidsai/cuvs/pull/90)) [@benfred](https://github.com/benfred)
- add --rm and --name to devcontainer run args ([#89](https://github.com/rapidsai/cuvs/pull/89)) [@trxcllnt](https://github.com/trxcllnt)
- Update pip devcontainers to UCX v1.15.0 ([#88](https://github.com/rapidsai/cuvs/pull/88)) [@trxcllnt](https://github.com/trxcllnt)
- Remove gtest from dependencies.yaml ([#87](https://github.com/rapidsai/cuvs/pull/87)) [@robertmaynard](https://github.com/robertmaynard)
- Moving and renaming distance namespaces from raft -&gt; cuvs ([#85](https://github.com/rapidsai/cuvs/pull/85)) [@cjnolet](https://github.com/cjnolet)
- Use static gtest ([#83](https://github.com/rapidsai/cuvs/pull/83)) [@robertmaynard](https://github.com/robertmaynard)
- Add python and rust bindings for Ivf-Flat ([#82](https://github.com/rapidsai/cuvs/pull/82)) [@benfred](https://github.com/benfred)
- Forward merge branch-24.04 to branch-24.06 ([#80](https://github.com/rapidsai/cuvs/pull/80)) [@benfred](https://github.com/benfred)
- Update devcontainers to use cuda12.2 ([#72](https://github.com/rapidsai/cuvs/pull/72)) [@benfred](https://github.com/benfred)
- Forward merge branch-24.04 to branch-24.06 ([#71](https://github.com/rapidsai/cuvs/pull/71)) [@benfred](https://github.com/benfred)
- Enable forward-merger ops-bot plugin ([#70](https://github.com/rapidsai/cuvs/pull/70)) [@benfred](https://github.com/benfred)
- Adds missing files to `update-version.sh` ([#69](https://github.com/rapidsai/cuvs/pull/69)) [@AyodeAwe](https://github.com/AyodeAwe)
- Add Cagra-Q compression to the python and rust api&#39;s ([#68](https://github.com/rapidsai/cuvs/pull/68)) [@benfred](https://github.com/benfred)
- ConfigureCUDA.cmake now sets CUVS_ prefixed variables ([#66](https://github.com/rapidsai/cuvs/pull/66)) [@robertmaynard](https://github.com/robertmaynard)
- Enable all tests for `arm` jobs ([#63](https://github.com/rapidsai/cuvs/pull/63)) [@galipremsagar](https://github.com/galipremsagar)
