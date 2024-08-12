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
