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


