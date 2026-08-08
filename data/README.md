# Workflow and resource data

This directory contains the inputs needed by the deterministic ADARE simulator.

- `workflows/`: source workflow instances, including Pegasus-style DAX/XML files and WfCommons JSON instances.
- `benchmarks/`: normalized task/dependency JSON consumed by the experiment runners.
- `environment/`: heterogeneous Edge/Fog/Cloud node parameters.
- `build/`: conversion utilities from source workflow formats to the local benchmark schema.

The Pegasus workflow families are associated with the Pegasus Workflow Gallery and the WfCommons instances identify WfCommons in their metadata. Converted benchmark JSON files are derived representations produced by the scripts in `data/build/`.

The repository's MIT license applies to ADARE's original code. It does not relicense third-party workflow datasets. Users redistributing or repurposing the workflow files should verify the terms and citation requirements of their original providers:

- Pegasus: https://pegasus.isi.edu/workflow_gallery/
- WfCommons: https://wfcommons.org/

Generated caches and conversion history belong under `data/history/`, which is excluded from Git.
