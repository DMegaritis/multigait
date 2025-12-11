# MGait: Real-World Gait Pipeline for Wrist-Worn Devices

MGait (Multimorbidity Gait) is a Python implementation of a gait analysis pipeline for real-world assessment, primarily developed for wrist-worn inertial measurement units (IMUs). This pipeline integrates signal-processing algorithms for gait detection, initial contact detection, and stride-length estimation, providing a streamlined workflow for mobility research in people with multiple long-term conditions (multimorbidity). While the pipeline is wrist-focused, fine-tuned versions for lower-back devices are also provided for use in custom workflows.

The individual algorithms included in this library have been developed and validated in multimorbidity cohorts [1]. However, the full integration of these algorithms into a complete pipeline has not yet been formally evaluated, so we do not currently recommend any specific pipeline configuration. In future releases, we plan to systematically assess combinations of algorithms and provide recommended pipelines tailored to different multimorbidity clusters.

Planned future releases include:
- Fully validated pipelines for different multimorbidity clusters.
- Novel algorithm implementations.
- Validated wear-time detection algorithms.

Note on biomechanical definitions
: The biomechanical logic and gait event definitions implemented in MGait are based on the specifications defined within Mobilise-D.

---

Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Pipeline Use](#pipeline-use)
  - [Specific Algorithms](#specific-algorithms)
- [Citation](#citation)
- [Validation reference for the underlying algorithms](#validation-reference-for-the-underlying-algorithms)
- [Funding and Support](#funding-and-support)
- [License](#license)

---

## Installation

Install MGait directly from GitHub:

```bash
pip install git+https://github.com/DMegaritis/mgait.git
```

Or clone the repository and install locally:

```bash
git clone https://github.com/DMegaritis/mgait.git
cd mgait
pip install .
```

---

## Usage

The package is designed to be used in two main modes:

### Pipeline Use

High-level pipelines allow loading raw IMU data and obtaining gait outcomes end-to-end. Example:

```python
from mgait.pipeline.mgait_pipeline import MGaitPipelineSuggested

pipeline = MGaitPipelineSuggested()
pipeline.safe_run(long_trial)
```

Note: At this stage we offer a suggested pipeline with the best-performing algorithms from [1], but the full pipeline has not been fully validated yet. Please interpret results cautiously.

Additional note: While this library has been primarily developed as a complete gait pipeline for wrist-worn devices, it also includes support for lower-back algorithms. Fine-tuned versions for lower-back devices are provided and can be used in custom pipelines.

### Specific Algorithms

You can also use individual algorithms separately to build custom workflows. Example modules include:
- Gait Sequence Detection (GSD)
- Initial Contact Detection (ICD)
- Stride Length Estimation (SL)

For usage examples and input/output formats, see the examples in this repository or in DMegaritis/multimobility_wrist.

---

## Citation

If you use MGait in your research, please cite:

```bibtex
@software{megaritis2025wristmobility,
  author    = {Megaritis, Dimitrios and Alcock, Lisa and Scott, Kirsty and Hiden, Hugo and Cereatti, Andrea and Vogiatzis, Ioannis and Del Din, Silvia},
  title     = {MGait: Real-World Gait Pipeline for Wrist-Worn Devices for Multimorbid Populations},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {xyz},
  url       = {xyz}
}
```

## Validation reference for the underlying algorithms
[1] Megaritis, D., Alcock, L., Scott, K., Hiden, H., Cereatti, A., Vogiatzis, I., & Del Din, S. (2025). Real-World Wrist-Derived Digital Mobility Outcomes in People with Multiple Long-Term Conditions: A Comparison of Algorithms. Bioengineering, 12(10), 1108. https://doi.org/10.3390/bioengineering12101108

---

## Funding and Support

This work was supported by the Medical Research Council (MRC) Gap Fund award (UKRI/MR/B000091/1).

---

## License

The MGait library is licensed under the Apache License 2.0. It is free to use for any purpose, including commercial use, but all distributions must include the license text.
