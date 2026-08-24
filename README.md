# MultiGait: Real-World Gait Pipeline for Wrist-Worn Devices

[![PyPI version](https://img.shields.io/pypi/v/multigait.svg)](https://pypi.org/project/multigait/)

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.22082062-blue)](https://doi.org/10.5281/zenodo.22082062)

MultiGait (Multimorbidity Gait) is a Python implementation of a gait analysis pipeline for real-world assessment, primarily developed for wrist-worn inertial measurement units (IMUs). This pipeline integrates signal-processing algorithms for gait detection, initial contact detection, stride-length estimation, cadence, and walking speed providing a streamlined workflow for mobility research in people with multiple long-term conditions (multimorbidity). While the pipeline is wrist-focused, fine-tuned versions for lower-back devices are also provided for use in custom workflows.

The individual algorithms included in this library have been developed and validated in multimorbidity cohorts [1]. 
Validation of the full wrist-worn device pipeline is provided in [2], which presents the recommended pipeline configuration and reports the associated errors and performance metrics.

Planned future releases include:
- Novel algorithm implementations.
- Validated wear-time detection algorithms.

Note on biomechanical definitions
: The biomechanical logic and gait event definitions implemented in MultiGait are based on the specifications defined within Mobilise-D.

Software developed by Dr Dimitrios Megaritis. Scientific authorship is listed in `CITATION.cff`.

---

Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Pipeline Use](#pipeline-use)
  - [Specific Algorithms](#specific-algorithms)
- [Citation](#citation)
- [Validation reference for the underlying algorithms](#validation-literature-for-the-underlying-wrist-worn-device-signal-processing-algorithms)
- [Validation reference for the complete wrist-device pipeline](#validation-literature-for-the-complete-wrist-worn-device-signal-processing-pipeline)
- [Funding and Support](#funding-and-support)
- [License](#license)

---

## Installation

From PyPI (recommended)
```bash
python3 -m pip install multigait
```

From GitHub:

```bash
pip install git+https://github.com/DMegaritis/multigait.git
```

Or clone the repository and install locally:

```bash
git clone https://github.com/DMegaritis/multigait.git
cd multigait
pip install .
```

---

## Usage

The package is designed to be used in two main modes:

### Pipeline Use

High-level pipelines allow loading raw IMU data and obtaining gait outcomes end-to-end. Example:

```python
from multigait.pipeline import MultiGaitPipelineMultimorbidityImpaired

pipeline = MultiGaitPipelineMultimorbidityImpaired()
pipeline.safe_run(long_trial)
```

Note: We offer the best performing pipeline for older adults with multimorbidity presented and validated in [2].

Additional note: While this library has been primarily developed as a complete gait pipeline for wrist-worn devices, it also includes support for lower-back algorithms. Fine-tuned versions for lower-back devices are provided and can be used in custom pipelines.

### Specific Algorithms

You can also use individual algorithms separately to build custom workflows. Example modules include:
- Gait Sequence Detection (GSD)
- Initial Contact Detection (ICD)
- Stride Length Estimation (SL)
- Cadence (CAD; requires prior GSD and ICD)
- Walking Speed (Ws; requires prior GSD, CAD, and SL)

For usage examples and input/output formats, see the examples in this repository or in DMegaritis/multimobility_wrist.

### Digital Mobility Outcomes (DMOs)

The following table summarizes all Digital Mobility Outcomes (DMOs) extracted by MultiGait, including walking bout types and the statistics calculated:

| Walking Bout   | DMO                                               | Statistic/Measure |
|----------------|---------------------------------------------------|-----------------|
| All WBs        | Number of walking bouts                           | Sum             |
| All WBs        | Total Walking Duration (min)                      | Sum             |
| All WBs        | Initial Contacts (n_raw)                          | Sum             |
| All WBs        | WB Duration (s)                                   | Mean            |
| All WBs        | Max WB Duration (s)                               | 90th percentile |
| All WBs        | WB Duration bout to bout variability (s)         | CV              |
| All WBs        | Cadence (steps/min)                               | Mean            |
| All WBs        | Stride Duration (s)                               | Mean            |
| All WBs        | Cadence bout to bout variability (steps/min)     | CV              |
| All WBs        | Stride Duration bout to bout variability (s)     | CV              |
| All WBs        | Cadence within bout variability (CV)             | Mean            |
| All WBs        | Stride Duration within bout variability (CV)     | Mean            |
| All WBs        | Cadence within bout variability (RMSSD)          | Mean            |
| All WBs        | Stride Duration within bout variability (RMSSD)  | Mean            |
| 10–30s WBs     | Number of walking bouts                           | Sum             |
| 10–30s WBs     | Walking Speed (m/s)                               | Mean            |
| 10–30s WBs     | Stride Length (m)                                 | Mean            |
| 10–30s WBs     | Walking Speed within bout variability (CV) (m/s) | Mean            |
| 10–30s WBs     | Stride Length within bout variability (CV) (m)   | Mean            |
| 10–30s WBs     | Walking Speed within bout variability (RMSSD) (m/s) | Mean          |
| 10–30s WBs     | Stride Length within bout variability (RMSSD) (m) | Mean           |
| >10s WBs       | Number of walking bouts                           | Sum             |
| >10s WBs       | Max Walking Speed (m/s)                           | 90th percentile |
| >30s WBs       | Number of walking bouts                           | Sum             |
| >30s WBs       | Walking Speed (m/s)                               | Mean            |
| >30s WBs       | Stride Length (m)                                 | Mean            |
| >30s WBs       | Cadence (steps/min)                               | Mean            |
| >30s WBs       | Stride Duration (s)                               | Mean            |
| >30s WBs       | Max Walking Speed (m/s)                           | 90th percentile |
| >30s WBs       | Max Cadence (steps/min)                            | 90th percentile |
| >30s WBs       | Walking Speed bout to bout variability (m/s)     | CV              |
| >30s WBs       | Stride Length bout to bout variability (m)       | CV              |
| >30s WBs       | Cadence within bout variability (CV) (steps/min) | Mean            |
| >30s WBs       | Stride Duration within bout variability (CV) (s) | Mean            |
| >30s WBs       | Walking Speed within bout variability (CV) (m/s) | Mean            |
| >30s WBs       | Stride Length within bout variability (CV) (m)   | Mean            |
| >30s WBs       | Cadence within bout variability (RMSSD) (steps/min) | Mean         |
| >30s WBs       | Stride Duration within bout variability (RMSSD) (s) | Mean         |
| >30s WBs       | Walking Speed within bout variability (RMSSD) (m/s) | Mean         |
| >30s WBs       | Stride Length within bout variability (RMSSD) (m) | Mean           |
| >60s WBs       | Number of walking bouts                           | Sum             |
| All WBs        | Alpha                                             | -               |

*CV: Coefficient of Variation; RMSSD: Root Mean Square of Successive Differences*

---

## Citation

If you use MultiGait in your research, please cite:

```bibtex
@software{megaritis2025wristmobility,
  author    = {Megaritis, Dimitrios and Alcock, Lisa and Scott, Kirsty and Hiden, Hugo and Cereatti, Andrea and Vogiatzis, Ioannis and Del Din, Silvia},
  title     = {MultiGait: Real-World Gait Pipeline for Wrist-Worn Devices for Multimorbid Populations},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {https://doi.org/10.5281/zenodo.17903930},
  url       = {https://zenodo.org/records/17903930}
}
```

## Validation literature for the underlying wrist-worn device signal-processing algorithms
[1] Megaritis D, Alcock L, Scott K, Hiden H, Cereatti A, Vogiatzis I, Del Din S. Real-World Wrist-Derived Digital Mobility Outcomes in People with Multiple Long-Term Conditions: A Comparison of Algorithms. Bioengineering (Basel). 2025 Oct 15;12(10):1108. doi: 10.3390/bioengineering12101108. PMID: 41155107; PMCID: PMC12561645.

## Validation literature for the complete wrist-worn device signal-processing pipeline
[2] Megaritis D, Alcock L, Scott K, Hiden H, Vogiatzis I, Del Din S. Validation of an Algorithmic Pipeline for Wrist-Worn Devices to Estimate Walking Speed in People with Multiple Long-Term Conditions. Sensors (Basel). 2026 Aug 5;26(15):4946. doi: 10.3390/s26154946. PMID: 42590721; PMCID: PMC13468760.

---

## Funding and Support

This work was supported by the Medical Research Council (MRC) Gap Fund award (UKRI/MR/B000091/1).

---

## License

The MultiGait library is licensed under the Apache License 2.0. It is free to use for any purpose, including commercial use, but all distributions must include the license text.


<img width="500" height="180" alt="Northumbria_University_Logo" src="https://github.com/user-attachments/assets/4572c9f6-ad7e-4d8b-a398-5bae0547666f" />
<img width="400" height="180" alt="Medical_Research_Council_logo svg" src="https://github.com/user-attachments/assets/bba73ebd-3e0b-4831-b8c0-e5215c0fe14a" />
<img width="400" height="180" alt="newcastle-university-logo-png_seeklogo-355206" src="https://github.com/user-attachments/assets/e7a05ab0-6458-4c86-b5d0-723b76f8ac66" />~~
