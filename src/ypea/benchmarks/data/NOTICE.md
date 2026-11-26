# CEC benchmark data

The files in this folder hold the official input data of two benchmark suites:

- `cec2017.npz`: the shift vectors, rotation matrices and shuffle data of the CEC 2017 competition on single objective bound constrained numerical optimization. Source: https://github.com/P-N-Suganthan/CEC2017-BoundContrained (`codes.rar`, folder `C version/input_data`).

- `cec2022.npz`: the same data for the CEC 2022 competition on single objective bound constrained numerical optimization. Source: https://github.com/P-N-Suganthan/2022-SO-BO (`CEC2022.zip`, folder `C-Code/input_data`).

The data was created by the competition organizers, and all credit for it goes to them:

- N. H. Awad, M. Z. Ali, J. J. Liang, B. Y. Qu and P. N. Suganthan, "Problem Definitions and Evaluation Criteria for the CEC 2017 Special Session and Competition on Single Objective Bound Constrained Real-Parameter Numerical Optimization", Technical Report, Nanyang Technological University, Singapore, 2016.

- A. Kumar, K. V. Price, A. W. Mohamed, A. A. Hadi and P. N. Suganthan, "Problem Definitions and Evaluation Criteria for the CEC 2022 Special Session and Competition on Single Objective Bound Constrained Numerical Optimization", Technical Report, Nanyang Technological University, Singapore, 2021.

The data is redistributed unchanged, in a compressed NumPy format. Each text file of the official `input_data` folder is one array named after the file without `.txt` (for example `M_5_D10` or `shift_data_21`), with the rows and columns of the file. Rotation matrices and shift vectors are stored as float64, and shuffle data as int64.

Please cite the technical report of a suite when you publish results obtained with it.
