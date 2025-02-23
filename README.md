# Sensitivity-Constrained Fourier Neural Operators (SC-FNO)

## Overview
This repository contains the implementation of **Sensitivity-Constrained Fourier Neural Operators (SC-FNO)**, a novel approach to improve neural operators for parametric differential equations of the form \(\frac{\partial u}{\partial t} = f(u, x, t, p)\). SC-FNO enhances the Fourier Neural Operator (FNO) by addressing challenges in inverse problems, sensitivity calculations (\(\frac{\partial u}{\partial p}\)), and concept drift using a sensitivity loss regularizer. It outperforms standard FNO and FNO with PINN regularization, offering superior accuracy in solution paths and parameter inversion, scalability (tested up to 82 parameters), and reduced training demands.

## Key Features
- High accuracy in forward and inverse problems
- Robustness to sparse data and complex parameter spaces
- Reduced training time and data requirements
- Applicable to various differential equations and neural operators

## Authors
- **Abdolmehdi Behroozi** (amb10399@psu.edu)  
  Department of Civil and Environmental Engineering, Penn State University  
- **Chaopeng Shen** (Corresponding author, cxs1024@psu.edu)  
  Department of Civil and Environmental Engineering, Penn State University  
- **Daniel Kifer** (duk17@psu.edu)  
  School of Electrical Engineering and Computer Science, Penn State University  

## Code and Models
Models, and selected experiment code will be released at:  
[https://github.com/AMBehroozi/SC_Neural_Operators](https://github.com/AMBehroozi/SC_Neural_Operators)

## Citation
This work is Published as a conference paper at ICLR 2025. Please cite as:  
Behroozi, A., Shen, C., & Kifer, D. (2025). Sensitivity-Constrained Fourier Neural Operators for Forward and Inverse Problems in Parametric Differential Equations. *ICLR 2025*.

## Requirements
- Python 3.x
- PyTorch
- Additional dependencies in `requirements.txt` (TBD)

## Usage
Instructions will be updated upon code release. Stay tuned!

## License
TBD upon release.