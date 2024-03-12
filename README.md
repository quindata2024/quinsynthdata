# Setting Up a Virtual Environment for Synthcity

This guide details how to create a virtual environment on your personal PC for running the Python framework Synthcity. Using a virtual environment helps isolate project dependencies and avoids conflicts with other Python installations on your system.

## Prerequisites

- A Python interpreter (version 3.6 or later is recommended). You can download it from [Python Downloads](https://www.python.org/downloads/).
- A virtual environment tool. Two popular options are:
  - `venv` (included in Python 3.3+): Use this if you prefer a built-in solution.
  - `virtualenv`: Install it using `pip install virtualenv` if you want more flexibility.

## Steps

### 1. Create a Virtual Environment

#### Using venv (Python 3.3+)

```bash
python -m venv synthcity_env


### 2. Activate the Virtual Environment

#### For Windows:
```bash
.\synthcity_env\Scripts\activate.bat

#### For macOS/Linux:
```bash
source synthcity_env/bin/activate

Your command prompt should now indicate that the virtual environment is active (e.g., (synthcity_env) before your prompt).

### 3. Install Synthcity
Within the activated virtual environment, use pip to install Synthcity:
```bash
pip install synthcity

### 4. Verify Installation
Once the installation is complete, you can verify it by running:
```bash
python -c "import synthcity; print(synthcity.__version__)"
