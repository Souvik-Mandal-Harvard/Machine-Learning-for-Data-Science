# Beginner Guide: Creating Python Virtual Environment (macOS, Windows, Linux) for Your ML Projects using Jupyter Notebook

This guide walks beginner-level learners through creating and using a Python virtual environment named `ML_Venv`.

You can follow every step using only your command line:

- **Terminal** on macOS or Linux
- **PowerShell** or **Command Prompt** on Windows

No code editor is required.

---

## What is a virtual environment and why use one?

A virtual environment is an isolated copy of Python that is separate from the one installed on your computer. Packages you install inside it do not affect anything else on your system, and packages installed outside it do not interfere with your project.

This keeps your project self-contained and makes it easier to share or reproduce later.

---

## Before you start

You only need Python 3 installed on your computer.

To check, open your command line and run:

```bash
python --version
```

If that does not work, try:

```bash
python3 --version
```

If neither command works, install Python 3 from [https://www.python.org/downloads](https://www.python.org/downloads), then run the check again.

---

## 1) Go to your project folder

Open your command line and navigate to the folder where you want to keep your environment and project files. Use the `cd` command (short for "change directory") to move between folders.

Example on macOS or Linux:

```bash
cd ~/Documents/Machine-Learning-for-Data-Science
```

Example on Windows:

```powershell
cd C:\Users\YourName\Documents\Machine-Learning-for-Data-Science
```

Replace the path with the actual location of your project folder. After running `cd`, your prompt will show that you are now inside that folder.

---

## 2) Create the virtual environment

On macOS or Linux:

```bash
python3 -m venv ML_Venv
```

On Windows:

```powershell
python -m venv ML_Venv
```

This creates a new folder named `ML_Venv` inside your current directory. That folder holds a self-contained Python environment for this project. You only need to do this once.

---

## 3) Activate the environment

You must activate the environment every time you start a new terminal session.

On macOS or Linux:

```bash
source ML_Venv/bin/activate
```

On Windows PowerShell:

```powershell
.\ML_Venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
ML_Venv\Scripts\activate.bat
```

When activation succeeds, you will see `(ML_Venv)` at the start of your prompt. This tells you that any Python commands or package installs will now apply to this environment only.

---

## 4) Upgrade pip

`pip` is Python's built-in package installer. Keeping it up to date avoids compatibility warnings. Run:

```bash
python -m pip install --upgrade pip
```

---

## 5) Install Jupyter Notebook support

With `ML_Venv` active, install the two packages needed to run `.ipynb` notebooks using this environment:

```bash
python -m pip install ipykernel notebook
```

What each package is for:

- `ipykernel`: allows `ML_Venv` to be used as the Python kernel that runs your notebook cells.
- `notebook`: the Jupyter Notebook application you launch from the terminal to open and work with `.ipynb` files.

---

## 6) Register the environment as a Jupyter kernel

This step makes `ML_Venv` available as a selectable kernel when you open a notebook. Run:

```bash
python -m ipykernel install --user --name ML_Venv --display-name "Python (ML_Venv)"
```

You only need to do this once.

---

## 7) Install your ML packages

With `ML_Venv` still active, install the course libraries:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels plotly
```

What each package is for:

- `numpy`: arrays, matrix operations, and numerical computing.
- `pandas`: tables, data cleaning, and data analysis.
- `matplotlib`: basic charts and plots.
- `seaborn`: statistical plots with cleaner default styling.
- `scikit-learn`: machine learning models, preprocessing, and evaluation.
- `scipy`: scientific computing and statistical functions.
- `statsmodels`: classical statistics and regression analysis.
- `plotly`: interactive visualizations.

You can install additional packages at any time by running `python -m pip install <package-name>` while the environment is active.

---

## 8) Launch Jupyter Notebook and open your file

### 8.1. On your web browser
From the terminal, with `ML_Venv` active, run:

```bash
jupyter notebook
```

This will start the Jupyter Notebook server and open a browser window showing your project folder. From there:

1. Click on any `.ipynb` file to open it.
2. In the top menu, go to **Kernel** → **Change kernel** and select **Python (ML_Venv)**.

This ensures your notebook runs using the `ML_Venv` environment you just set up.

To stop the server when you are done, go back to the terminal and press `Ctrl + C`.

### 8.2. How to use Jupyter Notebook in VS Code

8.2.1. Open the notebook in VS Code, then click the kernel selector (top-right of the notebook).
    - Click on "Jupyter Kernel"
    - Choose **ML_Venv**, if you see it.

8.2.2. If you do not see `ML_Venv` in the list, go back and check "Python Environments`.
    - If you still don't see ML_Venv, run an interactive cell printing `sys.executable` / `sys.version` to confirm the interpreter in use.
    - Try the next step.

8.2.3. Tell VS Code where to look: VS Code automatically scans standard folders (like .venv or envs) but it does not scan custom folders like `ML_Venv` by default. To make sure environments in this folder always show up in the future:

    - Press `Cmd + ,` to open Settings.
	- Search for "Python: Venv Folders".
	- Add the parent path of your environments (where you saved the folder containing the virtual environment): `/path/to/ML_Venv`
	- Restart VS Code. It will now automatically discover any new venvs you create inside that folder.
---

## 9) Verify the installation

In a Python cell of your notebook, run:

```python
import numpy as np
import pandas as pd
import sklearn

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
```

If three version numbers print without errors, your environment is ready.

---

## 10) Everyday workflow

Each time you start working on this project:

1. Open your terminal or command line.
2. Navigate to the project folder with `cd`.
3. Activate `ML_Venv`.
4. Run `jupyter notebook` to open the notebook interface in your browser.
5. Open your `.ipynb` file and confirm the kernel is set to **Python (ML_Venv)**.

As long as `(ML_Venv)` is visible in your terminal prompt, you are working inside the environment.

---

## 11) Deactivate when finished

To stop the Jupyter server, press `Ctrl + C` in the terminal. Then, to leave the virtual environment and return to your normal system Python, run:

```bash
deactivate
```

This returns you to your system Python.

---

## Common beginner issues

### `python` or `python3` is not recognized

Python is either not installed or not in your system's PATH.

Install Python 3 from [https://www.python.org/downloads](https://www.python.org/downloads). During installation on Windows, check the box that says **"Add Python to PATH"**.

### Activation is blocked on Windows PowerShell

By default, PowerShell may prevent scripts from running. To allow it, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Close and reopen PowerShell, then try activating again.

### `Python (ML_Venv)` does not appear in the kernel list

This means the kernel was not registered, or was registered before `ipykernel` was installed.

1. Confirm `(ML_Venv)` is showing in your prompt.
2. Re-run the install command from step 5.
3. Re-run the registration command from step 6.
4. Restart the Jupyter Notebook server.

### Packages were installed but imports fail in the notebook

This usually means the packages were installed outside `ML_Venv`, or the wrong kernel is selected.

1. Check that the notebook kernel is set to **Python (ML_Venv)**.
2. Confirm `(ML_Venv)` is showing in your terminal prompt.
3. Re-run the install command from step 7.
4. Re-run the verify cell from step 9.
