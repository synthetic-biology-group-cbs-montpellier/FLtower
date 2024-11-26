# FLtower

![Pytest](https://github.com/synthetic-biology-group-cbs-montpellier/FLtower/actions/workflows/test.yml/badge.svg?branch=main)

This Python project uses **conda** to manage dependencies with an **`environment.yml`** file (ensuring a reproducible and easy-to-setup environment).

---

## **1. Prerequisites**
Before starting, make sure:
1. **Conda** is installed on your system. If not, download it via [Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install).

2. You have access to a terminal or command prompt.


## **2. Installing the Environment**

### **a) Create the Environment**
1. Clone the project repository:
   ```bash
   git clone https://github.com/synthetic-biology-group-cbs-montpellier/FLtower.git
   cd FLtower
   ```

2. Create the conda environment from the **`environment.yml`** file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate fltower
   ```

4. Install FLtower package:
   ```bash
   pip install -e .
   ```


## **3. Usage**

With the environment activated (*`conda activate fltower`*), you can run FLtower script from your data folder (*`cd <path>`*) with just this keyword:
```bash
fltower
```

You can also specify your input data folder and where you want to save outputs:
```bash
fltower --input <your/in/data/path> --output <your/output/folder>
```


---

## 4. Deactivating and Removing the Environment

### a) Deactivate the Environment
To deactivate the currently active environment, run:
```bash
conda deactivate
```

### b) Remove the Environment
To completely remove the environment:
```bash
conda env remove -n my_project_env
```

---

## 5. Recreating or Updating the Environment

### a) Update an Existing Environment

If changes have been made to **`environment.yml`**, update the environment:
```bash
conda env update -f environment.yml
```

### b) Recreate the Environment
If needed, you can delete and recreate the environment:
1. Remove the existing environment:
   ```bash
   conda env remove -n my_project_env
   ```

2. Recreate it:
   ```bash
   conda env create -f environment.yml
   ```

---

## **Support**

If you encounter any issues or have questions, [open an issue in the GitHub repository](https://github.com/synthetic-biology-group-cbs-montpellier/FLtower/issues).


---

## **Contributor's section**

### Updating Dependencies

1. Add a New Dependency:
   ```bash
   conda install <package_name>
   ```
   or, for Python-only dependencies not available in conda:
   ```bash
   pip install <package_name>
   ```

2. Export the updated **`environment.yml`** file:
   ```bash
   conda env export -f environment.yml
   ```
