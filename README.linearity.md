# Sensor Linearity Analysis Tool

**Author:** Dale Ghent  
**License:** MIT License  

This tool analyzes the linearity of a camera sensor by processing a directory of FITS files. It computes mean pixel values, filters saturated frames, performs a least-squares linear fit, calculates R², and generates an annotated plot. The output provides insight into how well the sensor response tracks exposure time under various settings.

---

## Features

- Calculates least-squares linear regression and R² across the provided images
- Computes mean ADU for each image. Mean ADUs are averaged together if multiple images of the same exposure time are provided
- Excludes images with a mean ADU above a certain amount from the fitting
- Supports exposure time filtering (`--min-exp` and `--max-exp`)
- Annotates percentage deviation from linearity
- Outputs a high-resolution plot in PNG format

---

## Usage

```bash
python linearity.py [-h] [-o OUTPUT_DIR] [-s SATURATION] [--min-exp MIN] [--max-exp MAX] directory
```

### Required Argument

- `directory`: Path to directory containing `.fits`, `.fit`, `.fts`, or `.fz` files.

### Optional Arguments

| Argument           | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `-o`, `--output-dir` | Output directory for the graph (default: current directory)                 |
| `-s`, `--saturation` | Maximum mean ADU value to include in the linear fit (default: 65000)       |
| `--min-exp`          | Minimum exposure time (in seconds) to include                              |
| `--max-exp`          | Maximum exposure time (in seconds) to include                              |

---

## Virtual Environment Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
```

### 2. Activate it

- macOS/Linux:
  ```bash
  source venv/bin/activate
  ```
- Windows:
  ```cmd
  venv\Scripts\activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

You can create `requirements.txt` with the following content:

```text
numpy
matplotlib
astropy
```

---

## Example

```bash
python linearity.py ./test_data -s 64000 --min-exp 0.5 --max-exp 5 -o ./results
```

This will:
- Analyze FITS files in `./test_data`
- Exclude saturated frames above 64000 ADU from fitting, but still plots them
- Include only exposure times between 0.5 and 5 seconds
- Save the plot to `./results`

---

## Output

- A `.png` file of a plot showing:
  - Mean ADU vs. exposure time
  - Least-squares linear fit
  - Annotated % deviation from linearity
  - R² value
- Console output of statistics and sample counts

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions, improvements, or bug reports:  
**Dale Ghent** – [daleg@elemental.org](mailto:daleg@elemental.org)

