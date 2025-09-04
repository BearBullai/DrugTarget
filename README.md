# AI-Powered Drug Target Platform

A comprehensive platform for identifying and prioritizing drug targets, predicting protein structures, detecting binding pockets, and designing nanocarriers for targeted drug delivery.

## Features

- **Data Processing**: Import and preprocess differential gene expression data
- **Target Prioritization**: Score and rank potential drug targets
- **Structure Prediction**: Predict protein 3D structures using ColabFold/AF2
- **Pocket Detection**: Identify and analyze potential binding pockets
- **Ligand Docking**: Perform virtual screening and molecular docking
- **Nanocarrier Design**: Recommend nanocarrier formulations
- **Reporting**: Generate comprehensive PDF reports

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd project
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your DEG list (Excel or CSV) in the `data/` directory or upload it through the web interface.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your web browser and navigate to the provided local URL (usually http://localhost:8501)

## Project Structure

```
project/
├── data/                   # Input data files (DEG lists, etc.)
├── targets/                # Processed target information
├── structures/             # Predicted protein structures
├── pockets/                # Detected binding pockets
├── ligands/                # Ligand data and docking results
├── nanodelivery/           # Nanocarrier design specifications
├── notebooks/              # Jupyter notebooks for analysis
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Data Requirements

The input data file should contain at least the following columns:
- `gene`: Gene symbol
- `logFC`: Log2 fold change
- `adj.P.Val`: Adjusted p-value

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the web framework
- ColabFold/AlphaFold2 for structure prediction
- RDKit for cheminformatics
- Plotly for interactive visualizations
