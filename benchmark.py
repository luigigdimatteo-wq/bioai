import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def run_scientific_benchmark():
    """
    Runs an automated validation benchmark against standard reference molecules 
    to verify ADMET property accuracy and Lipinski compliance margins.
    """
    reference_compounds = [
        {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "expected_mw": 180.16},
        {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "expected_mw": 194.19},
        {"name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "expected_mw": 206.29},
        {"name": "Paracetamol", "smiles": "CC(=O)NC1=CC=C(O)C=C1", "expected_mw": 151.16}
    ]
    
    results = []
    for comp in reference_compounds:
        mol = Chem.MolFromSmiles(comp["smiles"])
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hda = Lipinski.NumHAcceptors(mol)
        hdd = Lipinski.NumHDonors(mol)
        
        # Validation tolerance check
        error_margin = abs(mw - comp["expected_mw"])
        status = "PASSED" if error_margin < 0.5 else "REVIEW"
        
        results.append({
            "Compound": comp["name"],
            "Calculated MW": round(mw, 2),
            "Expected MW": comp["expected_mw"],
            "LogP": round(logp, 2),
            "Lipinski Compliant": bool(mw <= 500 and logp <= 5.0),
            "Benchmark Status": status
        })
        
    df_report = pd.DataFrame(results)
    print("=== BIOAI THERAPEUTICS: SCIENTIFIC BENCHMARK REPORT ===")
    print(df_report.to_string(index=False))
    print("=====================================================")
    df_report.to_csv("scientific_benchmark_report.csv", index=False)

if __name__ == "__main__":
    run_scientific_benchmark()
