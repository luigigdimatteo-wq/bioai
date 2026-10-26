from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

app = FastAPI(
    title="BioAI Computational Core API",
    description="Enterprise API engine for molecular descriptors, toxicity screening, and ADMET prediction.",
    version="1.0.0"
)

class MoleculeRequest(BaseModel):
    smiles: str

@app.get("/")
def health_check():
    return {"status": "online", "engine": "BioAI Core v1.0"}

@app.post("/api/v1/analyze")
def analyze_molecule(req: MoleculeRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if not mol:
        raise HTTPException(status_code=400, detail="Invalid SMILES string provided.")
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hda = Lipinski.NumHAcceptors(mol)
    hdd = Lipinski.NumHDonors(mol)
    
    return {
        "smiles": req.smiles,
        "molecular_weight": round(mw, 2),
        "logp": round(logp, 2),
        "h_bond_acceptors": hda,
        "h_bond_donors": hdd,
        "lipinski_compliant": bool(mw <= 500 and logp <= 5.0 and hda <= 10 and hdd <= 5)
    }
