# 🛡️ BioAI Therapeutics — Intellectual Property & Trade Secret Policy

## 1. Executive Summary
BioAI Therapeutics operates a proprietary computational chemistry and generative AI platform for preclinical drug discovery. All underlying algorithms, molecular descriptor pipeline configurations, and generative models are protected under trade secret laws and proprietary software licensing agreements.

## 2. IP Architecture & Code Decoupling
To ensure zero IP leak during external demonstrations and cloud hosting:
* **Microservice Isolation**: Heavy computational algorithms (RDKit descriptors, docking engines, stochastic generation) are decoupled into a containerized FastAPI backend.
* **Client Data Confidentiality**: Customer SMILES strings, uploaded batch CSVs, and generated lead molecules are processed in-memory and never stored or logged on third-party public servers.
* **API Security**: Endpoints are prepared for JWT token authentication and role-based access control (RBAC).

## 3. Ownership of AI-Generated Leads
* All novel molecular structures, scaffolds, and SMILES candidates synthesized by the BioAI De Novo Engine belong exclusively to BioAI Therapeutics or designated enterprise co-discovery partners as per specific licensing agreements.
* Prior-art cross-referencing is conducted automatically against ChEMBL and PubChem databases to ensure novelty and freedom to operate (FTO).

## 4. Institutional Due Diligence Assurance
This document serves as an institutional guarantee during Venture Capital technical review, confirming that the platform meets global enterprise standards for IP protection, software security, and regulatory compliance.

---
**BioAI Therapeutics Legal & IP Team**  
`ip@bioai-therapeutics.com` | `luigi@bioai.io`
