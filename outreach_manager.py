import pandas as pd
import os

def manage_outreach():
    if not os.path.exists("investor_pipeline.csv"):
        print("❌ Error: investor_pipeline.csv not found.")
        return
        
    df = pd.read_csv("investor_pipeline.csv")
    
    print("==================================================")
    print("BIOAI THERAPEUTICS — INVESTOR OUTREACH DASHBOARD")
    print("==================================================")
    for idx, row in df.iterrows():
        print(f"\n[{idx+1}] Fund: {row['Fund Name']} ({row['Stage']})")
        print(f"    Focus  : {row['Focus']}")
        print(f"    Contact: {row['Contact Person']} <{row['Email']}>")
        print(f"    Status : {row['Status']} [Priority: {row['Priority']}]")
        print(f"    --- Tailored Pitch Snippet ---")
        print(f"    Subject: BioAI: Live Enterprise SaaS Platform for AI-Driven Drug Discovery")
        print(f"    Hi {row['Contact Person'].split()[0]}, building on our work in {row['Focus']}...")
    print("\n==================================================")
    print("💡 Tip: Use this manager to track weekly cold outreach waves.")

if __name__ == "__main__":
    manage_outreach()
