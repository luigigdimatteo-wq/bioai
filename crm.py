import pandas as pd

def update_investor_crm():
    data = [
        {
            "Fund Name": "Speedinvest",
            "Focus": "DeepTech & Industrial Tech (Europe)",
            "Stage": "Pre-Seed / Seed",
            "Contact Person": "DeepTech Partner",
            "Email": "deeptech@speedinvest.com",
            "Status": "Ready for Outreach",
            "Priority": "High"
        },
        {
            "Fund Name": "Eurazeo Science & Innovation",
            "Focus": "Life Sciences & AI Therapeutics",
            "Stage": "Seed / Series A",
            "Contact Person": "BioTech Investment Team",
            "Email": "biotech@eurazeo.com",
            "Status": "Ready for Outreach",
            "Priority": "High"
        },
        {
            "Fund Name": "468 Capital",
            "Focus": "B2B Software & AI Infrastructure",
            "Stage": "Pre-Seed / Seed",
            "Contact Person": "Founding Partner",
            "Email": "seed@468cap.com",
            "Status": "Ready for Outreach",
            "Priority": "Medium"
        },
        {
            "Fund Name": "Founders Fund",
            "Focus": "DeepTech & Breakthrough Science",
            "Stage": "Seed / Early Stage",
            "Contact Person": "Investment Associate",
            "Email": "dealflow@foundersfund.com",
            "Status": "Ready for Outreach",
            "Priority": "High"
        }
    ]
    
    df = pd.DataFrame(data)
    df.to_csv("investor_pipeline.csv", index=False)
    print("=== BIOAI THERAPEUTICS: EXPANDED INVESTOR PIPELINE ===")
    print(df.to_string(index=False))
    print("======================================================")

if __name__ == "__main__":
    update_investor_crm()
