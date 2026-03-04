"""
generate_dataset.py - Generates synthetic telecom churn dataset (7,000 rows)
with multiple numerical/categorical features and realistic missing values.

Usage:
    python src/generate_dataset.py
    python src/generate_dataset.py --rows 10000 --out data/telecom_customer_churn.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 7000

def generate(n_rows=7000, out_path="data/telecom_customer_churn.csv"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n = n_rows

    genders    = np.random.choice(["Male","Female"], n)
    ages       = np.random.randint(18, 85, n)
    married    = np.random.choice(["Yes","No"], n, p=[0.52,0.48])
    dependents = np.random.choice([0,1,2,3], n, p=[0.65,0.20,0.10,0.05])
    cities     = np.random.choice(["Los Angeles","San Francisco","San Diego",
                                    "Sacramento","Fresno","Long Beach","Oakland",
                                    "Bakersfield","Anaheim","Riverside"], n)
    zip_codes  = np.random.randint(90001, 96162, n)
    lats       = np.round(np.random.uniform(32.5, 42.0, n), 6)
    lons       = np.round(np.random.uniform(-124.5,-114.1, n), 6)
    referrals  = np.random.randint(0, 12, n)
    tenure     = np.random.randint(1, 73, n)
    offers     = np.random.choice(["None","Offer A","Offer B","Offer C","Offer D","Offer E"],
                                   n, p=[0.30,0.15,0.20,0.12,0.13,0.10])
    phone_svc  = np.random.choice(["Yes","No"], n, p=[0.90,0.10])
    avg_ld     = np.round(np.random.uniform(0, 50, n), 2)
    multi_line = np.random.choice(["Yes","No"], n, p=[0.45,0.55])
    internet   = np.random.choice(["Yes","No"], n, p=[0.82,0.18])
    inet_type  = np.where(internet=="Yes",
                          np.random.choice(["Fiber Optic","Cable","DSL"], n, p=[0.50,0.30,0.20]),
                          "None")
    gb_down    = np.where(internet=="Yes", np.round(np.random.uniform(5,200,n),1), 0)

    def svc(p=0.45):
        return np.where(internet=="Yes", np.random.choice(["Yes","No"], n, p=[p,1-p]), "No")

    online_sec=svc(0.35); online_bkp=svc(0.45); dev_protect=svc(0.40)
    tech_sup=svc(0.38);   stream_tv=svc(0.44);  stream_mov=svc(0.42)
    stream_mus=svc(0.30); unlim_data=svc(0.55)

    contracts  = np.random.choice(["Month-to-Month","One Year","Two Year"],n,p=[0.55,0.22,0.23])
    paperless  = np.random.choice(["Yes","No"], n, p=[0.60,0.40])
    payment    = np.random.choice(["Credit Card","Bank Withdrawal","Mailed Check"],n,p=[0.40,0.45,0.15])
    monthly    = np.round(np.random.uniform(20,120,n), 2)
    total_ch   = np.round(monthly * tenure * np.random.uniform(0.85,1.05,n), 2)
    refunds    = np.where(np.random.random(n)<0.08, np.round(np.random.uniform(5,50,n),2), 0.0)
    extra_data = np.random.choice([0,10,20,30], n, p=[0.75,0.12,0.08,0.05])
    total_ld   = np.round(avg_ld * tenure * np.random.uniform(0.9,1.1,n), 2)
    total_rev  = np.round(total_ch + extra_data + total_ld - refunds, 2)

    # Build probabilities row-by-row (guaranteed valid)
    status_list = []
    for i in range(n):
        cp = min(0.65, max(0.05,
             0.05
             + 0.35*(contracts[i]=="Month-to-Month")
             - 0.15*(contracts[i]=="Two Year")
             + 0.002*monthly[i]
             - 0.003*tenure[i]
             + 0.05*(online_sec[i]=="No")
             - 0.02*(tech_sup[i]=="Yes")))
        jp = 0.35 if tenure[i] <= 3 else 0.04
        sp = max(0.01, 1.0 - cp - jp)
        tot = cp + jp + sp
        p   = [cp/tot, jp/tot, sp/tot]
        status_list.append(np.random.choice(["Churned","Joined","Stayed"], p=p))
    status = np.array(status_list)

    churn_cats = np.where(status=="Churned",
        np.random.choice(["Competitor","Dissatisfaction","Attitude","Price","Other"],
                         n, p=[0.35,0.28,0.12,0.18,0.07]), None)
    churn_rsns = np.where(status=="Churned",
        np.random.choice(["Competitor had better devices","Product dissatisfaction",
                          "Network reliability","Price too high","Service attitude"], n),
        None)

    df = pd.DataFrame({
        "Customer ID":                       [f"CUST-{i:05d}" for i in range(n)],
        "Gender":                            genders,
        "Age":                               ages,
        "Married":                           married,
        "Number of Dependents":              dependents,
        "City":                              cities,
        "Zip Code":                          zip_codes,
        "Latitude":                          lats,
        "Longitude":                         lons,
        "Number of Referrals":               referrals,
        "Tenure in Months":                  tenure,
        "Offer":                             offers,
        "Phone Service":                     phone_svc,
        "Avg Monthly Long Distance Charges": avg_ld,
        "Multiple Lines":                    multi_line,
        "Internet Service":                  internet,
        "Internet Type":                     inet_type,
        "Avg Monthly GB Download":           gb_down,
        "Online Security":                   online_sec,
        "Online Backup":                     online_bkp,
        "Device Protection Plan":            dev_protect,
        "Premium Tech Support":              tech_sup,
        "Streaming TV":                      stream_tv,
        "Streaming Movies":                  stream_mov,
        "Streaming Music":                   stream_mus,
        "Unlimited Data":                    unlim_data,
        "Contract":                          contracts,
        "Paperless Billing":                 paperless,
        "Payment Method":                    payment,
        "Monthly Charge":                    monthly,
        "Total Charges":                     total_ch,
        "Total Refunds":                     refunds,
        "Total Extra Data Charges":          extra_data,
        "Total Long Distance Charges":       total_ld,
        "Total Revenue":                     total_rev,
        "Customer Status":                   status,
        "Churn Category":                    churn_cats,
        "Churn Reason":                      churn_rsns,
    })

    # Inject missing values (5-8% per column)
    for col, pct in {
        "Avg Monthly Long Distance Charges": 0.06,
        "Multiple Lines": 0.07, "Internet Type": 0.05,
        "Avg Monthly GB Download": 0.05, "Online Security": 0.06,
        "Online Backup": 0.05, "Device Protection Plan": 0.05,
        "Premium Tech Support": 0.06, "Streaming TV": 0.05,
        "Streaming Movies": 0.05, "Streaming Music": 0.07,
        "Total Charges": 0.03,
    }.items():
        mask = np.random.random(n) < pct
        df.loc[mask, col] = np.nan

    df.to_csv(out_path, index=False)
    print(f"Saved {n:,} rows -> {out_path}")
    print(f"  Columns       : {df.shape[1]}")
    print(f"  Missing cells : {df.isnull().sum().sum():,}")
    print(f"  Class balance :")
    print(df["Customer Status"].value_counts().to_string())
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=7000)
    parser.add_argument("--out",  default="data/telecom_customer_churn.csv")
    args = parser.parse_args()
    generate(n_rows=args.rows, out_path=args.out)
