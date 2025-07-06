import pandas as pd
import tabula

PDF_FILE = "cabin_2023.pdf"  # άλλαξε αν έχει άλλο όνομα
XLS_FILE = "TIMES καμπινων 2023_ για βοηθο.xlsx"
OUT_CSV  = "cabin_prices_clean.csv"

pdf_tables = tabula.read_pdf(PDF_FILE, pages="all", lattice=True)
df_pdf = pd.concat(pdf_tables, ignore_index=True)

df_xls = pd.read_excel(XLS_FILE)
df = pd.concat([df_pdf, df_xls], ignore_index=True).drop_duplicates()

df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("(", "")
              .str.replace(")", "")
)
df.to_csv(OUT_CSV, index=False)
print(f"Saved → {OUT_CSV}, rows={len(df)}")