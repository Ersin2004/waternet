import pandas as pd
import numpy as np

def clean_excel_data(input_file, output_file=None):
    """
    Verwijdert rijen met lege waarden uit een Excel bestand
    
    Parameters:
    input_file (str): Pad naar het input Excel bestand
    output_file (str): Pad naar het output Excel bestand (optioneel)
    """
    
    # Lees het Excel bestand
    print(f"Bestand inlezen: {input_file}")
    df = pd.read_excel(input_file)
    
    print(f"Originele dataset:")
    print(f"- Aantal rijen: {len(df)}")
    print(f"- Aantal kolommen: {len(df.columns)}")
    
    # Toon informatie over lege waarden
    print(f"\nLege waarden per kolom:")
    missing_info = df.isnull().sum()
    for col, missing_count in missing_info.items():
        if missing_count > 0:
            print(f"- {col}: {missing_count} lege waarden")

    # Verwijder rijen waar belangrijke kolommen leeg zijn
    # (bijvoorbeeld Stofnaam, CAS-nummer, Waarde, Hoedanigheid.omschrijving)
    belangrijke_kolommen = ['Stofnaam', 'CAS-nummer', 'Waarde', 'Hoedanigheid.omschrijving']
    df_method = df.dropna(subset=belangrijke_kolommen, how='any')
    print(f"\nMETHODE 3 - Rijen waar belangrijke kolommen ({', '.join(belangrijke_kolommen)}) leeg zijn:")
    print(f"- Resterende rijen: {len(df_method)}")
    print(f"- Verwijderde rijen: {len(df) - len(df_method)}")

    # Kies welke methode je wilt gebruiken (standaard methode 1)
    cleaned_df = df_method
    
    # Optioneel: vervang lege strings en spaties door NaN en clean opnieuw
    cleaned_df = cleaned_df.replace(r'^\s*$', np.nan, regex=True)  # Vervang lege strings/spaties
    cleaned_df = cleaned_df.dropna(how='all')  # Verwijder weer volledig lege rijen
    
    print(f"\nFinale gecleande dataset:")
    print(f"- Aantal rijen: {len(cleaned_df)}")
    print(f"- Aantal kolommen: {len(cleaned_df.columns)}")
    
    # Toon eerste 10 rijen van gecleande data
    print(f"\n=== EERSTE 10 RIJEN VAN GECLEANDE DATA ===")
    belangrijke_cols = ['Stofnaam', 'CAS-nummer', 'Waarde', 'Eenheid', 'Norm', 'Hoedanigheid.omschrijving']
    print(cleaned_df[belangrijke_cols].head(10).to_string(index=False))
    
    # Zoek naar specifieke CAS-nummers
    gezochte_cas = ['7439-92-1', '7439-97-6', '7440-43-9', '7440-38-2']
    print(f"\n=== ZOEKEN NAAR SPECIFIEKE CAS-NUMMERS ===")
    
    # Maak een lijst om alle gevonden rijen te verzamelen
    zware_metalen_data = []
    
    for cas_nummer in gezochte_cas:
        gevonden = cleaned_df[cleaned_df['CAS-nummer'] == cas_nummer]
        if len(gevonden) > 0:
            print(f"\n CAS-nummer {cas_nummer} GEVONDEN:")
            print(f"   Stofnaam: {gevonden.iloc[0]['Stofnaam']}")
            print(f"   Aantal rijen: {len(gevonden)}")
            
            # Voeg gevonden rijen toe aan zware metalen data
            zware_metalen_data.append(gevonden)
            
            # Toon alle gevonden rijen voor deze stof
            cols_tonen = ['Stofnaam', 'Norm', 'Waarde', 'Eenheid', 'Hoedanigheid.omschrijving']
            print(gevonden[cols_tonen].to_string(index=False))
        else:
            print(f"\n CAS-nummer {cas_nummer} NIET GEVONDEN")
    
    # Combineer alle zware metalen data en sla op in apart bestand
    if zware_metalen_data:
        zware_metalen_df = pd.concat(zware_metalen_data, ignore_index=True)
        
        # Definieer kolommen om te verwijderen (administratief/duplicaten)
        kolommen_om_te_verwijderen = [
            'Norm.code',                          # Duplicaat van Norm
            'Norm.omschrijving',                  # Duplicaat van Norm  
            'Compartiment.code',                  # Duplicaat van Compartiment
            'Compartiment.omschrijving',          # Duplicaat van Compartiment
            'Waarde µg/l',                        # Duplicaat van Waarde
            'Geldigheid.begindatum',              # Meestal leeg
            'Geldigheid.einddatum',               # Meestal leeg
            'Compartimentsubgroep.code',          # Te gedetailleerd
            'Compartimentsubgroep.omschrijving',  # Te gedetailleerd
            'Normsubgroup.code',                  # Te gedetailleerd  
            'Normsubgroup.omschrijving'           # Te gedetailleerd
        ]
        
        # Verwijder alleen kolommen die bestaan
        bestaande_kolommen_om_te_verwijderen = [col for col in kolommen_om_te_verwijderen if col in zware_metalen_df.columns]
        zware_metalen_df_gefilterd = zware_metalen_df.drop(columns=bestaande_kolommen_om_te_verwijderen)
        
        print(f"\n🧹 KOLOMMEN OPGESCHOOND:")
        print(f"   Origineel: {len(zware_metalen_df.columns)} kolommen")
        print(f"   Gefilterd: {len(zware_metalen_df_gefilterd.columns)} kolommen")
        print(f"   Verwijderd: {', '.join(bestaande_kolommen_om_te_verwijderen)}")
        
        # Sla op als Excel
        zware_metalen_file = r"c:\Users\tess2\Downloads\waternet\Zware_metalen_normen.xlsx"
        zware_metalen_df_gefilterd.to_excel(zware_metalen_file, index=False)
        
        # Sla ook op als CSV
        zware_metalen_csv = r"c:\Users\tess2\Downloads\waternet\Zware_metalen_normen.csv"
        zware_metalen_df_gefilterd.to_csv(zware_metalen_csv, index=False, encoding='utf-8-sig')

        print(f"\n ZWARE METALEN DATA OPGESLAGEN:")
        print(f"   Excel: {zware_metalen_file}")
        print(f"   CSV: {zware_metalen_csv}")
        print(f"   Aantal rijen: {len(zware_metalen_df_gefilterd)}")
        print(f"   Stoffen: {', '.join(zware_metalen_df_gefilterd['Stofnaam'].unique())}")
    
    # Filter onnodige kolommen weg
    kolommen_om_te_verwijderen = [
        'Norm.code',                          # Duplicaat van Norm
        'Norm.omschrijving',                  # Duplicaat van Norm  
        'Compartiment.code',                  # Duplicaat van Compartiment
        'Compartiment.omschrijving',          # Duplicaat van Compartiment
        'Waarde µg/l',                        # Duplicaat van Waarde
        'Geldigheid.begindatum',              # Meestal leeg
        'Geldigheid.einddatum',               # Meestal leeg
        'Compartimentsubgroep.code',          # Te gedetailleerd
        'Compartimentsubgroep.omschrijving',  # Te gedetailleerd
        'Normsubgroup.code',                  # Te gedetailleerd  
        'Normsubgroup.omschrijving'           # Te gedetailleerd
    ]
    
    # Verwijder alleen kolommen die bestaan
    bestaande_kolommen_om_te_verwijderen = [col for col in kolommen_om_te_verwijderen if col in cleaned_df.columns]
    cleaned_df_gefilterd = cleaned_df.drop(columns=bestaande_kolommen_om_te_verwijderen)
    
    print(f"\n KOLOMMEN OPGESCHOOND:")
    print(f"   Origineel: {len(cleaned_df.columns)} kolommen")  
    print(f"   Gefilterd: {len(cleaned_df_gefilterd.columns)} kolommen")
    
    # Sla op als er een output bestand is opgegeven
    if output_file:
        # Maak ook een CSV versie
        csv_file = output_file.replace('.xlsx', '.csv')
        cleaned_df_gefilterd.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"Ook opgeslagen als CSV: {csv_file}")
    
    return cleaned_df_gefilterd

def interactive_clean():
    """Interactieve functie om te kiezen welke cleaning methode te gebruiken"""
    
    input_file = r"c:\Users\tess2\Downloads\waternet\Normen_stoffen_zoetwater.xlsx"
    
    # Lees eerst het bestand om info te tonen
    df = pd.read_excel(input_file)
    
    print("OVERZICHT VAN JE DATA:")
    print(f"Totaal aantal rijen: {len(df)}")
    print(f"Totaal aantal kolommen: {len(df.columns)}")
    
    # Toon lege waarden info
    missing_info = df.isnull().sum()
    print(f"\nKolommen met lege waarden:")
    for col, missing_count in missing_info.items():
        if missing_count > 0:
            percentage = (missing_count / len(df)) * 100
            print(f"- {col}: {missing_count} ({percentage:.1f}%)")
    

    print("Verwijder rijen waar belangrijke kolommen leeg zijn")
    
    output_file = r"c:\Users\tess2\Downloads\waternet\Normen_stoffen_zoetwater_cleaned.xlsx"
    cleaned_df = clean_excel_data(input_file, output_file)
    
    return cleaned_df

if __name__ == "__main__":
    # Voer de cleaning uit
    cleaned_data = interactive_clean()