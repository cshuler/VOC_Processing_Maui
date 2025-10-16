#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 09:54:29 2025

@author: lizamclatchy
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
##### ALL FILTER DATA
all_filtered_data = pd.read_csv('/Users/lizamclatchy/RAPID_drinkingwater/CSVS/all_filtered_pairs_full.csv')
all_filtered_data.set_index('Peak', inplace=True)
comp_data_1 = []
chemicals = ["Total Trihalomethanes", "Trichloromethane (chloroform)", "Bromodichloromethane", "Dibromochloromethane", "Bromoform"]
              # 'Tetrahydrofuran','2-Butanone (MEK)', 'Bromochloromethane', 'Carbon disulfide',
              # 'Chloroethene (vinyl chloride)','Chloromethane (methyl chloride)','Dibromomethane',
              # 'Methyl tert-butyl ether (MTBE)','Methylene chloride (DCM)']
for chemical in chemicals:
    data = all_filtered_data.loc[chemical]
    columns = list(data.index)  # Maintain column order
    
    comparisons = []
    pending_taps = []  # Store consecutive taps
    
    # Iterate through columns while maintaining order
    for i, col in enumerate(columns):
        if "(Tap)" in col:
            pending_taps.append(col)  # Store taps until a filtered sample appears
        elif "(Filtered)" in col:
            if pending_taps:  # Ensure there are taps before matching
                # Match all stored taps to this filtered sample
                for tap_col in pending_taps:
                    comparisons.append((tap_col, col, data[tap_col], data[col]))
    
                # Continue checking for additional filtered samples (preserve original behavior)
                last_tap = pending_taps[-1]  # Keep last tap for additional matches
                
                for j in range(i + 1, len(columns)):
                    if "(Filtered)" in columns[j]:
                        comparisons.append((last_tap, columns[j], data[last_tap], data[columns[j]]))
                    elif "(Tap)" in columns[j]:  # Stop when a new tap appears
                        break
    
                # Reset tap buffer after processing all available filters
                pending_taps = []

    # Convert to DataFrame
    comp_df_2 = pd.DataFrame(comparisons, columns=["Tap Sample", "Filtered Sample", "Tap Value", "Filtered Value"])
    comp_df_2[["Tap Value", "Filtered Value"]] = comp_df_2[["Tap Value", "Filtered Value"]].apply(pd.to_numeric)
    comp_df_2 =  comp_df_2[~((comp_df_2['Tap Value'] == 0) & (comp_df_2['Filtered Value'] == 0))]

    comp_df_2['Difference'] = comp_df_2['Tap Value'] - comp_df_2['Filtered Value']
    def calculate_percent_reduction(row):
        tap = row['Tap Value']
        filtered = row['Filtered Value']
        
        if tap == 0 and filtered == 0:
            return 0.0  # No change
        elif tap == 0 and filtered > 0:
            return -100.0  # Went from 0 to something → "increase"
        else:
            return ((tap - filtered) / tap) * 100
    
    comp_df_2['Percent Reduction'] = comp_df_2.apply(calculate_percent_reduction, axis=1)
    comp_df_2["Chemical"] = chemical
    comp_data_1.append(comp_df_2)
# Fill NaN values (if any) with 0
multi_comp_df = pd.concat(comp_data_1, ignore_index=True)

working_filters = multi_comp_df[multi_comp_df['Percent Reduction'] > -5]
working_filters["Sample Pair"] = working_filters["Tap Sample"] + " → " + working_filters["Filtered Sample"]

avg_reduction_df = working_filters.groupby("Sample Pair", as_index=False)["Percent Reduction"].mean()
avg_reduction_df = avg_reduction_df.sort_values(by="Percent Reduction", ascending=False)
avg_reduction_df["Pair Label"] = ["Pair " + str(i+1) for i in range(len(avg_reduction_df))]





pairs_with_filtration = pd.read_csv('/Users/lizamclatchy/RAPID_drinkingwater/CSVS/filtered_pairs_data_full.csv')
pairs_with_filtration.fillna(0, inplace=True) 
filtration = pairs_with_filtration[pairs_with_filtration['Peak'] == 'Filtration']
filtration = filtration.replace(
    ['Carbon,Carbon', 'Carbon and Carbon', 'Carbon, Carbon', 'Charcoal', 'Micron Carbon'],
    'Carbon'
)
pairs_with_filtration = pairs_with_filtration[pairs_with_filtration['Peak'] != "Filtration"]
measurement_cols = pairs_with_filtration.columns[1:]  # Exclude first column (chemical names)
pairs_with_filtration[measurement_cols] = pairs_with_filtration[measurement_cols].apply(pd.to_numeric, errors='coerce')
pairs_with_filtration = pd.concat([pairs_with_filtration, filtration], ignore_index=True)

#Calculate the difference for certain chemicals, while maintaining the filtration mechanism
#Let's try TTHMs
comp_data = []
pairs_with_filtration.set_index('Peak', inplace=True)
chemicals = ["Total Trihalomethanes", "Trichloromethane (chloroform)", "Bromodichloromethane", "Dibromochloromethane", "Bromoform"]
              # 'Tetrahydrofuran','2-Butanone (MEK)', 'Bromochloromethane', 'Carbon disulfide',
              # 'Chloroethene (vinyl chloride)','Chloromethane (methyl chloride)','Dibromomethane',
              # 'Methyl tert-butyl ether (MTBE)','Methylene chloride (DCM)']
for chemical in chemicals:
    data = pairs_with_filtration.loc[chemical]
    tap_samples = {col: val for col, val in data.items() if "(Tap)" in col}
    filtered_samples = {col: val for col, val in data.items() if "(Filtered)" in col}
    print(tap_samples)
    #Match taps with corresponding filters based on order
    comparisons = []
    columns = list(data.index)  # Maintain column order
    
    for i, tap_col in enumerate(columns):
        if "(Tap)" in tap_col:
            tap_val = data[tap_col]
    
            # Find the next filtered sample(s)
            matched_filters = []
            for j in range(i + 1, len(columns)):
                if "(Tap)" in columns[j]:  # Skip if another tap is directly after
                    break
                if "(Filtered)" in columns[j]:  # Collect filtered matches
                    matched_filters.append(columns[j])
    
            # Store comparisons
            for filt_col in matched_filters:
                comparisons.append((tap_col, filt_col, tap_val, data[filt_col]))

    comp_df = pd.DataFrame(comparisons, columns=["Tap Sample", "Filtered Sample", "Tap Value", "Filtered Value"])
    comp_df[["Tap Value", "Filtered Value"]] = comp_df[["Tap Value", "Filtered Value"]].apply(pd.to_numeric)
    comp_df =  comp_df[~((comp_df['Tap Value'] == 0) & (comp_df['Filtered Value'] == 0))]

    # Compute difference and filtration type
    comp_df['Difference'] = comp_df['Tap Value'] - comp_df['Filtered Value']
    def calculate_percent_reduction(row):
        tap = row['Tap Value']
        filtered = row['Filtered Value']
        
        if tap == 0 and filtered == 0:
            return 0.0  # No change
        elif tap == 0 and filtered > 0:
            return -100.0  # Went from 0 to something → "increase"
        else:
            return ((tap - filtered) / tap) * 100
    
    comp_df['Percent Reduction'] = comp_df.apply(calculate_percent_reduction, axis=1)
    
    filtration_row = pairs_with_filtration.loc['Filtration']
    comp_df["Filtration Type"] = comp_df["Filtered Sample"].map(filtration_row).fillna("Unknown").str.strip()

    # Append chemical name
    comp_df["Chemical"] = chemical
    comp_data.append(comp_df)

    comp_df.fillna(0, inplace=True)
multi_comp_df_1 = pd.concat(comp_data, ignore_index=True)
#filtered_comp_df = multi_comp_df_1[multi_comp_df_1["Filtration Type"].isin(["Reverse Osmosis", "Carbon"])]
multi_comp_df_1["Sample Pair"] = multi_comp_df_1["Tap Sample"] + " → " + multi_comp_df_1["Filtered Sample"]
filtered_comp_df["Sample Pair"] = filtered_comp_df["Tap Sample"] + " → " + filtered_comp_df["Filtered Sample"]
avg_reduction_df_1 = multi_comp_df_1.groupby(["Sample Pair", "Filtration Type"], as_index=False)["Percent Reduction"].mean()
avg_reduction_df_1 = avg_reduction_df_1.sort_values(by="Percent Reduction", ascending=False)
## All data cleaned and loaded

all_data = avg_reduction_df["Percent Reduction"]
ro_data = avg_reduction_df_1.loc[avg_reduction_df_1["Filtration Type"] == "Reverse Osmosis", "Percent Reduction"]
carbon_data = avg_reduction_df_1.loc[avg_reduction_df_1["Filtration Type"] == "Carbon", "Percent Reduction"]

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 8), sharex=True)
fig.suptitle("Mean Percent Reduction of Trihalomethanes by Filtration Type", fontsize=14, fontweight='bold', y=0.98)

# Color scheme
colors = {
    "All": "gray",
    "Reverse Osmosis": "steelblue",
    "Carbon": "darkorange"
}

# Histogram for ALL
sns.histplot(all_data, bins=15, kde=False, color=colors["All"], edgecolor='black', ax=axes[0])
axes[0].axvline(all_data.mean(), color='green', linestyle='--', label='Mean')
axes[0].axvline(all_data.median(), color='red', linestyle='-.', label='Median')
axes[0].text(
    0.02, 0.95, f"All Filters (n={len(all_data)})",
    transform=axes[0].transAxes,
    fontsize=12, fontweight='bold',
    va='center', ha='left'
)
axes[0].set_ylabel("Frequency", fontsize=12)
axes[0].legend(fontsize=8, loc='center left')
axes[0].tick_params(axis='both', labelsize=8)

# Histogram for RO
sns.histplot(ro_data, bins=15, kde=False, color=colors["Reverse Osmosis"], edgecolor='black', ax=axes[1])
axes[1].axvline(ro_data.mean(), color='green', linestyle='--', label='Mean')
axes[1].axvline(ro_data.median(), color='red', linestyle='-.', label='Median')
axes[1].text(
    0.02, 0.95, f"Reverse Osmosis Filters (n={len(ro_data)})",
    transform=axes[1].transAxes,
    fontsize=12, fontweight='bold',
    va='center', ha='left'
)
axes[1].set_ylabel("Frequency", fontsize=12)
axes[1].legend(fontsize=8, loc='center left')
axes[1].tick_params(axis='both', labelsize=8)

# Histogram for Carbon
sns.histplot(carbon_data, bins=25, kde=False, color=colors["Carbon"], edgecolor='black', ax=axes[2])
axes[2].axvline(carbon_data.mean(), color='green', linestyle='--', label='Mean')
axes[2].axvline(carbon_data.median(), color='red', linestyle='-.', label='Median')
axes[2].text(
    0.02, 0.95,"Carbon Filters (n=8)" ,
    transform=axes[2].transAxes,
    fontsize=12, fontweight='bold',
    va='center', ha='left'
)
axes[2].set_xlabel("Percent Reduction (%)", fontsize=14)
axes[2].set_ylabel("Frequency", fontsize=12)
axes[2].legend(fontsize=8, loc='center left')
axes[2].tick_params(axis='both', labelsize=8)

# Global formatting
plt.xlim(-17, 100.5)
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)  # spacing between plots
#plt.savefig("filter_reduction_histograms.png", dpi=300, bbox_inches='tight')
plt.show()
