#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 09:24:51 2025

@author: lizamclatchy
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
all_filtered_data = pd.read_csv('/Users/lizamclatchy/RAPID_drinkingwater/CSVS/all_filtered_pairs_full.csv')
all_filtered_data.fillna(0, inplace=True) 

# Assuming all_filtered_data is already defined
all_filtered_data.set_index('Peak', inplace=True)
comp_data = []
chemicals = ["Total Trihalomethanes", "Trichloromethane (chloroform)", "Bromodichloromethane", "Dibromochloromethane", "Bromoform",
              'Tetrahydrofuran','2-Butanone (MEK)', 'Bromochloromethane', 'Carbon disulfide',
              'Chloroethene (vinyl chloride)','Chloromethane (methyl chloride)','Dibromomethane',
              'Methyl tert-butyl ether (MTBE)','Methylene chloride (DCM)']
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
    #comp_df_2 =  comp_df_2[~((comp_df_2['Tap Value'] == 0) & (comp_df_2['Filtered Value'] == 0))]

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
    comp_data.append(comp_df_2)
# Fill NaN values (if any) with 0
#comp_df_2["Sample Pair"] = comp_df_2["Tap Sample"] + " → " + comp_df_2["Filtered Sample"]
multi_comp_df = pd.concat(comp_data, ignore_index=True)

working_filters = multi_comp_df
working_filters["Sample Pair"] = working_filters["Tap Sample"] + " → " + working_filters["Filtered Sample"]
avg_reduction_df = working_filters.groupby("Sample Pair", as_index=False)["Percent Reduction"].mean()
avg_reduction_df = avg_reduction_df.sort_values(by="Percent Reduction", ascending=False)
avg_reduction_df["Pair Label"] = ["Pair " + str(i+1) for i in range(len(avg_reduction_df))]

all_sample_pairs = set(multi_comp_df["Sample Pair"].unique())

# Sample pairs that have average reduction calculated
reduction_sample_pairs = set(avg_reduction_df["Sample Pair"].unique())
# Find the missing ones (those in comp_df_2/multi_comp_df but not in avg_reduction_df)
missing_pairs = all_sample_pairs - reduction_sample_pairs

# Convert to a sorted list or DataFrame for easy inspection
missing_pairs_df = pd.DataFrame(sorted(list(missing_pairs)), columns=["Missing Sample Pair"])
print(missing_pairs_df)



percent_of = avg_reduction_df[avg_reduction_df['Percent Reduction'] > 10 ]
plt.figure(figsize=(6, 4))  # smaller size for compactness

# Use seaborn histogram for better styling
sns.histplot(avg_reduction_df['Percent Reduction'], bins=50, kde=False, color='steelblue', edgecolor='black')

# Add mean and median lines
plt.axvline(avg_reduction_df['Percent Reduction'].mean(), color='green', linestyle='--', label='Mean')
plt.axvline(avg_reduction_df['Percent Reduction'].median(), color='red', linestyle='-.', label='Median')

# Axis and title formatting
plt.xlabel("Percent Reduction (%)", fontsize=10)
plt.ylabel("Frequency", fontsize=10)
plt.title("Distribution of Average Percent Reduction", fontsize=12)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.grid(False)
plt.legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 40))


sns.barplot(
    data=avg_reduction_df,
    x="Percent Reduction",
    y="Pair Label",
    dodge=True
)

plt.xlabel("Percent Reduction (%)")
plt.ylabel("Sample Pair")
plt.title("Avg Percent Reduction for All Detected Chemicals per Pair")
plt.xlim(-110, 110)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()




plt.figure(figsize=(12, working_filters["Sample Pair"].nunique() * 0.25))


sns.barplot(
    data=working_filters,
    x="Percent Reduction",
    y="Sample Pair",
    hue="Chemical",
    dodge=True
)

plt.xlabel("Percent Reduction (%)")
plt.ylabel("Sample Pair")
plt.title("Grouped Percent Reduction by Sample Pair and Chemical")
plt.xlim(-110, 110)
plt.legend(title="Chemical", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
