import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# Define the function for CDF 
def cdf(d, prop, bins):
    """
    Calculate the Cumulative Distribution Function (CDF) for the given property.
    """
    data = d[prop].values
    data = data[~np.isnan(data)]
    h, b = np.histogram(data, bins, density=True)
    cdf = np.cumsum(h * np.diff(b))
    return cdf, b

def consecutive(data):
    splits = np.split(data, np.where((np.diff(data) != 1) & (np.diff(data) != -1))[0] + 1)
    return [split for split in splits if len(split) > 1]

# Load and preprocess the data
df = pd.read_csv(r"PATH TO YOUR HISTORICAL DATA CSV FILE", usecols=[1, 2, 3, 4, 5, 6, 8, 12])
df.rename(columns={'surface_total_sky_direct': 'Solar'}, inplace=True)
df.drop(df[np.logical_and(df['Month'] == 2, df['Day'] == 29)].index, inplace=True)
df['Solar'] = df['Solar']  
df = df[df['Year'].between(1962, 1991)]

# Defining weights
weights = {
    'AirTemp_mean': 2/24, 'AirTemp_min': 1/24, 'AirTemp_max': 1/24,
    'Wind_mean': 2/24, 'Wind_max': 2/24,
    'Solar_sum': 12/24,
    'Tdp_mean': 2/24, 'Tdp_min': 1/24, 'Tdp_max': 1/24
}

# Define statistics for each parameter
agg_funcs = {
    'AirTemp': ['mean', 'min', 'max'],
    'Wind': ['mean', 'max'],
    'Solar': ['sum'],
    'Tdp': ['mean', 'min', 'max']
}

# Calculate short-term daily statistics for each parameter
short_term_daily_stats = df.groupby(['Year', 'Month', 'Day']).agg(agg_funcs).reset_index()
short_term_daily_stats.columns = [s[0] for s in short_term_daily_stats.columns.values[:3]] + ['_'.join(s) for s in short_term_daily_stats.columns.values[3:]]

candidate_years = {}

for month in range(1, 13):
    monthly_data = short_term_daily_stats[short_term_daily_stats['Month'] == month]
    n_bins = 25
    cdfs = {}
    fs = {}
    score = {year: 0 for year in monthly_data['Year'].unique()}
    
    for weight in weights:
        cdfs[weight] = {}
        fs[weight] = {}

        # Calculate the long term CDF for this weight
        cdfs[weight]['all'], bin_edges = cdf(monthly_data, weight, n_bins)
        x = bin_edges[:-1] + np.diff(bin_edges)/2
        # plt.figure(figsize=(10, 6))
        # plt.plot(x, cdfs[weight]['all'], 'ko-', label='Long-term CDF', linewidth=2)
        
        for year in monthly_data['Year'].unique():
            year_data = monthly_data[monthly_data['Year'] == year]

            # Calculate the CDF for this weight for specific year
            cdfs[weight][year], _ = cdf(year_data, weight, bin_edges)   
            
            # Finkelstein-Schafer statistic (difference between long term CDF and year CDF)
            fs[weight][year] = np.mean(np.abs(cdfs[weight]['all'] - cdfs[weight][year]))

            # Add weighted FS value to score for this year
            score[year] += fs[weight][year] * weights[weight]
            
            # Plot CDF for the specific year
        #     plt.plot(x, cdfs[weight][year], linestyle=':', label=f'Year {year}')

        # # Finalize plot for the current weight and month
        # plt.title(f'CDF Comparison for {weight} in Month {month}')
        # plt.xlabel(weight)
        # plt.ylabel('CDF')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
            
    # Rank the years in increasing order of their scores
    ranked_years = sorted(score, key=score.get)
    
    # Get the top 5 highest ranked years for the current month
    candidate_years[month] = ranked_years[:5]

# Calculating long-term daily percentile values for AirTemp and Solar parameter
percentiles = {}
for month in range(1, 13):
    temp_values = short_term_daily_stats[(short_term_daily_stats['Month'] == month)]['AirTemp_mean']
    solar_values = short_term_daily_stats[(short_term_daily_stats['Month'] == month)]['Solar_sum']
    percentiles[month] = {
            'AirTemp_67th': np.percentile(temp_values, 67),
            'AirTemp_33rd': np.percentile(temp_values, 33),
            'Solar_33rd': np.percentile(solar_values, 33)
        }

run_lengths = {}

for month, years in candidate_years.items():
    run_lengths[month] = {}
    for year in years:
        # Filter and sort data for the specific month and year
        filtered_data = short_term_daily_stats[(short_term_daily_stats['Year'] == year) & (short_term_daily_stats['Month'] == month)]
        temp_values = filtered_data[['Day', 'AirTemp_mean']].sort_values('AirTemp_mean')
        solar_values = filtered_data[['Day', 'Solar_sum']].sort_values('Solar_sum')
        
        sorted_days_temp = temp_values['Day'].values
        #print(sorted_days_temp)
        sorted_days_solar = solar_values['Day'].values
        sorted_temps = temp_values['AirTemp_mean'].values
        sorted_solar = solar_values['Solar_sum'].values
        
        airtemp_67th = percentiles[month]['AirTemp_67th']
        airtemp_33rd = percentiles[month]['AirTemp_33rd']
        solar_33rd = percentiles[month]['Solar_33rd']
       
        # Identify days for each condition
        above_67th_temp = sorted_days_temp[np.flatnonzero(sorted_temps > airtemp_67th)]
        below_33rd_temp = sorted_days_temp[np.flatnonzero(sorted_temps < airtemp_33rd)]
        below_33rd_solar = sorted_days_solar[np.flatnonzero(sorted_solar < solar_33rd)]
        
        # Calculate runs
        runs_above_67th_temp = consecutive(above_67th_temp)
        runs_below_33rd_temp = consecutive(below_33rd_temp)
        runs_below_33rd_solar = consecutive(below_33rd_solar)

        run_lengths[month][year] = {
            'Runs_Above_67th_AirTemp': [len(run) for run in runs_above_67th_temp],  # RUN LENGTHS ABOVE 67TH AirTemp for different runs
            'Runs_Below_33rd_AirTemp': [len(run) for run in runs_below_33rd_temp],
            'Frequency_Above_67th_AirTemp': len(runs_above_67th_temp),
            'Frequency_Below_33rd_AirTemp': len(runs_below_33rd_temp),
            'Runs_Below_33rd_Solar': [len(run) for run in runs_below_33rd_solar],
            'Frequency_Below_33rd_Solar': len(runs_below_33rd_solar),
            'Total_Frequency': sum(map(len, [runs_above_67th_temp, runs_below_33rd_temp, runs_below_33rd_solar])),
            'Max_Run_Length': max([len(run) for run in runs_above_67th_temp + runs_below_33rd_temp + runs_below_33rd_solar], default=0)
        }

final_tmy_selection = {}

for month, years in candidate_years.items():
    remaining_years = years.copy()

    # Most Runs Criterion
    total_frequencies = [run_lengths[month][year]['Total_Frequency'] for year in remaining_years]
    if total_frequencies.count(max(total_frequencies)) == 1:
        max_freq = max(total_frequencies)
        remaining_years = [year for year in remaining_years if run_lengths[month][year]['Total_Frequency'] < max_freq]
        
    if not remaining_years:
        print(f"No remaining years for month {month} after Most Runs Criterion")

    # Max Run Length Criterion
    if remaining_years:
        max_run_length = max(run_lengths[month][year]['Max_Run_Length'] for year in remaining_years)
        remaining_years = [year for year in remaining_years if run_lengths[month][year]['Max_Run_Length'] < max_run_length]
    if not remaining_years:
        print(f"No remaining years for month {month} after Max Run Length Criterion")

    # Zero-Run Criterion
    if remaining_years:
        remaining_years = [year for year in remaining_years if run_lengths[month][year]['Total_Frequency'] > 0]
    if not remaining_years:
        print(f"No remaining years for month {month} after Zero-Run Criterion")

    # If no remaining years, select the highest-ranked year from the original candidate list
    if not remaining_years:
        remaining_years = years

    # Select the first remaining candidate year in the ranking list
    final_tmy_selection[month] = remaining_years[0]

print(final_tmy_selection)

final_tmy_list = []

for month in final_tmy_selection:
    # For each year, add the candidate month
    final_tmy_list.append(df[np.logical_and(df['Year'] == final_tmy_selection[month], df['Month'] == month)])

final_tmy = pd.concat(final_tmy_list)
final_tmy.sort_values(['Month', 'Day', 'Hour'], ascending=[True, True, True], inplace=True)
final_tmy.reset_index(inplace=True, drop=True)

# Smoothen 6 hrs on either side
for month in final_tmy_selection:
    if month == 12: 
        continue  # Skip the end of December!
    last_hr_idx = final_tmy[final_tmy['Month'] == month].index[-1]
    interp_indx = np.arange(last_hr_idx - 6, last_hr_idx + 7)

    for parameter in agg_funcs.keys():
        interp_vals = final_tmy[parameter].iloc[interp_indx].to_numpy()
        
        interp_func = np.poly1d(np.polyfit(interp_indx, interp_vals, 3))
        interp_out = interp_func(interp_indx) 
        
        if parameter == 'Solar' or parameter == 'Wind':
            interp_out[interp_out < 0.0] = 0.0
        
        final_tmy.loc[interp_indx, parameter] = interp_out

# Save final TMY data to CSV
# final_tmy.to_csv('1962-1991_Chandigarh_tmy_data.csv', index=False)







