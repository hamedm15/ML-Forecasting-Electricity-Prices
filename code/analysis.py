import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *
from auxiliary import *
from statisticaltests import *


def combine_forecasts(data_file_path, naive_results_path, glear_results_path, lear24_results_path, dnn_results_path, calibration_windows):

    glear_df, global_og_df = get_data_lear(glear_results_path, calibration_windows)
    lear24_df, m24_og_df = get_data_lear(lear24_results_path, calibration_windows, extra_cols=False)
    dnn_df, dnn_og_df = get_data_dnn(dnn_results_path, calibration_windows)

    naive_df = get_naive_results(data_file_path)
    naive_df = naive_df.loc['2022-12-09 01:00:00+00:00':'2023-11-30 00:00:00+00:00']
    naive_df = naive_df.rename(columns={'Day-ahead Price [GBP/MWh]':'Actual'})

    glear_df = glear_df.loc['2022-12-09 01:00:00+00:00':'2023-11-30 00:00:00+00:00']
    lear24_df = lear24_df.loc['2022-12-09 01:00:00+00:00':'2023-11-30 00:00:00+00:00']
    dnn_df = dnn_df.loc['2022-12-09 01:00:00+00:00':'2023-11-30 00:00:00+00:00']  
    glear_df.columns = [f'gLEAR {col}' for col in glear_df.columns]
    lear24_df.columns = [f'24LEAR {col}' for col in lear24_df.columns]
    dnn_df.columns = [f'DNN {col}' for col in dnn_df.columns]

    combined_df = pd.concat([naive_df,glear_df, lear24_df, dnn_df], axis=1)

    combined_df = combined_df.drop(columns=['gLEAR Actual','24LEAR Actual','DNN Ytrue'])
    combined_df.columns = [col.replace('Predicted ','') for col in combined_df.columns]
    combined_df.to_csv('forecasts/combined_forecasts.csv')
    print('Combined forecasts saved to forecasts/combined_forecasts.csv')
    actual_df = pd.DataFrame(combined_df['Actual'], index=combined_df.index)
    forecasts_df = combined_df.drop(columns='Actual')

    return combined_df,actual_df,forecasts_df

def save_dataframe_as_image(df, path):
    df = df.round(2)
    fig, ax = plt.subplots(figsize=(35, 10)) 
    # ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colColours=['lightgrey']*df.shape[1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (i, j), cell in table.get_celld().items():
        cell.set_fontsize(14)
        cell.set_height(0.05)
    ax.set_title('Accuracy Metrics', fontsize=18, fontweight='bold')
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    plt.close()


def statistical_tests(actual_df,forecasts, norm=1, plots_path='analysis/plots'):
    actual_df_reshaped = actual_df['Actual'].values.reshape(-1, 24)
    forecasts_reshaped = pd.DataFrame({
        col: forecasts[col].values.reshape(-1, 24).flatten() for col in forecasts.columns
    })

    # Initialize p-values matrices for GW and DM tests
    p_values_gw = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns)
    p_values_dm = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns)

    # Fill p-values matrices by iterating over all model pairs
    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            if model1 == model2:
                p_values_gw.loc[model1, model2] = 1  # Set diagonal elements to 1 for self-comparison
                p_values_dm.loc[model1, model2] = 1  # Set diagonal elements to 1 for self-comparison
            else:
                p_values_gw.loc[model1, model2] = GW(
                    p_real=actual_df_reshaped,
                    p_pred_1=forecasts_reshaped[model1].values.reshape(-1, 24),
                    p_pred_2=forecasts_reshaped[model2].values.reshape(-1, 24),
                    norm=norm,
                    version='multivariate'
                )
                p_values_dm.loc[model1, model2] = DM(
                    p_real=actual_df_reshaped,
                    p_pred_1=forecasts_reshaped[model1].values.reshape(-1, 24),
                    p_pred_2=forecasts_reshaped[model2].values.reshape(-1, 24),
                    norm=norm,
                    version='multivariate'
                )

    # Plot results using predefined plot functions, assuming they can handle the DataFrame structure
    plot_multivariate_GW_test(real_price=pd.DataFrame(actual_df_reshaped), forecasts=forecasts_reshaped, norm=1, title='Multivariate GW Test', savefig=True, path=plots_path)
    plot_multivariate_DM_test(real_price=pd.DataFrame(actual_df_reshaped), forecasts=forecasts_reshaped, norm=1, title='Multivariate DM Test', savefig=True, path=plots_path)





