import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import statsmodels.api as sm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import traceback # To print detailed error messages
import logging # For better error logging
import tempfile
import os

# --- Configuration ---
st.set_page_config(
    page_title="Climate & Health Data Profiler",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "A Streamlit app to profile climate (GRIB) and health (CSV/Excel) data."
    }
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

@st.cache_data(show_spinner="Loading health data...") # Add spinner message to cached function
def load_health_data(uploaded_file):
    """
    Loads health data from an uploaded CSV or Excel file into a Pandas DataFrame.

    Args:
        uploaded_file: The file uploaded via st.file_uploader.

    Returns:
        pandas.DataFrame or None: The loaded DataFrame, or None if loading fails.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            logging.info(f"Successfully loaded CSV: {uploaded_file.name}")
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            # Ensure openpyxl is installed (listed in requirements.txt)
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            logging.info(f"Successfully loaded Excel: {uploaded_file.name}")
        else:
            st.error("Unsupported file format for health data. Please upload CSV or Excel.")
            logging.error(f"Unsupported health file format: {uploaded_file.name}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading health data from '{uploaded_file.name}': {e}")
        st.error("Please ensure the file is not corrupted and is a valid CSV or Excel file.")
        logging.error(f"Error loading health data: {e}", exc_info=True) # Log traceback
        # st.error(traceback.format_exc()) # Optionally show traceback in UI for debugging
        return None

@st.cache_data(show_spinner="Loading GRIB data...") # Add spinner message
def load_climate_data(uploaded_file):
    """
    Loads climate data from an uploaded GRIB file using xarray and cfgrib.

    Args:
        uploaded_file: The GRIB file uploaded via st.file_uploader.

    Returns:
        xarray.Dataset or None: The loaded Xarray Dataset, or None if loading fails.
    """
    try:
        # Create a temp file that will NOT auto-delete
        fd, tmp_path = tempfile.mkstemp(suffix=".grib")
        # Write the uploaded bytes
        with os.fdopen(fd, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Now open_dataset from that file; it will still exist later
        ds = xr.open_dataset(tmp_path, engine="cfgrib")
        logging.info(f"Successfully loaded GRIB file: {uploaded_file.name}")
        return ds

    except ValueError as ve:
        st.error(f"Error loading GRIB data from '{uploaded_file.name}'. Is it a valid GRIB?")
        st.error(f"Details: {ve}")
        st.warning("Ensure 'eccodes' and 'cfgrib' are installed correctly.")
        logging.error(f"ValueError loading GRIB: {ve}", exc_info=True)
        return None

    except FileNotFoundError:
        st.error("Error loading GRIB: `eccodes` library might be missing.")
        st.warning("Install via Conda: `conda install -c conda-forge eccodes`")
        logging.error("FileNotFoundError loading GRIB", exc_info=True)
        return None

    except Exception as e:
        st.error(f"Unexpected error loading GRIB from '{uploaded_file.name}': {e}")
        st.warning("Ensure 'eccodes' and 'cfgrib' are installed correctly.")
        logging.error(f"Unexpected error loading GRIB: {e}", exc_info=True)
        return None
    
def identify_potential_columns(df):
    """
    Identifies potential spatial (latitude, longitude) and temporal columns
    in a DataFrame based on common naming conventions. Also attempts to find
    a column that can be successfully converted to datetime.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        tuple: Contains lists of potential lat, lon, time columns, and the name
               of the first column successfully parsed as datetime (or None).
    """
    if df is None:
        return [], [], [], None

    cols = df.columns.str.lower()
    lat_cols = [c for c in df.columns if 'lat' in c.lower()]
    lon_cols = [c for c in df.columns if 'lon' in c.lower() or 'lng' in c.lower()] # Added 'lng'
    # Prioritize common, specific names for time columns
    time_keywords = ['date', 'time', 'datetime', 'dttm', 'timestamp', 'year', 'month', 'day']
    time_cols = [c for c in df.columns if any(kw in c.lower() for kw in time_keywords)]

    potential_time_col = None
    if time_cols:
        # Try converting potential time columns to find one that works
        for col in time_cols:
            try:
                # Attempt conversion on a sample first for efficiency
                sample_size = min(1000, len(df)) # Check first 1000 non-NA values
                pd.to_datetime(df[col].dropna().head(sample_size), errors='raise')
                potential_time_col = col
                logging.info(f"Auto-identified '{col}' as potential datetime column.")
                break # Use the first one that converts successfully
            except (ValueError, TypeError, OverflowError):
                logging.debug(f"Column '{col}' could not be parsed as datetime.")
                continue # Try the next potential time column

    return lat_cols, lon_cols, time_cols, potential_time_col

@st.cache_data(show_spinner="Performing STL decomposition...")
def perform_stl_decomposition(df, time_col, value_col, period=None):
    """
    Performs STL (Seasonal-Trend decomposition using LOESS) on a time series.

    Args:
        df (pd.DataFrame): DataFrame containing the time series.
        time_col (str): Name of the column containing datetime information.
        value_col (str): Name of the column containing the values to decompose.
        period (int, optional): The seasonal period. If None, attempts to infer.
                                Defaults to None.

    Returns:
        tuple: (matplotlib.figure.Figure or None, statsmodels.tsa.seasonal.STLResult or None)
               Returns the plot figure and the decomposition result object, or (None, None) on failure.
    """
    try:
        if time_col not in df.columns or value_col not in df.columns:
            st.error(f"Selected columns '{time_col}' or '{value_col}' not found in DataFrame.")
            return None, None

        # Prepare the time series data
        df_ts = df[[time_col, value_col]].copy()
        try:
            df_ts[time_col] = pd.to_datetime(df_ts[time_col])
        except (ValueError, TypeError) as dt_error:
            st.error(f"Could not convert column '{time_col}' to datetime: {dt_error}")
            return None, None

        df_ts = df_ts.sort_values(by=time_col).set_index(time_col)

        # Drop rows where the value column is NaN, as STL cannot handle them
        df_ts = df_ts.dropna(subset=[value_col])

        if df_ts.empty:
            st.warning("No valid data points remain after handling missing values and date conversion. Cannot perform STL.")
            return None, None

        # STL requires at least 2 full periods of data
        min_data_points = 2 * (period if period else 7) + 1 # Heuristic, +1 for safety
        if df_ts.shape[0] < min_data_points:
             st.warning(f"Not enough data points ({df_ts.shape[0]}) for STL decomposition with the selected period ({period}). Need at least {min_data_points}. Skipping.")
             return None, None

        # Perform STL decomposition
        # seasonal=13 is often a good default for monthly data if period isn't specified
        # Use robust=True to handle outliers better
        stl = sm.tsa.STL(df_ts[value_col], seasonal=period if period else 13, robust=True)
        res = stl.fit()

        # Create the decomposition plot
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
        axes[0].plot(res.observed, label='Observed')
        axes[0].legend(loc='upper left')
        axes[1].plot(res.trend, label='Trend')
        axes[1].legend(loc='upper left')
        axes[2].plot(res.seasonal, label='Seasonal')
        axes[2].legend(loc='upper left')
        axes[3].plot(res.resid, label='Residual')
        axes[3].legend(loc='upper left')
        axes[3].set_xlabel(f"Time ({time_col})")
        fig.suptitle(f'STL Decomposition of {value_col}', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout for title
        plt.close(fig) # Prevent matplotlib from displaying the plot directly

        logging.info(f"STL decomposition successful for {value_col} vs {time_col}.")
        return fig, res

    except Exception as e:
        st.error(f"Error during STL decomposition: {e}")
        logging.error("Error during STL decomposition", exc_info=True)
        # st.error(traceback.format_exc())
        return None, None

@st.cache_data(show_spinner="Performing geospatial clustering...")
def perform_geo_clustering(df, lat_col, lon_col, n_clusters):
    """
    Performs K-Means clustering on latitude/longitude data.

    Args:
        df (pd.DataFrame): DataFrame containing the coordinates.
        lat_col (str): Name of the latitude column.
        lon_col (str): Name of the longitude column.
        n_clusters (int): The number of clusters (K).

    Returns:
        tuple: (pd.DataFrame or None, sklearn.cluster.KMeans or None)
               Returns the DataFrame with cluster assignments and the fitted
               KMeans model, or (None, None) on failure.
    """
    try:
        if not lat_col or not lon_col:
            st.warning("Latitude or Longitude column not selected. Skipping clustering.")
            return None, None
        if lat_col not in df.columns or lon_col not in df.columns:
            st.error(f"Selected columns '{lat_col}' or '{lon_col}' not found in DataFrame.")
            return None, None

        # Select and drop rows with missing coordinates
        coords_df = df[[lat_col, lon_col]].dropna()

        if coords_df.empty:
            st.warning("No valid coordinate pairs found after dropping missing values. Cannot perform clustering.")
            return None, None

        # Ensure n_clusters is not more than the number of samples
        if coords_df.shape[0] < n_clusters:
            st.warning(f"Number of data points ({coords_df.shape[0]}) is less than the requested number of clusters ({n_clusters}). Adjusting K to {coords_df.shape[0]}.")
            n_clusters = max(1, coords_df.shape[0]) # Ensure K is at least 1

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Set n_init explicitly
        # Create a copy to avoid modifying the original slice
        coords_result_df = coords_df.copy()
        coords_result_df['cluster'] = kmeans.fit_predict(coords_result_df[[lat_col, lon_col]])

        # Prepare DataFrame for st.map (requires 'lat' and 'lon' column names)
        map_df = coords_result_df.rename(columns={lat_col: 'lat', lon_col: 'lon'})

        logging.info(f"Geospatial clustering successful with K={n_clusters}.")
        return map_df, kmeans

    except Exception as e:
        st.error(f"Error during geospatial clustering: {e}")
        logging.error("Error during geospatial clustering", exc_info=True)
        # st.error(traceback.format_exc())
        return None, None

def perform_data_quality_checks(df, df_name="Data"):
    """
    Performs and displays basic data quality checks for a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        df_name (str): A descriptive name for the data source (e.g., "Health Data").
    """
    if df is None:
        st.warning(f"No data available for quality checks on {df_name}.")
        return

    st.subheader(f"Data Quality Report: {df_name}")

    results = {}
    results['Total Rows'] = len(df)
    results['Total Columns'] = len(df.columns)

    # Missing Values
    missing_values = df.isnull().sum()
    results['Missing Values (Total)'] = int(missing_values.sum()) # Ensure integer type
    results['Missing Values (Per Column)'] = missing_values[missing_values > 0].astype(int) # Ensure integer type
    results['Percentage Missing (Total)'] = (results['Missing Values (Total)'] / (results['Total Rows'] * results['Total Columns'])) * 100 if results['Total Rows'] > 0 and results['Total Columns'] > 0 else 0

    # Duplicate Rows
    results['Duplicate Rows'] = int(df.duplicated().sum()) # Ensure integer type
    results['Percentage Duplicate'] = (results['Duplicate Rows'] / results['Total Rows']) * 100 if results['Total Rows'] > 0 else 0

    # Display Summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", results['Total Rows'])
        st.metric("Total Missing Values", f"{results['Missing Values (Total)']} ({results['Percentage Missing (Total)']:.2f}%)")
    with col2:
        st.metric("Total Columns", results['Total Columns'])
        st.metric("Duplicate Rows", f"{results['Duplicate Rows']} ({results['Percentage Duplicate']:.2f}%)")


    # Detailed Missing Values
    if not results['Missing Values (Per Column)'].empty:
        with st.expander("Columns with Missing Values"):
            st.dataframe(results['Missing Values (Per Column)'])
    else:
        st.write("**Columns with Missing Values:** None")

    # Data Types
    with st.expander("Data Types per Column"):
        st.dataframe(df.dtypes.astype(str), use_container_width=True) # Show dtypes as strings

    # Specific checks for lat/lon if present
    lat_cols, lon_cols, _, _ = identify_potential_columns(df)

    if lat_cols:
        lat_col = lat_cols[0] # Use the first identified lat column
        if pd.api.types.is_numeric_dtype(df[lat_col]):
            invalid_lat = df[(df[lat_col] < -90) | (df[lat_col] > 90)][lat_col]
            if not invalid_lat.empty:
                st.warning(f"**Latitude Range Check ('{lat_col}'):** Found {len(invalid_lat)} values outside the valid range [-90, 90].")
            else:
                 st.success(f"**Latitude Range Check ('{lat_col}'):** All values within [-90, 90].")
        else:
            st.info(f"**Latitude Range Check ('{lat_col}'):** Column is not numeric, cannot perform range check.")

    if lon_cols:
        lon_col = lon_cols[0] # Use the first identified lon column
        if pd.api.types.is_numeric_dtype(df[lon_col]):
             # Check for values strictly outside both common ranges
            invalid_lon = df[~((df[lon_col] >= -180) & (df[lon_col] <= 180)) & ~((df[lon_col] >= 0) & (df[lon_col] <= 360))][lon_col]
            if not invalid_lon.empty:
                 st.warning(f"**Longitude Range Check ('{lon_col}'):** Found {len(invalid_lon)} values outside the common ranges [-180, 180] or [0, 360].")
            else:
                 # Check which range is likely used
                 in_180 = df[(df[lon_col] >= -180) & (df[lon_col] <= 180)][lon_col]
                 in_360 = df[(df[lon_col] >= 0) & (df[lon_col] <= 360)][lon_col]
                 if len(in_180) > len(in_360):
                     st.success(f"**Longitude Range Check ('{lon_col}'):** All values within [-180, 180].")
                 elif len(in_360) > len(in_180):
                      st.success(f"**Longitude Range Check ('{lon_col}'):** All values within [0, 360].")
                 else: # Handle cases where it might be ambiguous or empty
                      st.success(f"**Longitude Range Check ('{lon_col}'):** Values appear valid (within common ranges).")
        else:
            st.info(f"**Longitude Range Check ('{lon_col}'):** Column is not numeric, cannot perform range check.")

    logging.info(f"Data quality checks performed for {df_name}.")


# --- Streamlit App Layout ---

st.title("🌍 Climate & Health Data Profiler 🩺")

st.markdown("""
Welcome! This tool helps you analyze and understand your health data (CSV/Excel)
and climate data (GRIB format). Upload your files using the sidebar to get started.
""")

# --- Sidebar for File Uploads ---
st.sidebar.header("📤 Upload Data Files")
uploaded_health_file = st.sidebar.file_uploader(
    "Upload Health Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'], key="health_uploader"
)
uploaded_climate_file = st.sidebar.file_uploader(
    "Upload Climate Data (GRIB)", type=['grib', 'grb', 'grib2'], key="climate_uploader"
)

# --- Initialize session state variables ---
# Use keys that are less likely to clash if the script reruns unexpectedly
if 'health_data_df' not in st.session_state:
    st.session_state.health_data_df = None
if 'climate_data_ds' not in st.session_state:
    st.session_state.climate_data_ds = None
if 'climate_data_df' not in st.session_state: # For the DataFrame version of climate data
    st.session_state.climate_data_df = None
if 'health_profile_generated' not in st.session_state:
    st.session_state.health_profile_generated = False # Track if profile report ran

# --- Load data when files are uploaded ---
# Health Data Loading
if uploaded_health_file is not None:
    # Check if the file is new or hasn't been loaded yet
    if st.session_state.health_data_df is None or getattr(st.session_state, 'loaded_health_filename', '') != uploaded_health_file.name:
        st.session_state.health_data_df = load_health_data(uploaded_health_file)
        st.session_state.loaded_health_filename = uploaded_health_file.name if st.session_state.health_data_df is not None else ''
        st.session_state.health_profile_generated = False # Reset profile flag when new file loaded
        if st.session_state.health_data_df is not None:
            st.sidebar.success(f"Loaded: {uploaded_health_file.name}")
            # Automatically switch tab or give feedback
        else:
            st.sidebar.error(f"Failed to load: {uploaded_health_file.name}")


# Climate Data Loading
if uploaded_climate_file is not None:
     # Check if the file is new or hasn't been loaded yet
    if st.session_state.climate_data_ds is None or getattr(st.session_state, 'loaded_climate_filename', '') != uploaded_climate_file.name:
        st.session_state.climate_data_ds = load_climate_data(uploaded_climate_file)
        st.session_state.loaded_climate_filename = uploaded_climate_file.name if st.session_state.climate_data_ds is not None else ''
        st.session_state.climate_data_df = None # Reset DataFrame version when new GRIB loaded
        if st.session_state.climate_data_ds is not None:
            st.sidebar.success(f"Loaded: {uploaded_climate_file.name}")
            # Convert to DataFrame immediately after loading GRIB for quality checks/integration tab
            try:
                with st.spinner("Converting GRIB to DataFrame for analysis..."):
                    st.session_state.climate_data_df = st.session_state.climate_data_ds.to_dataframe().reset_index()
                    logging.info("Converted GRIB dataset to DataFrame.")
            except Exception as e:
                 st.warning(f"Could not automatically convert GRIB dataset to DataFrame: {e}")
                 logging.warning("Failed to auto-convert GRIB to DataFrame", exc_info=True)
                 st.session_state.climate_data_df = None
        else:
            st.sidebar.error(f"Failed to load: {uploaded_climate_file.name}")


# --- Main Area Tabs ---
tab_list = [
    "✔️ Quality Checks",
    "📊 Health Data Analysis",
    "🌡️ Climate Data Analysis",
    "🔗 Integration Suggestions"
]
# Set Quality Checks as the default landing tab
tab1, tab2, tab3, tab4 = st.tabs(tab_list)


# --- Quality Checks Tab ---
with tab1:
    st.header("✔️ Data Quality Checks")
    st.markdown("Basic checks for missing values, duplicates, and potential data range issues.")

    st.subheader("Health Data Quality")
    if st.session_state.get('health_data_df') is not None:
         perform_data_quality_checks(st.session_state['health_data_df'], df_name="Health Data")
    elif uploaded_health_file:
         st.warning("Health data could not be loaded. Please check the file and error messages in the sidebar.")
    else:
         st.info("Upload health data (CSV/Excel) using the sidebar to see quality checks.")

    st.divider() # Visual separator

    st.subheader("Climate Data Quality")
    # Perform checks on the DataFrame version if available
    if st.session_state.get('climate_data_df') is not None:
         perform_data_quality_checks(st.session_state['climate_data_df'], df_name="Climate Data (from GRIB)")
    elif st.session_state.get('climate_data_ds') is not None:
         st.info("GRIB data loaded, but could not be converted to a table (DataFrame) for detailed quality checks. Basic structure is available in the 'Climate Data Analysis' tab.")
    elif uploaded_climate_file:
         st.warning("Climate data (GRIB) could not be loaded. Please check the file, error messages, and ensure 'eccodes'/'cfgrib' are installed correctly (see README).")
    else:
         st.info("Upload climate data (GRIB) using the sidebar to see quality checks.")


# --- Health Data Analysis Tab ---
with tab2:
    st.header("📊 Health Data Analysis")
    if st.session_state.get('health_data_df') is not None:
        health_df = st.session_state['health_data_df']
        st.success(f"Displaying analysis for: `{st.session_state.loaded_health_filename}`")
        st.dataframe(health_df.head(), use_container_width=True)

        # Identify columns
        lat_cols, lon_cols, time_cols, auto_time_col = identify_potential_columns(health_df)
        numeric_cols = health_df.select_dtypes(include='number').columns.tolist()
        all_cols = health_df.columns.tolist()

        # --- Automated Profiling ---
        st.subheader("🤖 Automated Data Profiling")
        st.markdown("Generate a detailed report covering variable types, distributions, correlations, missing values, and more.")
        if st.button("Generate Profiling Report", key="profile_button"):
            # Check if report already generated for this file to avoid re-computation if button clicked again
            if not st.session_state.health_profile_generated:
                with st.spinner("Generating Pandas Profiling report (this may take a while for large datasets)..."):
                    try:
                        profile = ProfileReport(health_df,
                                                title=f"Profiling Report: {st.session_state.loaded_health_filename}",
                                                explorative=True,
                                                minimal=False) # Use minimal=True for faster reports on very large data
                        st.session_state.health_profile_report = profile
                        st.session_state.health_profile_generated = True
                        logging.info("Pandas Profiling report generated.")
                    except Exception as e:
                         st.error(f"Could not generate Pandas Profiling report: {e}")
                         logging.error("Error generating Pandas Profiling report", exc_info=True)
                         st.session_state.health_profile_report = None
                         st.session_state.health_profile_generated = False # Mark as failed

            # Display the report if it exists in session state
            if 'health_profile_report' in st.session_state and st.session_state.health_profile_report:
                with st.expander("View Profiling Report", expanded=True):
                     st_profile_report(st.session_state.health_profile_report)
            elif st.session_state.health_profile_generated: # If generation failed
                 st.warning("Report generation failed. Please check the data or try again.")

        # --- Time Series Analysis (STL) ---
        st.subheader("📈 Time Series Analysis (STL Decomposition)")
        st.markdown("Decompose a time series into trend, seasonal, and residual components. Requires a datetime column and a numeric column.")
        if auto_time_col and numeric_cols:
            st.info(f"Detected potential time column: `{auto_time_col}`. Select columns below.")
            col1, col2, col3 = st.columns(3)
            with col1:
                 # Default to auto-detected time column if available
                 time_col_select_stl = st.selectbox("Select Time Column:", options=all_cols, index=all_cols.index(auto_time_col) if auto_time_col in all_cols else 0, key="stl_time_col")
            with col2:
                 value_col_select_stl = st.selectbox("Select Value Column:", options=numeric_cols, index=0 if numeric_cols else -1, key="stl_value_col")
            with col3:
                 # Sensible default period (e.g., 7 for daily, 12 for monthly) - let user adjust
                 period_stl = st.number_input("Seasonal Period (e.g., 7, 12, 52)", min_value=2, value=7, step=1, key="stl_period")

            if time_col_select_stl and value_col_select_stl:
                 if st.button("Run STL Decomposition", key="stl_button"):
                     stl_fig, _ = perform_stl_decomposition(health_df, time_col_select_stl, value_col_select_stl, period=period_stl)
                     if stl_fig:
                         st.pyplot(stl_fig)
                     # Error/warning messages are handled inside the function
            else:
                st.warning("Please select valid Time and Value columns for STL decomposition.")
        else:
            st.info("Could not automatically identify suitable time or numeric columns for STL decomposition. Ensure your data has datetime-like and numeric columns, or select them manually.")
            # Allow manual selection even if auto-detection fails
            col1, col2, col3 = st.columns(3)
            with col1:
                 time_col_select_manual_stl = st.selectbox("Select Time Column (Manual):", options=all_cols, index=0, key="stl_time_col_manual")
            with col2:
                 value_col_select_manual_stl = st.selectbox("Select Value Column (Manual):", options=numeric_cols, index=0 if numeric_cols else -1, key="stl_value_col_manual")
            with col3:
                 period_manual_stl = st.number_input("Seasonal Period (Manual):", min_value=2, value=7, step=1, key="stl_period_manual")
            if time_col_select_manual_stl and value_col_select_manual_stl:
                 if st.button("Run STL Decomposition (Manual)", key="stl_button_manual"):
                     stl_fig_manual, _ = perform_stl_decomposition(health_df, time_col_select_manual_stl, value_col_select_manual_stl, period=period_manual_stl)
                     if stl_fig_manual:
                         st.pyplot(stl_fig_manual)


        # --- Geospatial Clustering ---
        st.subheader("🗺️ Geospatial Cluster Analysis (K-Means)")
        st.markdown("Group data points based on their latitude and longitude coordinates.")
        if lat_cols and lon_cols:
            st.info(f"Detected potential coordinate columns: Lat (`{lat_cols[0]}`), Lon (`{lon_cols[0]}`). Select columns below.")
            col1, col2, col3 = st.columns(3)
            with col1:
                lat_col_select_cluster = st.selectbox("Select Latitude Column:", options=lat_cols, index=0, key="cluster_lat_col")
            with col2:
                lon_col_select_cluster = st.selectbox("Select Longitude Column:", options=lon_cols, index=0, key="cluster_lon_col")
            with col3:
                n_clusters_cluster = st.slider("Number of Clusters (K):", min_value=2, max_value=20, value=5, step=1, key="cluster_k")

            if lat_col_select_cluster and lon_col_select_cluster:
                 if st.button("Run Geospatial Clustering", key="cluster_button"):
                    clustered_df, kmeans_model = perform_geo_clustering(health_df, lat_col_select_cluster, lon_col_select_cluster, n_clusters_cluster)
                    if clustered_df is not None:
                         st.success("Clustering complete.")
                         # Use st.map for interactive map
                         st.map(clustered_df[['lat','lon']])
                         # Show cluster centers
                         with st.expander("View Cluster Centers"):
                             centers_df = pd.DataFrame(kmeans_model.cluster_centers_, columns=['Latitude', 'Longitude'])
                             st.dataframe(centers_df, use_container_width=True)
                         # Show sample data with cluster assignments
                         with st.expander("View Sample Data with Cluster Assignments"):
                            st.dataframe(clustered_df.head(50), use_container_width=True)
                    # Error/warning messages handled inside function
        else:
            st.info("Could not automatically identify potential latitude/longitude columns for clustering. Ensure your data has columns named like 'lat', 'latitude', 'lon', 'longitude', or 'lng'.")

    elif uploaded_health_file:
        st.warning("Health data could not be loaded. Please check the file and error messages in the sidebar.")
    else:
        st.info("Upload a health data file (CSV or Excel) in the sidebar to begin analysis.")

# --- Climate Data Analysis Tab ---
with tab3:
    st.header("🌡️ Climate Data Analysis")
    if st.session_state.get('climate_data_ds') is not None:
        climate_ds = st.session_state['climate_data_ds']
        st.success(f"Displaying analysis for: `{st.session_state.loaded_climate_filename}`")

        st.subheader("GRIB Dataset Structure")
        st.markdown("Explore the variables, coordinates, and dimensions within the GRIB file.")
        with st.expander("Show Raw Dataset Structure"):
            buffer = io.StringIO()
            climate_ds.info(buf=buffer)
            st.text(buffer.getvalue())

        # Show tabular preview if available
        if st.session_state.get('climate_data_df') is not None:
            st.subheader("Data as Table (Sample)")
            st.markdown("A sample of the climate data converted into a tabular format.")
            st.dataframe(st.session_state['climate_data_df'].head(), use_container_width=True)
            st.caption(f"Full table shape (rows, columns): {st.session_state['climate_data_df'].shape}")
        else:
            st.info("Could not convert the GRIB dataset into a table for preview. Visualization might still be possible.")

        # --- Basic Visualization ---
        st.subheader("🗺️ Climate Variable Visualization")
        st.markdown("Select a variable and specific dimensions (like time or vertical level) to plot it on a map.")

        data_vars = list(climate_ds.data_vars)
        if not data_vars:
            st.warning("No data variables found in the GRIB file to visualize.")
        else:
            # 1) Variable selector
            selected_var_grib = st.selectbox("Select Climate Variable:", data_vars, key="grib_var_select")

            # 2) Inspect dims of the chosen variable
            var = climate_ds[selected_var_grib]
            lat_coord_grib = next((d for d in var.dims if 'lat' in d.lower()), None)
            lon_coord_grib = next((d for d in var.dims if 'lon' in d.lower()), None)
            time_coord_grib = next((d for d in var.dims if d in ('time','valid_time','step')), None)
            level_coord_grib = next(
                (d for d in var.dims
                 if d in ('level','isobaricInhPa','isobaricInhpa',
                          'heightAboveGround','depthBelowLandLayer','hybrid')),
                None
            )

            # 3) Build selectors based on available dims
            selection_dict_grib = {}
            sliders_cols = st.columns(2)

            with sliders_cols[0]:
                if time_coord_grib:
                    time_values = var[time_coord_grib].values
                    if len(time_values) > 1:
                        selected_time = st.select_slider(
                            f"Select Time ({time_coord_grib}):",
                            options=time_values,
                            value=time_values[0],
                            key="grib_time_select"
                        )
                        selection_dict_grib[time_coord_grib] = selected_time
                    else:
                        single_time = time_values[0]
                        st.write(f"Time ({time_coord_grib}): {single_time}")
                        selection_dict_grib[time_coord_grib] = single_time

            with sliders_cols[1]:
                if level_coord_grib:
                    raw_vals = var[level_coord_grib].values
                    level_values = np.atleast_1d(raw_vals)
                    if len(level_values) > 1:
                        selected_level = st.selectbox(
                            f"Select Level ({level_coord_grib}):",
                            options=level_values,
                            key="grib_level_select"
                        )
                        selection_dict_grib[level_coord_grib] = selected_level
                    else:
                        single_level = level_values[0]
                        st.write(f"Level ({level_coord_grib}): {single_level}")
                        selection_dict_grib[level_coord_grib] = single_level

            # Other dims (e.g., ensemble member “number”)
            other_dims = [d for d in var.dims
                          if d not in (lat_coord_grib, lon_coord_grib, time_coord_grib, level_coord_grib)]
            if other_dims:
                st.write("Other Dimensions:")
                for dim in other_dims:
                    vals = var[dim].values
                    if len(vals) > 1:
                        sel = st.selectbox(f"Select {dim}:", options=vals, key=f"grib_other_dim_{dim}")
                        selection_dict_grib[dim] = sel
                    else:
                        st.write(f"{dim}: {vals[0]}")
                        selection_dict_grib[dim] = vals[0]

            # 4) Plot if we have lat/lon
            if lat_coord_grib and lon_coord_grib:
                if st.button("Generate Climate Map", key="grib_plot_button"):
                    try:
                        data_slice = var.sel(**selection_dict_grib, method="nearest")
                        # Ensure numeric dtype for plotting
                        try:
                            data_slice = data_slice.astype(float)
                        except Exception:
                            pass
                        with st.spinner(f"Generating plot for {selected_var_grib}..."):
                            fig, ax = plt.subplots(figsize=(10, 6))
                            try:
                                data_slice.plot.contourf(
                                    ax=ax,
                                    x=lon_coord_grib,
                                    y=lat_coord_grib,
                                    cmap='viridis',
                                    add_colorbar=True
                                )
                            except Exception:
                                ax.clear()
                                data_slice.plot(ax=ax, x=lon_coord_grib, y=lat_coord_grib)
                            ax.set_title(f"{selected_var_grib} ({data_slice.attrs.get('units','')})")
                            st.pyplot(fig)
                            plt.close(fig)
                    except Exception as e:
                        st.error(f"Error preparing climate data slice: {e}")
            else:
                st.warning("Could not identify latitude/longitude dims for plotting.")

    elif uploaded_climate_file:
        st.warning("Climate data (GRIB) could not be loaded. Please check the file and ensure dependencies are installed.")
    else:
        st.info("Upload a climate data file (GRIB) in the sidebar to begin analysis.")
# --- Integration Suggestions Tab ---
with tab4:
    st.header("🔗 Integration Suggestions")
    st.markdown("""
    Based on the analyses of the individual datasets, here are potential strategies
    and considerations for integrating your health and climate data. Successful
    integration typically requires aligning the data in **time** and **space**.
    """)

    # Check if both datasets (or their DataFrame versions) are loaded
    health_df_present = st.session_state.get('health_data_df') is not None
    # Use climate DataFrame if available, otherwise indicate GRIB is loaded but not tabular
    climate_df_present = st.session_state.get('climate_data_df') is not None
    climate_ds_present = st.session_state.get('climate_data_ds') is not None

    if health_df_present and (climate_df_present or climate_ds_present):
        health_df = st.session_state['health_data_df']
        # Use climate_df if available, otherwise we know climate_ds exists but wasn't converted
        climate_df = st.session_state.get('climate_data_df')

        # Identify potential keys again for clarity in this section
        h_lat_cols, h_lon_cols, h_time_cols, h_auto_time_col = identify_potential_columns(health_df)
        # For climate, check the DataFrame if available, otherwise infer from DS coords
        if climate_df is not None:
             c_lat_cols, c_lon_cols, c_time_cols, c_auto_time_col = identify_potential_columns(climate_df)
             c_has_latlon = bool(c_lat_cols and c_lon_cols)
             c_has_time = bool(c_auto_time_col)
        elif climate_ds_present: # GRIB loaded but not as DF
             coords = list(st.session_state['climate_data_ds'].coords)
             c_has_latlon = any('lat' in c.lower() for c in coords) and any('lon' in c.lower() for c in coords)
             c_has_time = any(tc in coords for tc in ['time', 'valid_time', 'step'])
             c_auto_time_col = next((tc for tc in ['time', 'valid_time', 'step'] if tc in coords), None) # Name of potential time coord
             st.info("Climate data (GRIB) loaded but not converted to a table. Suggestions based on GRIB structure.")
        else: # Should not happen if outer condition is met, but for safety
             c_has_latlon = False
             c_has_time = False
             c_auto_time_col = None


        # 1. Temporal Alignment
        st.subheader("1. Temporal Alignment (Matching Time)")
        if h_auto_time_col and c_has_time:
            st.success("✅ Both datasets appear to have time information.")
            st.markdown(f"- Health data time column (auto-detected): `{h_auto_time_col}`")
            st.markdown(f"- Climate data time coordinate/column (auto-detected): `{c_auto_time_col}`")
            st.markdown("""
            **Potential Steps:**
            * **Standardize Format:** Convert both time columns/coordinates to a consistent datetime format (e.g., using `pd.to_datetime`).
            * **Choose Resolution:** Decide on the target time resolution for your analysis (e.g., daily, weekly, monthly). Climate data often dictates the finest possible resolution.
            * **Aggregate/Resample:** Aggregate or resample one or both datasets to match the chosen resolution.
                * *Example (Health):* If climate data is daily, you might sum or average daily health counts: `health_df.resample('D', on='datetime_col').agg({'cases': 'sum'})`
                * *Example (Climate):* If health data is weekly, you might calculate weekly average temperature: `climate_df.resample('W', on='datetime_col').agg({'temperature': 'mean'})`
            * **Time Lag:** Consider potential time lags between climate events and health outcomes (e.g., heatwave effects might appear days later). You might need to merge data from previous time steps (e.g., using `pd.shift()`).
            """)
        elif h_auto_time_col:
            st.warning("⚠️ Health data has time information, but it wasn't clearly identified or converted in the climate data.")
            st.markdown("- Check the climate data structure (in the 'Climate Data Analysis' tab) for coordinates like 'time', 'valid_time', or 'step'.")
            st.markdown("- If converting GRIB to a DataFrame failed, temporal alignment will require processing the `xarray.Dataset` directly.")
        elif c_has_time:
             st.warning("⚠️ Climate data has time information, but it wasn't clearly identified in the health data.")
             st.markdown("- Check if your health data contains date/time information in columns not automatically detected.")
        else:
             st.error("❌ Could not automatically identify time information in one or both datasets. Manual inspection and processing are required for temporal alignment.")

        # 2. Spatial Alignment
        st.subheader("2. Spatial Alignment (Matching Location)")
        h_has_latlon = bool(h_lat_cols and h_lon_cols)

        if h_has_latlon and c_has_latlon:
            st.success("✅ Both datasets appear to have geographic coordinates (latitude/longitude).")
            st.markdown(f"- Health data coordinates (auto-detected): Lat=`{h_lat_cols[0]}`, Lon=`{h_lon_cols[0]}`")
            if climate_df is not None:
                 st.markdown(f"- Climate data coordinates (auto-detected): Lat=`{c_lat_cols[0]}`, Lon=`{c_lon_cols[0]}`")
            else:
                 st.markdown(f"- Climate data coordinates detected in GRIB structure.")

            st.markdown("""
            **Potential Steps (Choose one based on data structure):**
            * **Method 1: Nearest Neighbor Join (Health Points -> Climate Grid):**
                * *Use Case:* Your health data consists of specific point locations (e.g., individual cases, clinics).
                * *Process:* For each health data point (lat/lon), find the nearest climate grid cell center and merge the climate variables from that cell at the corresponding time.
                * *Tools:* Requires spatial indexing. Libraries like `scipy.spatial.KDTree` or `xarray`'s `.sel(..., method='nearest')` can be used.
            * **Method 2: Zonal Statistics (Health Regions -> Climate Grid):**
                * *Use Case:* Your health data is aggregated by geographic regions (e.g., districts, counties, zip codes).
                * *Process:* For each region in your health data, calculate aggregate statistics (e.g., mean, max, min) of the climate variables within that region's boundaries.
                * *Tools:* Requires region boundary definitions (e.g., Shapefiles, GeoJSON) and libraries like `geopandas`, `rasterio`, and `rasterstats`.
            * **Method 3: Grid-to-Grid (If Health Data is Also Gridded):**
                * *Use Case:* Your health data is already aggregated onto a grid similar to the climate data.
                * *Process:* If grids align perfectly, merge directly. If grids differ, you'll need to **regrid** one dataset to match the other (e.g., using libraries like `xESMF`). This involves interpolation and can be complex.

            **Considerations:**
            * **Coordinate Reference Systems (CRS):** Ensure both datasets use the same CRS (usually WGS84, EPSG:4326 for lat/lon). GRIB is typically WGS84.
            * **Spatial Resolution:** Climate grids can be coarse. Is the climate resolution appropriate for the scale of your health data? Interpolation might be needed but adds uncertainty.
            """)

        elif h_has_latlon and not c_has_latlon and climate_ds_present:
             st.warning("⚠️ Health data has coordinates, but standard lat/lon coordinates were not found or converted from the climate GRIB file.")
             st.markdown("- Check the 'Climate Data Analysis' tab for available coordinates in the GRIB structure. They might have non-standard names.")

        elif not h_has_latlon and c_has_latlon:
             st.warning("⚠️ Climate data has coordinates, but they weren't clearly identified in the health data.")
             st.markdown("""
             - If your health data has location information like addresses, city names, or region codes, you may need to:
                 - **Geocode:** Convert addresses/names to latitude/longitude coordinates (requires external services or libraries like `geopy`).
                 - **Join by Region:** If both datasets share common administrative region names or codes, merge based on those identifiers. You might still need Zonal Statistics (Method 2 above) to get region-specific climate values.
             """)
        else:
             st.error("❌ Could not automatically identify geographic coordinates in one or both datasets. Spatial alignment requires manual inspection and potentially geocoding or joining by administrative regions.")


        # 3. Merging Data
        st.subheader("3. Merging the Aligned Data")
        st.markdown("""
        Once data is aligned temporally and spatially, you can combine them into a single dataset for modeling or further analysis.

        * **Identify Join Keys:** These will be the standardized time column(s) and the spatial identifier (e.g., matched grid cell ID, region name, potentially lat/lon if using nearest neighbor).
        * **Use Pandas Merge/Join:** Employ functions like `pd.merge()` or `df.join()`.
            * `pd.merge(health_aligned, climate_aligned, on=['time_key', 'location_key'], how='inner')`
        * **Choose Merge Type (`how`):**
            * `inner`: Keep only rows where keys exist in *both* datasets.
            * `left`: Keep all rows from the 'left' (health) dataset and matching climate data; fill missing climate data with NaN. (Commonly used).
            * `outer`: Keep all rows from both datasets.
        """)

    else:
        st.info("Upload both health and climate data files using the sidebar to see integration suggestions.")


# --- Footer or additional info ---
st.sidebar.divider()
st.sidebar.info("App developed using Streamlit.")
