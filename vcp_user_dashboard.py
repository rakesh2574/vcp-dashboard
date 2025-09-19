import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

# --- Streamlit UI & Application Logic ---

st.set_page_config(
    page_title="VCP Results Viewer",
    page_icon="📋",
    layout="wide"
)

# Custom CSS for a dark theme, consistent with the analyzer app
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .stDataFrame {
        font-size: 12px;
    }
    .dataframe td, .dataframe th {
        padding: 4px !important;
        font-size: 12px !important;
        border-color: #333 !important;
    }
    h1, h2, h3 {
        color: #FAFAFA;
    }
    .stButton > button {
        background-color: #1976d2;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def convert_to_ist(df):
    """Convert all timestamp columns from UTC to IST for NSE/BSE exchanges"""
    # List of all timestamp columns
    timestamp_columns = [
        'Resistance Time', 'SwingLow1 Time', 'Touch1 Time',
        'SwingLow2 Time', 'Touch2 Time', 'SwingLow3 Time', 
        'Touch3 Time', 'Breakout Time', 'Target Hit Time',
        'SL Hit Time', 'Analysis Time'
    ]
    
    # Create a copy to avoid modifying original
    df_display = df.copy()
    
    # Check if Exchange column exists
    if 'Exchange' in df_display.columns:
        for col in timestamp_columns:
            if col in df_display.columns:
                # Process each row
                for idx in df_display.index:
                    value = df_display.at[idx, col]
                    exchange = df_display.at[idx, 'Exchange']
                    
                    # Skip if value is '-' or NaN
                    if pd.isna(value) or value == '-':
                        continue
                    
                    try:
                        # Parse the timestamp string
                        if isinstance(value, str) and value != '-':
                            # Try to parse the datetime
                            # Remove any existing timezone labels
                            clean_value = value.replace(' UTC', '').replace(' IST', '').replace(' EST', '')
                            dt = pd.to_datetime(clean_value)
                            
                            # Convert based on exchange
                            if exchange in ['NSE', 'BSE']:
                                # Add 5 hours 30 minutes for IST
                                ist_time = dt + timedelta(hours=5, minutes=30)
                                df_display.at[idx, col] = ist_time.strftime('%Y-%m-%d %H:%M') + ' IST'
                            elif exchange in ['NASDAQ', 'NYSE']:
                                # Subtract 5 hours for EST (approximate, ignores DST)
                                est_time = dt - timedelta(hours=5)
                                df_display.at[idx, col] = est_time.strftime('%Y-%m-%d %H:%M') + ' EST'
                            else:
                                # Keep as UTC for others
                                df_display.at[idx, col] = dt.strftime('%Y-%m-%d %H:%M') + ' UTC'
                    except:
                        # If parsing fails, keep original value
                        pass
    
    return df_display


def check_password():
    """Returns `True` if the user has entered the correct password."""
    if "viewer_password_correct" not in st.session_state:
        st.session_state.viewer_password_correct = False

    if st.session_state.viewer_password_correct:
        return True

    st.header("🔑 Viewer Access")
    st.write("Please log in to view the VCP analysis results.")

    username = st.text_input("Username", key="viewer_username")
    password = st.text_input("Password", type="password", key="viewer_password")

    if st.button("Login"):
        if username == "sherlock" and password == "watson":
            st.session_state.viewer_password_correct = True
            st.rerun()
        else:
            st.error("😕 Incorrect username or password. Please try again.")

    return False


def display_results_page():
    """
    Renders the main results page, displaying the data from the CSV file.
    """
    st.title("📋 VCP Analysis Results Dashboard")

    output_filename = "vcp_analysis_results.csv"

    if os.path.exists(output_filename):
        try:
            # Get the last modification time of the file
            last_modified_time = datetime.fromtimestamp(os.path.getmtime(output_filename))
            
            # Convert file modification time to IST
            ist_modified_time = last_modified_time + timedelta(hours=5, minutes=30)
            st.success(
                f"Displaying results from the last analysis on: **{ist_modified_time.strftime('%Y-%m-%d %H:%M:%S IST')}**")

            # Read the CSV file into a DataFrame
            df = pd.read_csv(output_filename)

            if df.empty:
                st.info("The last analysis ran successfully but found no VCP patterns.")
            else:
                # Convert timestamps to appropriate timezone
                df_display = convert_to_ist(df)
                
                # Show timezone information
                if 'Exchange' in df.columns:
                    exchanges = df['Exchange'].unique()
                    timezone_info = []
                    for exchange in exchanges:
                        if exchange in ['NSE', 'BSE']:
                            timezone_info.append(f"**{exchange}**: Times shown in IST (UTC+5:30)")
                        elif exchange in ['NASDAQ', 'NYSE']:
                            timezone_info.append(f"**{exchange}**: Times shown in EST (UTC-5)")
                        elif exchange == 'BINANCE':
                            timezone_info.append(f"**{exchange}**: Times shown in UTC")
                    
                    if timezone_info:
                        st.info("⏰ " + " | ".join(timezone_info))
                
                # --- Display Summary Metrics ---
                st.subheader("📊 Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_patterns = len(df_display)
                    st.metric("Total Patterns", total_patterns)
                
                with col2:
                    valid_patterns = len(df_display[df_display['Pattern Status'] == '✅ Valid']) if 'Pattern Status' in df_display.columns else 0
                    st.metric("Valid Patterns", valid_patterns)
                
                with col3:
                    setup_ready = len(df_display[df_display['Trade Status'] == 'Setup Ready']) if 'Trade Status' in df_display.columns else 0
                    st.metric("Setup Ready", setup_ready)
                
                with col4:
                    ongoing = len(df_display[df_display['Trade Status'] == 'Ongoing']) if 'Trade Status' in df_display.columns else 0
                    st.metric("Ongoing Trades", ongoing)
                
                # --- Display Filters ---
                st.subheader("🔍 Filter and View Data")
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Filter by Symbol
                    unique_symbols = df_display['Symbol'].unique() if 'Symbol' in df_display.columns else []
                    selected_symbols = st.multiselect("Filter by Symbol", options=unique_symbols,
                                                      default=unique_symbols)

                with col2:
                    # Filter by Trade Status
                    unique_statuses = df_display['Trade Status'].unique() if 'Trade Status' in df_display.columns else []
                    selected_statuses = st.multiselect("Filter by Trade Status", options=unique_statuses,
                                                       default=unique_statuses)
                
                with col3:
                    # Filter by Pattern Status
                    if 'Pattern Status' in df_display.columns:
                        pattern_statuses = df_display['Pattern Status'].unique()
                        selected_pattern_status = st.multiselect("Filter by Pattern Status", 
                                                                options=pattern_statuses,
                                                                default=pattern_statuses)
                    else:
                        selected_pattern_status = []

                # Apply filters
                filtered_df = df_display.copy()
                if 'Symbol' in df_display.columns and selected_symbols:
                    filtered_df = filtered_df[filtered_df['Symbol'].isin(selected_symbols)]
                if 'Trade Status' in df_display.columns and selected_statuses:
                    filtered_df = filtered_df[filtered_df['Trade Status'].isin(selected_statuses)]
                if 'Pattern Status' in df_display.columns and selected_pattern_status:
                    filtered_df = filtered_df[filtered_df['Pattern Status'].isin(selected_pattern_status)]

                # Display the filtered data
                st.dataframe(filtered_df, use_container_width=True, height=600)

                # --- Download Button ---
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Filtered Results as CSV",
                    data=csv_data,
                    file_name=f"filtered_vcp_results_{datetime.now().strftime('%Y%m%d_%H%M')}_IST.csv",
                    mime='text/csv',
                )
                
                # --- Additional Analysis ---
                with st.expander("📈 Additional Analysis"):
                    if 'Trade Status' in df_display.columns:
                        st.subheader("Trade Status Distribution")
                        status_counts = df_display['Trade Status'].value_counts()
                        st.bar_chart(status_counts)
                    
                    if 'Symbol' in df_display.columns:
                        st.subheader("Patterns by Symbol")
                        symbol_counts = df_display['Symbol'].value_counts()
                        st.bar_chart(symbol_counts)

        except Exception as e:
            st.error(f"An error occurred while reading the results file: {e}")
            st.info("The CSV file might be corrupted or in an unexpected format.")
            st.code(str(e), language='text')  # Show detailed error for debugging

    else:
        st.warning("⚠️ No analysis file found.")
        st.info(
            f"The analysis runs automatically every hour. The file `{output_filename}` will appear after the next run.")
        st.info(
            f"You can also manually trigger an analysis by running: `python vcp_classic_csv.py --headless`")
        
        # Show next run times in IST
        st.subheader("📅 Scheduled Run Times (IST)")
        current_time = datetime.now() + timedelta(hours=5, minutes=30)  # Convert to IST
        st.write(f"Current time: **{current_time.strftime('%H:%M IST')}**")
        st.write("The analyzer runs automatically:")
        st.write("- Every hour (e.g., 13:00 IST, 14:00 IST, 15:00 IST...)")
        st.write("- Special runs at 9:00 AM and 3:00 PM IST on weekdays")


# --- Main Application Logic ---
if check_password():
    display_results_page()
