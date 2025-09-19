import streamlit as st
import pandas as pd
import os
from datetime import datetime

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
            st.success(
                f"Displaying results from the last analysis on: **{last_modified_time.strftime('%Y-%m-%d %H:%M:%S')}**")

            # Read the CSV file into a DataFrame
            df = pd.read_csv(output_filename)

            if df.empty:
                st.info("The last analysis ran successfully but found no VCP patterns.")
            else:
                # --- Display Filters ---
                st.subheader("Filter and View Data")
                col1, col2 = st.columns(2)
                with col1:
                    # Filter by Symbol
                    unique_symbols = df['Symbol'].unique()
                    selected_symbols = st.multiselect("Filter by Symbol", options=unique_symbols,
                                                      default=unique_symbols)

                with col2:
                    # Filter by Trade Status
                    unique_statuses = df['Trade Status'].unique()
                    selected_statuses = st.multiselect("Filter by Trade Status", options=unique_statuses,
                                                       default=unique_statuses)

                # Apply filters
                filtered_df = df[df['Symbol'].isin(selected_symbols) & df['Trade Status'].isin(selected_statuses)]

                # Display the filtered data
                st.dataframe(filtered_df, use_container_width=True, height=600)

                # --- Download Button ---
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Filtered Results as CSV",
                    data=csv_data,
                    file_name=f"filtered_vcp_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"An error occurred while reading the results file: {e}")
            st.info("The CSV file might be corrupted or in an unexpected format.")

    else:
        st.warning("⚠️ No analysis file found.")
        st.info(
            f"Please run the analyzer (`vcp_auto_analyzer.py --headless`) to generate the `{output_filename}` file.")


# --- Main Application Logic ---
if check_password():
    display_results_page()
