"""Target Analysis Page"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import logging
from utils import TargetProcessor
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Target Analysis - AI Drug Target Platform",
    page_icon="🔍"
)

# Check if data is loaded
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("⚠️ Please upload data in the 'Data Upload' section first.")
    if st.button("Go to Data Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

# Main function
def main():
    st.title("🔍 Target Analysis")
    st.markdown("---")
    
    df = st.session_state.df
    
    # Get default values from config
    default_min_logfc = config.get('analysis.default_min_logfc', 1.0)
    default_max_pval = config.get('analysis.default_max_pval', 0.05)
    top_n = config.get('analysis.top_n_targets', 50)
    
    # Create two columns for filters and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Filter Settings")
        
        # Get min/max values for sliders
        logfc_min, logfc_max = float(df['logfc'].min()), float(df['logfc'].max())
        pval_min, pval_max = float(df['adj.p.val'].min()), float(df['adj.p.val'].max())
        
        # Ensure we have valid ranges
        if abs(logfc_max - logfc_min) < 0.1:
            logfc_min, logfc_max = -5.0, 5.0
        
        # Create sliders
        logfc_threshold = st.slider(
            "Log2 Fold Change Threshold", 
            min_value=0.0, 
            max_value=max(5.0, abs(logfc_max)), 
            value=default_min_logfc,
            step=0.1,
            help="Minimum absolute log2 fold change for significant genes"
        )
        
        pval_threshold = st.slider(
            "Adjusted p-value Threshold", 
            min_value=0.0, 
            max_value=0.1, 
            value=default_max_pval,
            step=0.001,
            format="%.3f",
            help="Maximum adjusted p-value for significant genes"
        )
        
        # Filter button
        if st.button("Apply Filters", type="primary"):
            try:
                filtered_df = st.session_state.processor.filter_targets(
                    df, 
                    min_logfc=logfc_threshold, 
                    max_pval=pval_threshold
                )
                st.session_state.filtered_df = filtered_df
                st.session_state.filter_params = {
                    'logfc_threshold': logfc_threshold,
                    'pval_threshold': pval_threshold
                }
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error filtering data: {str(e)}")
                logger.exception("Error in target analysis")
    
    with col2:
        st.subheader("📊 Volcano Plot")
        
        # Show volcano plot
        try:
            fig = st.session_state.processor.create_volcano_plot(
                df,
                logfc_threshold=st.session_state.get('filter_params', {}).get('logfc_threshold', default_min_logfc),
                pval_threshold=st.session_state.get('filter_params', {}).get('pval_threshold', default_max_pval)
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error generating volcano plot: {str(e)}")
    
    # Show filtered results if available
    if st.session_state.filtered_df is not None and not st.session_state.filtered_df.empty:
        filtered_df = st.session_state.filtered_df
        
        st.subheader(f"🎯 Filtered Targets (n={len(filtered_df)})")
        
        # Show top targets
        st.dataframe(
            filtered_df.head(top_n).style.background_gradient(
                subset=['score'], 
                cmap='YlOrRd'
            ),
            height=400,
            use_container_width=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Results",
            data=csv,
            file_name="filtered_targets.csv",
            mime="text/csv"
        )
        
        # Show some statistics
        st.subheader("📈 Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Targets", len(filtered_df))
        with col2:
            st.metric("Mean Score", f"{filtered_df['score'].mean():.2f}")
        with col3:
            st.metric("Top Score", f"{filtered_df['score'].max():.2f}")
    else:
        st.info("ℹ️ Apply filters to see the filtered results")

# Run the app
if __name__ == "__main__":
    main()
