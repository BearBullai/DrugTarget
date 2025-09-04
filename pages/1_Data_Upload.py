"""Data Upload Page"""
import streamlit as st
import pandas as pd
from pathlib import Path
import logging
from utils import TargetProcessor
from config import config, ensure_dir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Data Upload - AI Drug Target Platform",
    page_icon="📤"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'processor' not in st.session_state:
    st.session_state.processor = TargetProcessor(Path("data"))

# Main function
def main():
    st.title("📤 Data Upload")
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your DEG list (CSV or Excel)", 
        type=config.get('data.allowed_extensions', ['.csv', '.xlsx', '.xls'])
    )
    
    if uploaded_file is not None:
        try:
            # Process the uploaded file
            df = st.session_state.processor.load_data(uploaded_file)
            df = st.session_state.processor.preprocess_data(df)
            
            # Update session state
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.filtered_df = None
            
            # Show success message
            st.success("✅ Data loaded and preprocessed successfully!")
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head())
                
                # Basic stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Genes", len(df))
                with col2:
                    st.metric("Mean |logFC|", f"{abs(df['logfc']).mean():.2f}")
                with col3:
                    st.metric("Mean p-value", f"{df['adj.p.val'].mean():.3f}")
            
            # Save the uploaded file
            save_path = Path("data") / "deg_list.csv"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            
            # Show file info
            st.info(f"💾 Data saved to: `{save_path}`")
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            logger.exception("Error in data upload")
    else:
        st.info("ℹ️ Please upload a file to get started")
        
        # Check if data already exists
        default_data = Path("data") / "deg_list.csv"
        if default_data.exists():
            if st.button("📂 Load existing data"):
                try:
                    df = pd.read_csv(default_data)
                    df = st.session_state.processor.preprocess_data(df)
                    
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.session_state.filtered_df = None
                    
                    st.success("✅ Existing data loaded successfully!")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"❌ Error loading existing data: {str(e)}")
                    logger.exception("Error loading existing data")

# Run the app
if __name__ == "__main__":
    main()
