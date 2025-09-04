"""
Nanocarrier Design Page

This module provides functionality for designing and evaluating nanocarriers
for drug delivery based on the properties of the target and ligands.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import yaml
import json
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Nanocarrier Design - AI Drug Target Platform",
    page_icon="🚀"
)

# Check if we have docking results
if 'docking_results' not in st.session_state:
    st.warning("⚠️ Please run ligand docking first.")
    if st.button("Go to Ligand Docking"):
        st.switch_page("pages/5_Ligand_Docking.py")
    st.stop()

class NanocarrierDesigner:
    """Class for designing and evaluating nanocarriers for drug delivery."""
    
    def __init__(self):
        self.available_materials = [
            "Liposomes", "Polymeric Nanoparticles", "Dendrimers", 
            "Micelles", "Gold Nanoparticles", "Silica Nanoparticles"
        ]
        
        # Load nanocarrier properties from YAML (in a real app, this would be a file)
        self.carrier_properties = {
            "Liposomes": {
                "size": "50-200 nm",
                "loading_capacity": "High",
                "release_kinetics": "Sustained",
                "stability": "Moderate",
                "surface_modification": "Easy",
                "biocompatibility": "Excellent",
                "cost": "$$",
                "description": "Spherical vesicles with a hydrophilic core and lipid bilayer, ideal for both hydrophobic and hydrophilic drugs."
            },
            "Polymeric Nanoparticles": {
                "size": "20-200 nm",
                "loading_capacity": "High",
                "release_kinetics": "Controlled",
                "stability": "High",
                "surface_modification": "Moderate",
                "biocompatibility": "Good",
                "cost": "$$",
                "description": "Biodegradable polymers that can encapsulate drugs and provide controlled release."
            },
            "Dendrimers": {
                "size": "1-15 nm",
                "loading_capacity": "Moderate",
                "release_kinetics": "Controlled",
                "stability": "High",
                "surface_modification": "Excellent",
                "biocompatibility": "Variable",
                "cost": "$$$",
                "description": "Highly branched, monodisperse macromolecules with multiple functional groups for drug conjugation."
            },
            "Micelles": {
                "size": "10-100 nm",
                "loading_capacity": "Moderate",
                "release_kinetics": "Fast",
                "stability": "Low",
                "surface_modification": "Moderate",
                "biocompatibility": "Good",
                "cost": "$",
                "description": "Amphiphilic molecules that self-assemble in aqueous solutions, suitable for hydrophobic drugs."
            },
            "Gold Nanoparticles": {
                "size": "5-100 nm",
                "loading_capacity": "Low",
                "release_kinetics": "Controlled",
                "stability": "Excellent",
                "surface_modification": "Excellent",
                "biocompatibility": "Good",
                "cost": "$$$",
                "description": "Inert metal nanoparticles with unique optical properties, suitable for theranostic applications."
            },
            "Silica Nanoparticles": {
                "size": "20-200 nm",
                "loading_capacity": "High",
                "release_kinetics": "Controlled",
                "stability": "Excellent",
                "surface_modification": "Excellent",
                "biocompatibility": "Good",
                "cost": "$$",
                "description": "Porous nanoparticles with high surface area for drug loading and controlled release."
            }
        }
        
    def recommend_carriers(self, ligand_props: Dict) -> List[Dict]:
        """Recommend nanocarriers based on ligand properties."""
        # In a real implementation, this would use ML or rule-based matching
        # For now, we'll return all carriers with a recommendation score
        
        recommendations = []
        
        for material in self.available_materials:
            # Simple scoring based on ligand properties and carrier characteristics
            score = 0.5  # Base score
            
            # Adjust score based on ligand properties
            if ligand_props['logp'] > 3:  # Hydrophobic ligand
                if material in ["Liposomes", "Micelles"]:
                    score += 0.3
            else:  # Hydrophilic ligand
                if material in ["Dendrimers", "Silica Nanoparticles"]:
                    score += 0.3
            
            # Add some randomness for demo purposes
            score += np.random.uniform(-0.1, 0.1)
            score = max(0.1, min(1.0, score))  # Clamp between 0.1 and 1.0
            
            recommendations.append({
                'material': material,
                'score': score,
                'properties': self.carrier_properties[material]
            })
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    def get_carrier_comparison_chart(self, recommendations: List[Dict]) -> go.Figure:
        """Generate a radar chart comparing different nanocarriers."""
        categories = ['Loading', 'Stability', 'Release Control', 'Modifiability', 'Biocompatibility']
        
        # Map properties to radar chart values
        property_map = {
            'Loading': 'loading_capacity',
            'Stability': 'stability',
            'Release Control': 'release_kinetics',
            'Modifiability': 'surface_modification',
            'Biocompatibility': 'biocompatibility'
        }
        
        # Convert qualitative properties to numerical scores
        qual_to_num = {
            'Low': 1, 'Moderate': 2, 'High': 3, 'Excellent': 4,
            'Fast': 1, 'Sustained': 2, 'Controlled': 3,
            'Easy': 3, 'Moderate': 2, 'Difficult': 1
        }
        
        # Prepare data for the radar chart
        fig = go.Figure()
        
        # Add a trace for each recommended carrier (top 3)
        for i, rec in enumerate(recommendations[:3]):
            material = rec['material']
            props = rec['properties']
            
            # Convert qualitative properties to numerical scores
            values = [
                qual_to_num.get(props[property_map[cat]], 0) 
                for cat in categories
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=categories + [categories[0]],
                name=material,
                line=dict(color=px.colors.qualitative.Plotly[i]),
                hoverinfo='text',
                hovertext=f"<b>{material}</b><br>" +
                         f"Score: {rec['score']:.2f}<br>" +
                         f"Size: {props['size']}<br>" +
                         f"Cost: {props['cost']}"
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 4],
                    tickvals=[1, 2, 3, 4],
                    ticktext=['Low', 'Moderate', 'High', 'Excellent']
                )
            ),
            title="Nanocarrier Comparison",
            showlegend=True,
            height=600,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig

# Initialize nanocarrier designer
designer = NanocarrierDesigner()

def main():
    st.title("🚀 Nanocarrier Design")
    st.markdown("---")
    
    # Get the docking results
    docking_data = st.session_state.docking_results
    
    # Display target and ligand info
    st.subheader("🎯 Target & Ligand Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Target", docking_data['target'])
        st.metric("Docking Program", docking_data['program'])
    with col2:
        # Get the top ligand
        top_ligand = min(docking_data['results'], key=lambda x: x['score'])
        st.metric("Top Ligand", top_ligand['ligand_name'])
        st.metric("Docking Score", f"{top_ligand['score']:.2f} kcal/mol")
    
    # Ligand properties (in a real app, these would be calculated)
    st.subheader("🧪 Ligand Properties")
    
    # Generate some dummy ligand properties
    ligand_props = {
        'mw': np.random.uniform(200, 600),
        'logp': np.random.uniform(-2, 5),
        'hbd': np.random.randint(0, 5),
        'hba': np.random.randint(1, 8),
        'tpsa': np.random.uniform(20, 120),
        'rotatable_bonds': np.random.randint(0, 10),
        'aromatic_rings': np.random.randint(1, 4)
    }
    
    # Display ligand properties
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Molecular Weight", f"{ligand_props['mw']:.1f} g/mol")
        st.metric("H-Bond Donors", ligand_props['hbd'])
    with col2:
        st.metric("LogP", f"{ligand_props['logp']:.2f}")
        st.metric("H-Bond Acceptors", ligand_props['hba'])
    with col3:
        st.metric("TPSA", f"{ligand_props['tpsa']:.1f} Å²")
        st.metric("Rotatable Bonds", ligand_props['rotatable_bonds'])
    with col4:
        st.metric("Aromatic Rings", ligand_props['aromatic_rings'])
    
    # Nanocarrier design
    st.subheader("🧬 Nanocarrier Recommendations")
    
    # Get recommendations
    recommendations = designer.recommend_carriers(ligand_props)
    
    # Display top recommendations
    st.markdown("### Top Recommended Nanocarriers")
    
    # Show top 3 recommendations as cards
    cols = st.columns(3)
    for i, rec in enumerate(recommendations[:3]):
        with cols[i]:
            with st.expander(f"**{i+1}. {rec['material']}** (Score: {rec['score']:.2f})", expanded=True):
                st.markdown(f"**Description:** {rec['properties']['description']}")
                st.markdown(f"**Size:** {rec['properties']['size']}")
                st.markdown(f"**Loading Capacity:** {rec['properties']['loading_capacity']}")
                st.markdown(f"**Release Kinetics:** {rec['properties']['release_kinetics']}")
                st.markdown(f"**Stability:** {rec['properties']['stability']}")
                st.markdown(f"**Surface Modification:** {rec['properties']['surface_modification']}")
                st.markdown(f"**Biocompatibility:** {rec['properties']['biocompatibility']}")
                st.markdown(f"**Cost:** {rec['properties']['cost']}")
    
    # Show comparison chart
    st.subheader("📊 Nanocarrier Comparison")
    
    fig = designer.get_carrier_comparison_chart(recommendations)
    st.plotly_chart(fig, use_container_width=True)
    
    # Custom nanocarrier design
    st.subheader("🎨 Custom Nanocarrier Design")
    
    # Initialize session state for nanocarrier design if it doesn't exist
    if 'nanocarrier_design' not in st.session_state:
        st.session_state.nanocarrier_design = None
    
    with st.form("custom_carrier_form"):
        st.markdown("### Design Your Own Nanocarrier")
        
        col1, col2 = st.columns(2)
        
        with col1:
            material = st.selectbox(
                "Base Material",
                options=designer.available_materials,
                index=0
            )
            
            size = st.slider(
                "Target Size (nm)",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
            
            surface_charge = st.select_slider(
                "Surface Charge",
                options=['Strongly Negative', 'Weakly Negative', 'Neutral', 'Weakly Positive', 'Strongly Positive'],
                value='Neutral'
            )
        
        with col2:
            targeting_ligand = st.multiselect(
                "Targeting Ligands",
                options=["None", "Antibodies", "Aptamers", "Peptides", "Folic Acid", "Transferrin"],
                default=["None"]
            )
            
            release_trigger = st.selectbox(
                "Release Trigger",
                options=["pH", "Enzyme", "Redox", "Temperature", "Light", "Ultrasound"],
                index=0
            )
            
            coating = st.selectbox(
                "Surface Coating",
                options=["None", "PEG", "Chitosan", "Hyaluronic Acid", "Polysorbate 80"],
                index=0
            )
        
        # Submit button
        submitted = st.form_submit_button("Design Nanocarrier")
        
        if submitted:
            st.session_state.nanocarrier_design = {
                'material': material,
                'size': size,
                'surface_charge': surface_charge,
                'targeting_ligands': targeting_ligand,
                'release_trigger': release_trigger,
                'coating': coating,
                'ligand': top_ligand['ligand_name'],
                'target': docking_data['target']
            }
    
    # Display the design and download button outside the form
    if st.session_state.nanocarrier_design:
        design = st.session_state.nanocarrier_design
        
        st.success("🎉 Nanocarrier design ready!")
        
        # Show a summary of the design
        with st.expander("📝 Design Summary", expanded=True):
            st.markdown("### Custom Nanocarrier Design")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Base Material:** {design['material']}")
                st.markdown(f"**Size:** {design['size']} nm")
                st.markdown(f"**Surface Charge:** {design['surface_charge']}")
            
            with col2:
                st.markdown(f"**Targeting Ligands:** {', '.join(design['targeting_ligands']) if design['targeting_ligands'] else 'None'}")
                st.markdown(f"**Release Trigger:** {design['release_trigger']}")
                st.markdown(f"**Surface Coating:** {design['coating']}")
        
        # Generate a mock-up of the nanocarrier
        st.markdown("### 🎨 Nanocarrier Visualization")
        
        # Create a simple visualization
        st.image(
            "https://via.placeholder.com/800x400?text=Custom+Nanocarrier+Visualization",
            use_column_width=True,
            caption=f"{design['material']} nanocarrier with {', '.join(design['targeting_ligands']) if design['targeting_ligands'] else 'no'} targeting ligands"
        )
        
        # Save design button (outside the form)
        st.download_button(
            label="💾 Save Nanocarrier Design",
            data=json.dumps(design, indent=2),
            file_name=f"nanocarrier_{design['target']}_{design['material'].lower().replace(' ', '_')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
