import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob

# Try to import contextily for ESRI basemaps
try:
    import contextily as ctx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False
    st.warning("Install contextily for ESRI basemaps: pip install contextily")

# Page configuration
st.set_page_config(
    page_title="Maputo Flood Risk Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E40AF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .scenario-header {
        font-size: 1.5rem;
        color: #374151;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
    }
    .risk-high { color: #DC2626; font-weight: bold; }
    .risk-medium { color: #D97706; font-weight: bold; }
    .risk-low { color: #059669; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 24px;
        padding-right: 24px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_raster_data(file_path):
    """Load and cache raster data"""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            bounds = src.bounds
            transform = src.transform
            crs = src.crs
        return data, bounds, transform, crs
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None, None, None, None

def align_flood_data_to_urban(flood_data, urban_data):
    """Align flood data to match urban data dimensions"""
    if flood_data is None or urban_data is None:
        return flood_data
    
    # Check if dimensions match
    if flood_data.shape == urban_data.shape:
        return flood_data
    
    # If dimensions don't match, we need to resample
    try:
        from scipy.ndimage import zoom
        
        # Calculate zoom factors for each dimension
        zoom_factors = tuple(urban_data.shape[i] / flood_data.shape[i] for i in range(len(flood_data.shape)))
        
        # Resample flood data to match urban data
        aligned_flood = zoom(flood_data, zoom_factors, order=1)  # Linear interpolation
        
        return aligned_flood
        
    except ImportError:
        # Fallback: simple reshape (may distort data)
        try:
            aligned_flood = np.resize(flood_data, urban_data.shape)
            return aligned_flood
        except:
            return flood_data

@st.cache(allow_output_mutation=True)
def find_flood_files(data_path, flood_type, year, scenario, frequency):
    """Find flood files based on naming pattern"""
    if year == 2020:
        # Historical files
        pattern = f"{flood_type}_flooding_Historical_{frequency}.tif"
    else:
        # Future files
        pattern = f"{flood_type}_flooding_Future_{year}_{frequency}_{scenario}.tif"
        # Also try coastal naming pattern
        if flood_type.lower() == "coastal":
            alt_pattern = f"coastal_map_{frequency}_{year}_{scenario}.tif"
            alt_file_path = os.path.join(data_path, alt_pattern)
            if os.path.exists(alt_file_path):
                return alt_file_path
    
    file_path = os.path.join(data_path, pattern)
    
    if os.path.exists(file_path):
        return file_path
    else:
        # Try alternative patterns or search
        search_pattern = os.path.join(data_path, f"*{flood_type.lower()}*{frequency}*{year}*.tif")
        matches = glob.glob(search_pattern)
        if matches:
            return matches[0]
    
    return None

@st.cache(allow_output_mutation=True)
def find_urban_files(data_path, year, scenario):
    """Find urban files based on naming pattern"""
    if year <= 2024:
        # Use historical timeline data
        return os.path.join(data_path, "settlement_timeline_1985_2024_geo.tif")
    else:
        # Future urban projections
        pattern = f"Final urban_{year}_{scenario}.tif"
        file_path = os.path.join(data_path, pattern)
        if os.path.exists(file_path):
            return file_path
    
    return None

def load_urban_data(data_path, urban_year, flood_year, scenario):
    """Load appropriate urban data based on year and scenario"""
    if urban_year <= 2024:
        # Use historical timeline data
        timeline_file = os.path.join(data_path, "settlement_timeline_1985_2024_geo.tif")
        if os.path.exists(timeline_file):
            timeline_data, bounds, transform, crs = load_raster_data(timeline_file)
            if timeline_data is not None:
                urban_mask = (timeline_data > 0) & (timeline_data <= urban_year)
                return urban_mask.astype(np.uint8), bounds, transform, crs, timeline_data
    else:
        # Use future urban projections
        urban_file = find_urban_files(data_path, urban_year, scenario)
        if urban_file:
            urban_data, bounds, transform, crs = load_raster_data(urban_file)
            if urban_data is not None:
                urban_mask = urban_data > 0
                return urban_mask.astype(np.uint8), bounds, transform, crs, None
    
    return None, None, None, None, None

def create_enhanced_urban_layer(urban_data, intensity=2.0):
    """Create enhanced urban layer with better visibility"""
    if urban_data is None:
        return None
    
    # Create enhanced urban mask
    urban_enhanced = urban_data.astype(float) * intensity
    urban_enhanced[urban_data == 0] = np.nan
    return urban_enhanced

def create_flood_depth_colormap():
    """Create standardized blue colormap for flood depths"""
    colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    n_bins = 256
    return mcolors.LinearSegmentedColormap.from_list('flood_depth', colors, N=n_bins)

def create_dual_flood_models(data_dict, flood_year, urban_year, urban_alpha, flood_alpha, flood_type, ssp_scenario, flood_threshold_m=0.1, show_threshold_only=False):
    """Create side-by-side flood model visualizations with real data"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # Update title to show both years
    title = f'{flood_type} Flood Risk Models - Flood: {flood_year}, Urban: {urban_year} ({ssp_scenario})'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    urban_data = data_dict.get('urban')
    bounds = data_dict.get('bounds')
    
    # Load real flood data
    frequent_flood_data = data_dict.get('frequent_flood')
    rare_flood_data = data_dict.get('rare_flood')
    
    # Align flood data to urban grid
    if frequent_flood_data is not None:
        frequent_flood_data = align_flood_data_to_urban(frequent_flood_data, urban_data)
    if rare_flood_data is not None:
        rare_flood_data = align_flood_data_to_urban(rare_flood_data, urban_data)
    
    # Set up proper extent for basemap
    if bounds:
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    else:
        extent = None
    
    # Add ESRI basemaps if contextily is available
    if CONTEXTILY_AVAILABLE and bounds is not None:
        try:
            # Set the axis limits
            ax1.set_xlim(bounds.left, bounds.right)
            ax1.set_ylim(bounds.bottom, bounds.top)
            ax2.set_xlim(bounds.left, bounds.right)
            ax2.set_ylim(bounds.bottom, bounds.top)
            
            # Add ESRI World Imagery basemap to both panels
            ctx.add_basemap(ax1, crs='EPSG:4326', source=ctx.providers.Esri.WorldImagery, alpha=0.8)
            ctx.add_basemap(ax2, crs='EPSG:4326', source=ctx.providers.Esri.WorldImagery, alpha=0.8)
            
        except Exception as e:
            # Fallback to terrain background
            if frequent_flood_data is not None:
                terrain_bg = np.ones_like(frequent_flood_data) * 0.95
                texture = np.random.random(frequent_flood_data.shape) * 0.05
                terrain_bg = terrain_bg + texture
                ax1.imshow(terrain_bg, cmap='terrain', alpha=0.6, vmin=0.85, vmax=1.0, extent=extent)
                ax2.imshow(terrain_bg, cmap='terrain', alpha=0.6, vmin=0.85, vmax=1.0, extent=extent)
    else:
        # Fallback terrain background
        if frequent_flood_data is not None:
            terrain_bg = np.ones_like(frequent_flood_data) * 0.95
            texture = np.random.random(frequent_flood_data.shape) * 0.05
            terrain_bg = terrain_bg + texture
            ax1.imshow(terrain_bg, cmap='terrain', alpha=0.6, vmin=0.85, vmax=1.0, extent=extent)
            ax2.imshow(terrain_bg, cmap='terrain', alpha=0.6, vmin=0.85, vmax=1.0, extent=extent)
    
    # Get standardized blue colormap
    flood_cmap = create_flood_depth_colormap()
    
    # Left panel - Frequent Scenario
    if frequent_flood_data is not None:
        # Enhanced urban layer with brown/tan colormap - render first (underneath)
        if urban_data is not None:
            urban_enhanced = create_enhanced_urban_layer(urban_data, intensity=3.0)
            urban_colors = mcolors.LinearSegmentedColormap.from_list(
                'urban_brown', ['#D2B48C', '#A0522D', '#8B4513'], N=256  # Tan to dark brown
            )
            ax1.imshow(urban_enhanced, cmap=urban_colors, alpha=urban_alpha, extent=extent)
        
        # Process flood depth data with threshold - render on top
        frequent_display = frequent_flood_data.copy().astype(float)
        
        if show_threshold_only:
            # Only show areas above threshold
            frequent_display[frequent_display < flood_threshold_m] = np.nan
        else:
            # Show all flood areas, but use threshold for no-flood
            frequent_display[frequent_display <= 0] = np.nan
        
        # Fixed depth range: 0-2m
        depth_range = 2.0
        
        im1 = ax1.imshow(frequent_display, cmap=flood_cmap, alpha=flood_alpha, 
                        vmin=0, vmax=depth_range, extent=extent)
        
        ax1.set_title(f'Frequent Scenario\nFlood: {flood_year}, Urban: {urban_year}', fontsize=12, fontweight='bold')
        
        # Set proper axis limits and labels
        if bounds:
            ax1.set_xlim(bounds.left, bounds.right)
            ax1.set_ylim(bounds.bottom, bounds.top)
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
        
        # Colorbar for frequent scenario
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7, aspect=25)
        cbar1.set_label('Flood Depth (m)', fontsize=10)
    
    # Right panel - Rare Scenario
    if rare_flood_data is not None:
        # Enhanced urban layer with brown/tan colormap - render first (underneath)
        if urban_data is not None:
            urban_enhanced = create_enhanced_urban_layer(urban_data, intensity=3.0)
            urban_colors = mcolors.LinearSegmentedColormap.from_list(
                'urban_brown', ['#D2B48C', '#A0522D', '#8B4513'], N=256  # Tan to dark brown
            )
            ax2.imshow(urban_enhanced, cmap=urban_colors, alpha=urban_alpha, extent=extent)
        
        # Process flood depth data with threshold - render on top
        rare_display = rare_flood_data.copy().astype(float)
        
        if show_threshold_only:
            # Only show areas above threshold
            rare_display[rare_display < flood_threshold_m] = np.nan
        else:
            # Show all flood areas, but use threshold for no-flood
            rare_display[rare_display <= 0] = np.nan
        
        # Use same fixed depth range: 0-2m
        depth_range = 2.0
        
        im2 = ax2.imshow(rare_display, cmap=flood_cmap, alpha=flood_alpha, 
                        vmin=0, vmax=depth_range, extent=extent)
        
        ax2.set_title(f'Rare Scenario\nFlood: {flood_year}, Urban: {urban_year}', fontsize=12, fontweight='bold')
        
        # Set proper axis limits and labels
        if bounds:
            ax2.set_xlim(bounds.left, bounds.right)
            ax2.set_ylim(bounds.bottom, bounds.top)
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
        
        # Colorbar for rare scenario
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7, aspect=25)
        cbar2.set_label('Flood Depth (m)', fontsize=10)
    
    # Add statistics for both scenarios with larger, more prominent boxes
    if urban_data is not None:
        urban_mask = urban_data > 0
        total_urban = np.sum(urban_mask)
        
        # Calculate flood exposure statistics using dynamic threshold
        if frequent_flood_data is not None:
            frequent_urban_depths = frequent_flood_data[urban_mask]
            frequent_flooded = np.sum(frequent_urban_depths > flood_threshold_m)  # Use dynamic threshold
            frequent_pct = (frequent_flooded / total_urban * 100) if total_urban > 0 else 0
        else:
            frequent_pct = 0
        
        if rare_flood_data is not None:
            rare_urban_depths = rare_flood_data[urban_mask]
            rare_flooded = np.sum(rare_urban_depths > flood_threshold_m)  # Use dynamic threshold
            rare_pct = (rare_flooded / total_urban * 100) if total_urban > 0 else 0
        else:
            rare_pct = 0
        
        # Add stats boxes with larger, more prominent styling and better positioning
        box_props = dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.95, 
                        edgecolor='black', linewidth=2)
        
        # Include threshold in the stats display - removed average depth
        threshold_cm = int(flood_threshold_m * 100)
        frequent_stats = f'Urban: {total_urban * 0.01:.1f} km²\nFlooded (>{threshold_cm}cm): {frequent_pct:.1f}%'
        ax1.text(0.03, 0.97, frequent_stats, transform=ax1.transAxes,
                bbox=box_props, verticalalignment='top', fontsize=15, fontweight='bold')
        
        rare_stats = f'Urban: {total_urban * 0.01:.1f} km²\nFlooded (>{threshold_cm}cm): {rare_pct:.1f}%'
        ax2.text(0.03, 0.97, rare_stats, transform=ax2.transAxes,
                bbox=box_props, verticalalignment='top', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 Maputo Flood Risk Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**SSP Climate Scenarios: Frequent vs Rare Flood Events**")
    
    # Sidebar controls
    st.sidebar.header("📊 Analysis Controls")
    
    # Auto-detect if running locally or on cloud
    if os.path.exists("/Users/mauritz/Documents/GitHub/G20_hackathon/dashboard_data/"):
        default_path = "/Users/mauritz/Documents/GitHub/G20_hackathon/dashboard_data/"
    else:
        default_path = "./data/"  # Relative path for cloud deployment
    
    # Data directory path
    data_path = st.sidebar.text_input(
        "Data Directory Path:", 
        value=default_path,
        help="Path to the directory containing your exported raster files"
    )
    
    # SSP scenario selection with proper names
    st.sidebar.subheader("🌡️ Climate Scenario")
    ssp_display = st.sidebar.radio(
        "SSP Scenario:",
        ["SSP 2-4.5", "SSP 5-8.5"],
        help="SSP 2-4.5: Middle-of-the-road, SSP 5-8.5: High emissions pathway"
    )
    
    # Convert display names to file naming convention
    ssp_scenario = "SSP45" if ssp_display == "SSP 2-4.5" else "SSP85"
    
    scenario_descriptions = {
        "SSP 2-4.5": "**SSP 2-4.5**: Middle-of-the-road scenario with moderate climate change",
        "SSP 5-8.5": "**SSP 5-8.5**: High emissions scenario with severe climate change"
    }
    st.sidebar.info(scenario_descriptions[ssp_display])
    
    # Flood type selection
    st.sidebar.subheader("🌊 Flood Model Selection")
    flood_type = st.sidebar.radio(
        "Flood Type:",
        ["Coastal", "Riverine"],
        help="Choose which flood model to analyze"
    )
    
    # Flood depth threshold control
    st.sidebar.subheader("🌊 Flood Threshold")
    flood_threshold_cm = st.sidebar.number_input(
        "Minimum Flood Depth (cm):",
        min_value=1,
        max_value=100,
        value=10,
        step=5,
        help="Minimum water depth to consider as 'flooded' for statistics and visualization"
    )
    flood_threshold_m = flood_threshold_cm / 100.0  # Convert to meters
    
    # Optional: Threshold visualization toggle
    show_threshold_only = st.sidebar.checkbox(
        "Show Only Above Threshold",
        value=False,
        help="Display only flood areas above the threshold (hide areas below threshold)"
    )
    
    # Timeline Controls
    st.sidebar.subheader("🕒 Timeline Controls")
    
    # Flood Timeline Slider
    flood_years = [2020, 2030, 2040, 2050]
    selected_flood_year = st.sidebar.select_slider(
        "Flood Model Year:",
        options=flood_years,
        value=2020,
        help=f"Select flood model year (Historical: 2020, Future: {flood_type} projections)"
    )
    
    # Urbanization Timeline Slider - always include future projections
    historical_years = list(range(1985, 2025, 5)) + [2024]
    future_years = [2030, 2040, 2050]
    urban_years = historical_years + future_years
    
    selected_urban_year = st.sidebar.select_slider(
        "Urban Development Year:",
        options=urban_years,
        value=2020,  # Fixed default value, independent of flood year
        help=f"Urban development timeline (Historical: 1985-2024, Future: {ssp_display} projections)"
    )
    
    # Layer transparency controls
    st.sidebar.subheader("🎨 Layer Controls")
    urban_alpha = st.sidebar.slider("Urban Layer Opacity", 0.1, 1.0, 0.9, 0.1)
    flood_alpha = st.sidebar.slider("Flood Layer Opacity", 0.1, 1.0, 0.7, 0.1)
    
    # Load data
    data_dict = {}
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        st.error(f"Data directory not found: {data_path}")
        return
    
    # Load urban data based on both years and scenario
    urban_data, bounds, transform, crs, timeline_data = load_urban_data(
        data_path, selected_urban_year, selected_flood_year, ssp_scenario
    )
    
    if urban_data is not None:
        data_dict['urban'] = urban_data
        data_dict['bounds'] = bounds
        data_dict['timeline'] = timeline_data
    else:
        st.error(f"Could not load urban data for year {selected_urban_year}")
        return
    
    # Load flood data
    frequent_file = find_flood_files(data_path, flood_type, selected_flood_year, ssp_scenario, "Frequent")
    rare_file = find_flood_files(data_path, flood_type, selected_flood_year, ssp_scenario, "Rare")
    
    if frequent_file:
        frequent_data, _, _, _ = load_raster_data(frequent_file)
        data_dict['frequent_flood'] = frequent_data
    else:
        data_dict['frequent_flood'] = None
    
    if rare_file:
        rare_data, _, _, _ = load_raster_data(rare_file)
        data_dict['rare_flood'] = rare_data
    else:
        data_dict['rare_flood'] = None
    
    # Main visualization
    if bounds is not None:
        # Enhanced header showing both timelines
        st.markdown(f'<h2 class="scenario-header">{flood_type} Flood Risk Analysis</h2>', unsafe_allow_html=True)
        
        # Timeline summary
        timeline_info = f"**Flood Model:** {selected_flood_year} ({ssp_display}) | **Urban Development:** {selected_urban_year}"
        if selected_urban_year > 2024:
            timeline_info += f" ({ssp_display} projection)"
        elif selected_urban_year <= 2024:
            timeline_info += " (historical)"
        
        st.markdown(timeline_info)
        
        # Create and display dual flood models with threshold parameters
        dual_fig = create_dual_flood_models(
            data_dict, selected_flood_year, selected_urban_year, 
            urban_alpha, flood_alpha, flood_type, ssp_display,
            flood_threshold_m, show_threshold_only
        )
        st.pyplot(dual_fig)
        plt.close(dual_fig)
        
        # Summary insights section
        st.markdown("---")
        
        # Create insights based on temporal combinations
        if selected_urban_year < selected_flood_year:
            insight_text = f"**Analysis Insight**: Showing how {selected_urban_year} urban development would be exposed to future {selected_flood_year} flood conditions under {ssp_display} scenario."
            st.info(insight_text)
        elif selected_urban_year > selected_flood_year:
            insight_text = f"**Analysis Insight**: Showing how future {selected_urban_year} urban growth (under {ssp_display}) would be exposed to {selected_flood_year} flood conditions."
            st.warning(insight_text)
        else:
            insight_text = f"**Analysis Insight**: Synchronized analysis of both urban development and flood risk for {selected_flood_year} under {ssp_display} scenario."
            st.success(insight_text)
        
        # Climate scenario context
        scenario_info = {
            "SSP 2-4.5": {
                "description": "Middle-of-the-road pathway with moderate climate action",
                "temp_rise": "~2.4°C by 2100",
                "implications": "Manageable but significant risk increases"
            },
            "SSP 5-8.5": {
                "description": "High emissions pathway with limited climate action", 
                "temp_rise": "~4.4°C by 2100",
                "implications": "Severe risk increases requiring major adaptation"
            }
        }
        
        selected_info = scenario_info[ssp_display]
        st.markdown(f"**{ssp_display} Scenario Context**: {selected_info['description']} | {selected_info['temp_rise']} | {selected_info['implications']}")
    
    else:
        st.error("Could not load data files. Please check file paths and data availability.")
        
        # Show helpful debugging information
        st.subheader("Data Availability Check")
        
        # Check what files exist
        if os.path.exists(data_path):
            st.write("**Available files in data directory:**")
            files = os.listdir(data_path)
            for file in sorted(files):
                if file.endswith('.tif'):
                    st.write(f"• {file}")
        else:
            st.error(f"Data directory does not exist: {data_path}")
    
    # Footer with enhanced information
    st.markdown("---")
    st.markdown("**Enhanced Dashboard**: Dual timeline analysis with independent flood and urban development controls")
    st.markdown(f"**Current Analysis**: {flood_type} flood risk ({selected_flood_year}) vs Urban development ({selected_urban_year}) under {ssp_display} scenario")

if __name__ == "__main__":
    main()