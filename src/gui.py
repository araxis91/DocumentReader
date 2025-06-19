"""
GUI interface for DocumentReader application using Streamlit.
"""

import streamlit as st
import tempfile
from pathlib import Path
import json
import pandas as pd
from typing import Optional, List
import time
import sys

# Add src directory to path for imports
if __name__ == "__main__":
    src_path = Path(__file__).parent
    sys.path.insert(0, str(src_path))

try:
    from cli import DocumentProcessor
    from config import APP_NAME, VERSION, OCR_CONFIG, TABLE_CONFIG
except ImportError:
    # Fallback for relative imports
    from .cli import DocumentProcessor
    from .config import APP_NAME, VERSION, OCR_CONFIG, TABLE_CONFIG


def launch_gui(pdf_file: Optional[Path] = None, output_dir: Optional[Path] = None):
    """
    Launch Streamlit GUI interface.
    
    Args:
        pdf_file: Optional PDF file to process
        output_dir: Optional output directory
    """
    st.set_page_config(
        page_title=f"{APP_NAME} v{VERSION}",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title
    st.title(f"🇺🇦 {APP_NAME}")
    st.markdown(f"*Version {VERSION} - Ukrainian PDF Document Processing*")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # OCR Settings
    st.sidebar.subheader("OCR Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold (%)",
        min_value=0,
        max_value=100,
        value=OCR_CONFIG['confidence_threshold'],
        help="Minimum confidence level for OCR results"
    )
    
    ocr_language = st.sidebar.selectbox(
        "OCR Language",
        options=['ukr+eng', 'ukr', 'eng'],
        index=0,
        help="Language for OCR processing"
    )
    
    # Table Extraction Settings
    st.sidebar.subheader("Table Extraction")
    table_method = st.sidebar.selectbox(
        "Table Method",
        options=['camelot', 'tabula'],
        index=0,
        help="Method for table extraction"
    )
    
    camelot_flavor = st.sidebar.selectbox(
        "Camelot Flavor",
        options=['stream', 'lattice'],
        index=0,
        help="Camelot extraction flavor (if using Camelot)"
    )
    
    # Output Settings
    st.sidebar.subheader("Output Settings")
    output_formats = st.sidebar.multiselect(
        "Table Output Formats",
        options=['csv', 'json', 'xlsx'],
        default=['csv', 'json'],
        help="Output formats for extracted tables"
    )
    
    verbose_logging = st.sidebar.checkbox(
        "Verbose Logging",
        value=False,
        help="Enable detailed logging"
    )
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📄 Process Document", "📊 Results", "🔍 System Check"])
    
    with tab1:
        st.header("Document Processing")
        
        # File upload
        if pdf_file is None:
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload a scanned PDF document in Ukrainian"
            )
        else:
            st.info(f"Processing file: {pdf_file}")
            uploaded_file = pdf_file
        
        # Output directory selection
        if output_dir is None:
            output_path = st.text_input(
                "Output Directory",
                value="./output",
                help="Directory to save processing results"
            )
        else:
            output_path = str(output_dir)
            st.info(f"Output directory: {output_path}")
        
        # Process button
        if st.button("🚀 Process Document", type="primary", disabled=uploaded_file is None):
            process_document_gui(
                uploaded_file,
                output_path,
                confidence_threshold,
                ocr_language,
                table_method,
                camelot_flavor,
                output_formats,
                verbose_logging
            )
    
    with tab2:
        st.header("Processing Results")
        display_results_tab(output_path)
    
    with tab3:
        st.header("System Check")
        display_system_check()


def process_document_gui(
    uploaded_file,
    output_path: str,
    confidence_threshold: int,
    ocr_language: str,
    table_method: str,
    camelot_flavor: str,
    output_formats: List[str],
    verbose_logging: bool
):
    """
    Process document through GUI interface.
    """
    # Update configuration
    OCR_CONFIG['confidence_threshold'] = confidence_threshold
    OCR_CONFIG['language'] = ocr_language
    TABLE_CONFIG['detection_method'] = table_method
    TABLE_CONFIG['camelot_flavor'] = camelot_flavor
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Handle file input
            if hasattr(uploaded_file, 'read'):
                tmp_file.write(uploaded_file.read())
            else:
                # It's a Path object
                with open(uploaded_file, 'rb') as f:
                    tmp_file.write(f.read())
            
            tmp_path = Path(tmp_file.name)
        
        # Initialize processor
        output_dir = Path(output_path)
        processor = DocumentProcessor(output_dir, verbose_logging)
        
        # Processing steps
        status_text.text("🔄 Initializing processing...")
        progress_bar.progress(10)
        
        status_text.text("🖼️ Converting PDF to images...")
        progress_bar.progress(30)
        
        status_text.text("🔤 Extracting text with OCR...")
        progress_bar.progress(60)
        
        status_text.text("📊 Detecting and extracting tables...")
        progress_bar.progress(80)
        
        # Process the document
        result = processor.process_single_file(tmp_path)
        
        progress_bar.progress(100)
        
        if result['success']:
            status_text.text("✅ Processing completed successfully!")
            
            # Display results
            st.success("🎉 Document processed successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Pages Processed", result['pages_processed'])
            
            with col2:
                avg_confidence = result['ocr_results']['average_confidence']
                st.metric("OCR Confidence", f"{avg_confidence:.1f}%")
            
            with col3:
                tables_found = result['table_results']['tables_found']
                st.metric("Tables Found", tables_found)
            
            # Processing details
            st.subheader("📋 Processing Details")
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.write("**OCR Results:**")
                st.write(f"- Successful pages: {result['ocr_results']['successful_pages']}")
                st.write(f"- Total characters: {result['ocr_results']['total_characters']:,}")
                st.write(f"- Total words: {result['ocr_results']['total_words']:,}")
            
            with details_col2:
                st.write("**Table Results:**")
                st.write(f"- Tables found: {result['table_results']['tables_found']}")
                st.write(f"- Valid tables: {result['table_results']['valid_tables']}")
                st.write(f"- Processing time: {result['processing_time']:.2f}s")
            
            # Output directory info
            st.info(f"📁 Results saved to: `{result['output_directory']}`")
            
            # Store result in session state for results tab
            st.session_state['last_result'] = result
            
        else:
            status_text.text("❌ Processing failed!")
            st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    finally:
        # Clean up temporary file
        if 'tmp_path' in locals():
            try:
                tmp_path.unlink()
            except:
                pass


def display_results_tab(output_path: str):
    """Display results in the results tab."""
    if 'last_result' not in st.session_state:
        st.info("No processing results yet. Process a document first.")
        return
    
    result = st.session_state['last_result']
    
    if not result.get('success', False):
        st.error("Last processing attempt failed.")
        return
    
    st.subheader("📊 Processing Summary")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pages", result['pages_processed'])
    
    with col2:
        avg_conf = result['ocr_results']['average_confidence']
        st.metric("OCR Confidence", f"{avg_conf:.1f}%")
    
    with col3:
        st.metric("Tables Found", result['table_results']['tables_found'])
    
    with col4:
        st.metric("Processing Time", f"{result['processing_time']:.1f}s")
    
    # File browser
    st.subheader("📁 Output Files")
    output_dir = Path(result['output_directory'])
    
    if output_dir.exists():
        files = list(output_dir.rglob('*'))
        
        if files:
            # Group files by type
            text_files = [f for f in files if f.suffix in ['.txt', '.json']]
            table_files = [f for f in files if f.suffix in ['.csv', '.xlsx']]
            
            tab_text, tab_tables = st.tabs(["📄 Text Files", "📊 Table Files"])
            
            with tab_text:
                if text_files:
                    selected_text_file = st.selectbox(
                        "Select text file to preview:",
                        text_files,
                        format_func=lambda x: x.name
                    )
                    
                    if selected_text_file and st.button("Preview Text File"):
                        display_text_file(selected_text_file)
                else:
                    st.info("No text files found.")
            
            with tab_tables:
                if table_files:
                    selected_table_file = st.selectbox(
                        "Select table file to preview:",
                        table_files,
                        format_func=lambda x: x.name
                    )
                    
                    if selected_table_file and st.button("Preview Table File"):
                        display_table_file(selected_table_file)
                else:
                    st.info("No table files found.")
        else:
            st.warning("No output files found.")
    else:
        st.error("Output directory not found.")


def display_text_file(file_path: Path):
    """Display text file content."""
    try:
        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            st.text_area("File Content", content, height=400)
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            st.json(content)
    except Exception as e:
        st.error(f"Failed to read file: {e}")


def display_table_file(file_path: Path):
    """Display table file content."""
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            st.dataframe(df)
        elif file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)
            st.dataframe(df)
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            if 'data' in content:
                df = pd.DataFrame(content['data'])
                st.dataframe(df)
            else:
                st.json(content)
    except Exception as e:
        st.error(f"Failed to read table file: {e}")


def display_system_check():
    """Display system check information."""
    st.subheader("🔧 System Requirements")
    
    # Check Python dependencies
    st.write("**Python Dependencies:**")
    dependencies = [
        'pdf2image', 'pytesseract', 'camelot', 'tabula', 
        'opencv-python', 'numpy', 'pandas', 'click', 'streamlit'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
            st.write(f"✅ {dep}")
        except ImportError:
            st.write(f"❌ {dep}")
    
    # Check Tesseract
    st.write("**Tesseract OCR:**")
    try:
        import pytesseract
        try:
            from config import get_tesseract_path
        except ImportError:
            from .config import get_tesseract_path
        
        tesseract_path = get_tesseract_path()
        version = pytesseract.get_tesseract_version()
        
        st.write(f"✅ Tesseract found at: `{tesseract_path}`")
        st.write(f"📋 Version: {version}")
        
        # Check languages
        languages = pytesseract.get_languages()
        st.write(f"🌐 Available languages: {', '.join(languages)}")
        
        if 'ukr' in languages:
            st.write("✅ Ukrainian language support available")
        else:
            st.write("❌ Ukrainian language support not found")
            st.warning("Install Ukrainian language data: `apt-get install tesseract-ocr-ukr`")
            
    except Exception as e:
        st.write(f"❌ Tesseract check failed: {e}")
    
    # Installation instructions
    st.subheader("📥 Installation Instructions")
    
    st.write("**1. Install system dependencies:**")
    st.code("""
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-ukr poppler-utils

# macOS
brew install tesseract tesseract-lang poppler

# Windows
# Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
    """, language='bash')
    
    st.write("**2. Install Python dependencies:**")
    st.code("pip install -r requirements.txt", language='bash')
    
    st.write("**3. Usage examples:**")
    st.code("""
# Process single file
python -m src.cli process document.pdf

# Process multiple files
python -m src.cli batch /path/to/pdfs/

# Check system
python -m src.cli check
    """, language='bash')


if __name__ == "__main__":
    launch_gui() 