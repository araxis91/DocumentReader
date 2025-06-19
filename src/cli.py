"""
Command-line interface for DocumentReader application.
"""

import click
import sys
from pathlib import Path
from typing import Optional, List
import time
from tqdm import tqdm

from .config import create_directories, APP_NAME, VERSION
from .ingest import PDFProcessor
from .ocr import UkrainianOCR
from .tables import TableExtractor
from .utils.logger import setup_logger, main_logger


class DocumentProcessor:
    """Main document processing orchestrator."""
    
    def __init__(self, output_dir: Path, verbose: bool = False):
        """
        Initialize document processor.
        
        Args:
            output_dir: Output directory for results
            verbose: Enable verbose logging
        """
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Initialize processors
        self.pdf_processor = PDFProcessor()
        self.ocr_processor = UkrainianOCR()
        self.table_extractor = TableExtractor()
        
        # Setup logging
        log_level = 'DEBUG' if verbose else 'INFO'
        self.logger = setup_logger('DocumentProcessor', 'document_processor.log', log_level)
    
    def process_single_file(self, pdf_path: Path) -> dict:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Processing results summary
        """
        start_time = time.time()
        
        self.logger.info(f"Processing file: {pdf_path}")
        
        try:
            # Create output directory for this file
            file_output_dir = self.output_dir / pdf_path.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Convert PDF to images
            self.logger.info("Step 1: Converting PDF to images")
            images, pdf_info = self.pdf_processor.process_pdf(pdf_path)
            
            if not images:
                self.logger.error("Failed to convert PDF to images")
                return {
                    'success': False,
                    'error': 'PDF conversion failed',
                    'processing_time': time.time() - start_time
                }
            
            # Step 2: Extract text using OCR
            self.logger.info("Step 2: Extracting text using OCR")
            ocr_results = self.ocr_processor.extract_text_batch(images)
            
            # Save OCR results
            self.ocr_processor.save_results(ocr_results, file_output_dir, 'ocr_results')
            
            # Step 3: Extract tables
            self.logger.info("Step 3: Extracting tables")
            table_results = self.table_extractor.extract_tables(pdf_path, images)
            
            # Save table results
            if table_results:
                self.table_extractor.save_tables(table_results, file_output_dir, 'tables')
            
            # Generate summary
            processing_time = time.time() - start_time
            summary = {
                'success': True,
                'file_path': str(pdf_path),
                'pdf_info': pdf_info,
                'pages_processed': len(images),
                'ocr_results': {
                    'successful_pages': len([r for r in ocr_results if r.confidence > 0]),
                    'average_confidence': sum([r.confidence for r in ocr_results]) / len(ocr_results) if ocr_results else 0,
                    'total_characters': sum([len(r.text) for r in ocr_results]),
                    'total_words': sum([len(r.text.split()) for r in ocr_results])
                },
                'table_results': {
                    'tables_found': len(table_results),
                    'valid_tables': len([t for t in table_results if t.is_valid()])
                },
                'processing_time': processing_time,
                'output_directory': str(file_output_dir)
            }
            
            self.logger.info(f"Processing completed in {processing_time:.2f} seconds")
            return summary
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def process_batch(self, pdf_files: List[Path]) -> List[dict]:
        """
        Process multiple PDF files.
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            List of processing results
        """
        results = []
        
        self.logger.info(f"Starting batch processing of {len(pdf_files)} files")
        
        with tqdm(pdf_files, desc="Processing files") as pbar:
            for pdf_path in pbar:
                pbar.set_description(f"Processing {pdf_path.name}")
                result = self.process_single_file(pdf_path)
                results.append(result)
                
                # Update progress bar with status
                status = "✓" if result['success'] else "✗"
                pbar.set_postfix(status=status)
        
        # Generate batch summary
        successful = len([r for r in results if r['success']])
        failed = len(results) - successful
        
        self.logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        
        return results


@click.group()
@click.version_option(version=VERSION, prog_name=APP_NAME)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """
    DocumentReader - Extract text and tables from scanned Ukrainian PDF documents.
    
    This tool processes PDF documents to extract text using OCR and detect tables,
    with special support for Ukrainian language documents.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    # Create necessary directories
    create_directories()


@cli.command()
@click.argument('pdf_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), default='./output',
              help='Output directory for results')
@click.option('--pdf-method', type=click.Choice(['pdf2image', 'pymupdf']), default='pdf2image',
              help='PDF to image conversion method')
@click.option('--table-method', type=click.Choice(['camelot', 'tabula']), default='camelot',
              help='Table extraction method')
@click.option('--confidence-threshold', type=int, default=60,
              help='Minimum OCR confidence threshold (0-100)')
@click.pass_context
def process(ctx, pdf_file, output, pdf_method, table_method, confidence_threshold):
    """
    Process a single PDF file to extract text and tables.
    
    PDF_FILE: Path to the PDF file to process
    """
    verbose = ctx.obj.get('verbose', False)
    
    # Update configuration
    from .config import OCR_CONFIG, TABLE_CONFIG
    OCR_CONFIG['confidence_threshold'] = confidence_threshold
    TABLE_CONFIG['detection_method'] = table_method
    
    try:
        processor = DocumentProcessor(output, verbose)
        
        click.echo(f"Processing: {pdf_file}")
        click.echo(f"Output directory: {output}")
        
        result = processor.process_single_file(pdf_file)
        
        if result['success']:
            click.echo(click.style("✓ Processing completed successfully!", fg='green'))
            click.echo(f"  Pages processed: {result['pages_processed']}")
            click.echo(f"  OCR confidence: {result['ocr_results']['average_confidence']:.1f}%")
            click.echo(f"  Tables found: {result['table_results']['tables_found']}")
            click.echo(f"  Processing time: {result['processing_time']:.2f}s")
            click.echo(f"  Results saved to: {result['output_directory']}")
        else:
            click.echo(click.style("✗ Processing failed!", fg='red'))
            click.echo(f"  Error: {result['error']}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg='red'))
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), default='./output',
              help='Output directory for results')
@click.option('--pattern', '-p', default='*.pdf',
              help='File pattern to match (default: *.pdf)')
@click.option('--recursive', '-r', is_flag=True,
              help='Search recursively in subdirectories')
@click.option('--table-method', type=click.Choice(['camelot', 'tabula']), default='camelot',
              help='Table extraction method')
@click.option('--confidence-threshold', type=int, default=60,
              help='Minimum OCR confidence threshold (0-100)')
@click.pass_context
def batch(ctx, input_dir, output, pattern, recursive, table_method, confidence_threshold):
    """
    Process multiple PDF files in a directory.
    
    INPUT_DIR: Directory containing PDF files to process
    """
    verbose = ctx.obj.get('verbose', False)
    
    # Update configuration
    from .config import OCR_CONFIG, TABLE_CONFIG
    OCR_CONFIG['confidence_threshold'] = confidence_threshold
    TABLE_CONFIG['detection_method'] = table_method
    
    try:
        # Find PDF files
        if recursive:
            pdf_files = list(input_dir.rglob(pattern))
        else:
            pdf_files = list(input_dir.glob(pattern))
        
        if not pdf_files:
            click.echo(click.style(f"No PDF files found matching pattern '{pattern}'", fg='yellow'))
            sys.exit(0)
        
        click.echo(f"Found {len(pdf_files)} PDF files")
        click.echo(f"Output directory: {output}")
        
        processor = DocumentProcessor(output, verbose)
        results = processor.process_batch(pdf_files)
        
        # Print summary
        successful = len([r for r in results if r['success']])
        failed = len(results) - successful
        
        click.echo(f"\nBatch processing completed:")
        click.echo(f"  {click.style(str(successful), fg='green')} successful")
        click.echo(f"  {click.style(str(failed), fg='red')} failed")
        
        if failed > 0:
            click.echo(f"\nFailed files:")
            for result in results:
                if not result['success']:
                    click.echo(f"  ✗ {result.get('file_path', 'unknown')}: {result.get('error', 'unknown error')}")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg='red'))
        sys.exit(1)


@cli.command()
@click.option('--check-tesseract', is_flag=True, help='Check Tesseract installation')
@click.option('--check-languages', is_flag=True, help='Check available languages')
@click.option('--check-dependencies', is_flag=True, help='Check Python dependencies')
def check(check_tesseract, check_languages, check_dependencies):
    """
    Check system requirements and dependencies.
    """
    if not any([check_tesseract, check_languages, check_dependencies]):
        # Run all checks if none specified
        check_tesseract = check_languages = check_dependencies = True
    
    if check_dependencies:
        click.echo("Checking Python dependencies...")
        try:
            import pdf2image
            import pytesseract
            import camelot
            import tabula
            import cv2
            import numpy
            import pandas
            
            click.echo(click.style("✓ All Python dependencies are installed", fg='green'))
        except ImportError as e:
            click.echo(click.style(f"✗ Missing dependency: {e}", fg='red'))
    
    if check_tesseract:
        click.echo("Checking Tesseract installation...")
        try:
            from .config import get_tesseract_path
            import pytesseract
            
            tesseract_path = get_tesseract_path()
            version = pytesseract.get_tesseract_version()
            
            click.echo(click.style(f"✓ Tesseract found at: {tesseract_path}", fg='green'))
            click.echo(f"  Version: {version}")
            
        except Exception as e:
            click.echo(click.style(f"✗ Tesseract check failed: {e}", fg='red'))
    
    if check_languages:
        click.echo("Checking available languages...")
        try:
            import pytesseract
            languages = pytesseract.get_languages()
            
            click.echo(f"Available languages: {', '.join(languages)}")
            
            if 'ukr' in languages:
                click.echo(click.style("✓ Ukrainian language support available", fg='green'))
            else:
                click.echo(click.style("✗ Ukrainian language support not found", fg='red'))
                click.echo("  Install with: apt-get install tesseract-ocr-ukr")
                
        except Exception as e:
            click.echo(click.style(f"✗ Language check failed: {e}", fg='red'))


@cli.command()
@click.argument('pdf_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), default='./output',
              help='Output directory for results')
def gui(pdf_file, output):
    """
    Launch GUI interface for processing a PDF file.
    
    PDF_FILE: Path to the PDF file to process
    """
    try:
        from .gui import launch_gui
        launch_gui(pdf_file, output)
    except ImportError:
        click.echo(click.style("GUI dependencies not installed. Install with: pip install streamlit", fg='red'))
        sys.exit(1)


if __name__ == '__main__':
    cli() 