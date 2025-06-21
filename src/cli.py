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
    
    def __init__(self, output_dir: Path, verbose: bool = False, enable_language_splitting: bool = True,
                 enable_markdown: bool = True, enable_markitdown_fallback: bool = False):
        """
        Initialize document processor.
        
        Args:
            output_dir: Output directory for results
            verbose: Enable verbose logging
            enable_language_splitting: Enable language splitting
            enable_markdown: Enable markdown output generation
            enable_markitdown_fallback: Enable markitdown fallback for PDF conversion
        """
        self.output_dir = output_dir
        self.verbose = verbose
        self.enable_language_splitting = enable_language_splitting
        self.enable_markdown = enable_markdown
        self.enable_markitdown_fallback = enable_markitdown_fallback
        
        # Initialize processors
        self.pdf_processor = PDFProcessor()
        self.ocr_processor = UkrainianOCR(enable_language_splitting=self.enable_language_splitting)
        self.table_extractor = TableExtractor(enable_language_splitting=self.enable_language_splitting)
        
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
            self.ocr_processor.save_results(ocr_results, file_output_dir, 'ocr_results', 
                                           enable_markdown=self.enable_markdown)
            
            # Step 3: Extract tables
            self.logger.info("Step 3: Extracting tables")
            table_results = self.table_extractor.extract_tables(pdf_path, images)
            
            # Save table results
            if table_results:
                self.table_extractor.save_tables(table_results, file_output_dir, 'tables',
                                                enable_markdown=self.enable_markdown)
            
            # Step 4: Generate comprehensive markdown document (if enabled)
            if self.enable_markdown and (ocr_results or table_results):
                self._generate_comprehensive_markdown(ocr_results, table_results, file_output_dir, pdf_path)
            
            # Step 5: Try markitdown fallback if enabled
            markitdown_result = None
            if self.enable_markitdown_fallback:
                markitdown_result = self._try_markitdown_conversion(pdf_path, file_output_dir)
            
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
        
        # Calculate aggregate statistics for successful files
        successful_results = [r for r in results if r['success']]
        if successful_results:
            total_pages = sum(r['pages_processed'] for r in successful_results)
            total_characters = sum(r['ocr_results']['total_characters'] for r in successful_results)
            total_words = sum(r['ocr_results']['total_words'] for r in successful_results)
            total_tables = sum(r['table_results']['tables_found'] for r in successful_results)
            avg_confidence = sum(r['ocr_results']['average_confidence'] for r in successful_results) / len(successful_results)
        
        self.logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        
        return results
    
    def _generate_comprehensive_markdown(self, ocr_results: List, table_results: List, 
                                       output_dir: Path, pdf_path: Path):
        """
        Generate comprehensive markdown document combining OCR and table results.
        
        Args:
            ocr_results: List of OCR results
            table_results: List of table results
            output_dir: Output directory
            pdf_path: Original PDF path for reference
        """
        try:
            from src.utils.markdown_converter import create_markdown_converter
            
            converter = create_markdown_converter()
            document_title = f"Complete Analysis - {pdf_path.stem}"
            
            # Generate comprehensive document
            markdown_content = converter.create_document_markdown(
                ocr_results, table_results, document_title, include_metadata=True
            )
            
            if markdown_content:
                comprehensive_file = output_dir / "complete_analysis.md"
                with open(comprehensive_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                self.logger.info(f"Generated comprehensive markdown: {comprehensive_file}")
            
            # Generate language-specific comprehensive documents
            if self.enable_language_splitting:
                # Ukrainian version
                if any(hasattr(r, 'ukrainian_text') and r.ukrainian_text for r in ocr_results) or \
                   any(hasattr(t, 'ukrainian_data') and not t.ukrainian_data.empty for t in table_results):
                    uk_markdown = converter.create_language_specific_markdown(
                        ocr_results, table_results, 'uk', document_title
                    )
                    if uk_markdown:
                        uk_file = output_dir / "complete_analysis_uk.md"
                        with open(uk_file, 'w', encoding='utf-8') as f:
                            f.write(uk_markdown)
                        self.logger.info(f"Generated Ukrainian comprehensive markdown: {uk_file}")
                
                # English version
                if any(hasattr(r, 'english_text') and r.english_text for r in ocr_results) or \
                   any(hasattr(t, 'english_data') and not t.english_data.empty for t in table_results):
                    en_markdown = converter.create_language_specific_markdown(
                        ocr_results, table_results, 'en', document_title
                    )
                    if en_markdown:
                        en_file = output_dir / "complete_analysis_en.md"
                        with open(en_file, 'w', encoding='utf-8') as f:
                            f.write(en_markdown)
                        self.logger.info(f"Generated English comprehensive markdown: {en_file}")
                        
        except ImportError:
            self.logger.warning("Markdown converter not available. Install markitdown: pip install markitdown")
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive markdown: {e}")
    
    def _try_markitdown_conversion(self, pdf_path: Path, output_dir: Path) -> Optional[str]:
        """
        Try converting PDF using markitdown as fallback/comparison.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory
            
        Returns:
            Markitdown result content or None
        """
        try:
            from src.utils.markdown_converter import create_markdown_converter
            
            converter = create_markdown_converter()
            markitdown_content = converter.convert_pdf_with_markitdown(pdf_path)
            
            if markitdown_content:
                markitdown_file = output_dir / "markitdown_conversion.md"
                with open(markitdown_file, 'w', encoding='utf-8') as f:
                    f.write(markitdown_content)
                self.logger.info(f"Generated markitdown conversion: {markitdown_file}")
                return markitdown_content
            else:
                self.logger.warning("Markitdown conversion returned no content")
                return None
                
        except ImportError:
            self.logger.warning("Markitdown not available for PDF conversion")
            return None
        except Exception as e:
            self.logger.error(f"Markitdown conversion failed: {e}")
            return None


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
@click.option('--disable-language-splitting', is_flag=True,
              help='Disable automatic language splitting of extracted text')
@click.option('--disable-markdown', is_flag=True,
              help='Disable markdown output generation')
@click.option('--enable-markitdown-fallback', is_flag=True,
              help='Enable markitdown fallback for PDF conversion comparison')
@click.pass_context
def process(ctx, pdf_file, output, pdf_method, table_method, confidence_threshold, 
           disable_language_splitting, disable_markdown, enable_markitdown_fallback):
    """
    Process a single PDF file to extract text and tables.
    Creates separate Ukrainian and English output files unless --disable-language-splitting is used.
    Generates markdown output unless --disable-markdown is used.
    
    PDF_FILE: Path to the PDF file to process
    """
    verbose = ctx.obj.get('verbose', False)
    
    # Update configuration
    from .config import OCR_CONFIG, TABLE_CONFIG
    OCR_CONFIG['confidence_threshold'] = confidence_threshold
    TABLE_CONFIG['detection_method'] = table_method
    
    try:
        processor = DocumentProcessor(
            output, 
            verbose, 
            enable_language_splitting=not disable_language_splitting,
            enable_markdown=not disable_markdown,
            enable_markitdown_fallback=enable_markitdown_fallback
        )
        
        click.echo(f"Processing: {pdf_file}")
        click.echo(f"Output directory: {output}")
        
        if not disable_language_splitting:
            click.echo("Language splitting: Enabled (Ukrainian and English files will be created)")
        else:
            click.echo("Language splitting: Disabled")
            
        if not disable_markdown:
            click.echo("Markdown output: Enabled (.md files will be created)")
        else:
            click.echo("Markdown output: Disabled")
            
        if enable_markitdown_fallback:
            click.echo("Markitdown fallback: Enabled (alternative PDF conversion will be attempted)")
        
        result = processor.process_single_file(pdf_file)
        
        if result['success']:
            click.echo(click.style("✓ Processing completed successfully!", fg='green'))
            click.echo(f"  Pages processed: {result['pages_processed']}")
            click.echo(f"  OCR confidence: {result['ocr_results']['average_confidence']:.1f}%")
            click.echo(f"  Total characters: {result['ocr_results']['total_characters']:,}")
            click.echo(f"  Total words: {result['ocr_results']['total_words']:,}")
            click.echo(f"  Tables found: {result['table_results']['tables_found']}")
            
            # Show language breakdown if enabled
            if not disable_language_splitting and 'language_breakdown' in result.get('ocr_results', {}):
                lang_breakdown = result['ocr_results']['language_breakdown']
                click.echo(f"  Ukrainian characters: {lang_breakdown.get('ukrainian_characters', 0):,}")
                click.echo(f"  English characters: {lang_breakdown.get('english_characters', 0):,}")
            
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
@click.option('--pdf-method', type=click.Choice(['pdf2image', 'pymupdf']), default='pdf2image',
              help='PDF to image conversion method')
@click.option('--table-method', type=click.Choice(['camelot', 'tabula']), default='camelot',
              help='Table extraction method')
@click.option('--confidence-threshold', type=int, default=60,
              help='Minimum OCR confidence threshold (0-100)')
@click.option('--disable-language-splitting', is_flag=True,
              help='Disable automatic language splitting of extracted text')
@click.option('--disable-markdown', is_flag=True,
              help='Disable markdown output generation')
@click.option('--enable-markitdown-fallback', is_flag=True,
              help='Enable markitdown fallback for PDF conversion comparison')
@click.pass_context
def batch(ctx, input_dir, output, pdf_method, table_method, confidence_threshold, 
         disable_language_splitting, disable_markdown, enable_markitdown_fallback):
    """
    Process multiple PDF files in a directory.
    Creates separate Ukrainian and English output files unless --disable-language-splitting is used.
    Generates markdown output unless --disable-markdown is used.
    
    INPUT_DIR: Directory containing PDF files to process
    """
    verbose = ctx.obj.get('verbose', False)
    
    # Update configuration
    from .config import OCR_CONFIG, TABLE_CONFIG
    OCR_CONFIG['confidence_threshold'] = confidence_threshold
    TABLE_CONFIG['detection_method'] = table_method
    
    try:
        processor = DocumentProcessor(
            output, 
            verbose, 
            enable_language_splitting=not disable_language_splitting,
            enable_markdown=not disable_markdown,
            enable_markitdown_fallback=enable_markitdown_fallback
        )
        
        click.echo(f"Batch processing directory: {input_dir}")
        click.echo(f"Output directory: {output}")
        
        if not disable_language_splitting:
            click.echo("Language splitting: Enabled (Ukrainian and English files will be created)")
        else:
            click.echo("Language splitting: Disabled")
            
        if not disable_markdown:
            click.echo("Markdown output: Enabled (.md files will be created)")
        else:
            click.echo("Markdown output: Disabled")
            
        if enable_markitdown_fallback:
            click.echo("Markitdown fallback: Enabled (alternative PDF conversion will be attempted)")
        
        results = processor.process_batch(input_dir)
        
        # Print summary
        successful = len([r for r in results if r['success']])
        failed = len(results) - successful
        
        # Calculate aggregate statistics for successful files
        successful_results = [r for r in results if r['success']]
        if successful_results:
            total_pages = sum(r['pages_processed'] for r in successful_results)
            total_characters = sum(r['ocr_results']['total_characters'] for r in successful_results)
            total_words = sum(r['ocr_results']['total_words'] for r in successful_results)
            total_tables = sum(r['table_results']['tables_found'] for r in successful_results)
            avg_confidence = sum(r['ocr_results']['average_confidence'] for r in successful_results) / len(successful_results)
        
        click.echo(f"\nBatch processing completed:")
        click.echo(f"  {click.style(str(successful), fg='green')} successful")
        click.echo(f"  {click.style(str(failed), fg='red')} failed")
        
        if successful_results:
            click.echo(f"\nAggregate statistics:")
            click.echo(f"  Total pages processed: {total_pages}")
            click.echo(f"  Average OCR confidence: {avg_confidence:.1f}%")
            click.echo(f"  Total characters extracted: {total_characters:,}")
            click.echo(f"  Total words extracted: {total_words:,}")
            click.echo(f"  Total tables found: {total_tables}")
        
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