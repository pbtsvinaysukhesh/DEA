"""
Main Pipeline for On-Device AI Memory Intelligence Agent
UPDATED: Windows-Safe Logging (ASCII only), Multi-Model, Graph RAG
"""

import os
import sys
import yaml
import logging
from datetime import datetime
from dotenv import load_dotenv

# Core components
from src.collector import Collector, deduplicate_articles
from src.formatter import ReportFormatter
from src.mailer import Mailer, send_error_notification
from src.history import HistoryManager

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Configure logging for the application"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Force UTF-8 for file, but allow system default for stream (with ASCII content)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Attempt to force UTF-8 on stdout for Windows, but don't crash if it fails
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    
    return logging.getLogger(__name__)


logger = setup_logging()


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def validate_environment():
    """Validate required environment variables"""
    # Check for at least one AI provider
    has_gemini = bool(os.getenv("GOOGLE_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_ollama = False
    
    # Check if Ollama is available
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        has_ollama = response.status_code == 200
    except:
        pass
    
    if not (has_gemini or has_groq or has_ollama):
        logger.error("No AI provider available!")
        logger.error("Please set at least one of: GOOGLE_API_KEY, GROQ_API_KEY, or run Ollama")
        sys.exit(1)
    
    logger.info("Environment validation passed")
    if has_groq:
        logger.info("[OK] Groq API available")
    if has_ollama:
        logger.info("[OK] Ollama available")
    if has_gemini:
        logger.info("[OK] Gemini API available")


def initialize_processor(config: dict):
    """
    Initialize AI processor with multi-model support
    """
    # Check if multi-model is enabled in config
    use_multi_model = config.get('system', {}).get('multi_model', {}).get('enabled', False)
    use_graph_rag = config.get('system', {}).get('graph_rag', {}).get('enabled', False)
    
    if use_multi_model:
        logger.info("=" * 80)
        logger.info("MULTI-MODEL MODE ENABLED")
        logger.info("=" * 80)
        
        try:
            # Import multi-model components
            from src.multimodal_orchestrator import EnterpriseAIProcessor
            
            # Initialize knowledge manager if Graph RAG enabled
            knowledge_manager = None
            if use_graph_rag:
                logger.info("Graph RAG enabled - initializing knowledge graph")
                try:
                    from src.knowledge_graph import EnterpriseKnowledgeManager
                    knowledge_manager = EnterpriseKnowledgeManager(
                        data_dir="data/knowledge",
                        use_embeddings=True
                    )
                    logger.info("[OK] Knowledge graph initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize Graph RAG: {e}")
                    logger.warning("Continuing without Graph RAG")
            
            # Initialize multi-model processor
            processor = EnterpriseAIProcessor(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                gemini_api_key=os.getenv("GOOGLE_API_KEY"),
                ollama_url=config.get('system', {}).get('multi_model', {}).get('ollama_url', 'http://localhost:11434'),
                knowledge_manager=knowledge_manager
            )
            
            logger.info("[OK] Multi-model processor initialized")
            return processor, "multi-model"
            
        except ImportError as e:
            logger.warning(f"Multi-model components not available: {e}")
            logger.warning("Falling back to basic mode")
    
    # FALLBACK: Use basic single-model processor
    logger.info("Using basic single-model processor")
    from src.analyzer import AIProcessor

    # Try Groq first (primary), then fallback to other options
    api_key = os.getenv("GROQ_API_KEY")
    provider = "groq"

    if not api_key:
        # Fallback to Google if Groq missing
        api_key = os.getenv("GOOGLE_API_KEY")
        provider = "google"
        if not api_key:
            logger.error("No API Key found for basic mode (need GROQ_API_KEY or GOOGLE_API_KEY)")
            sys.exit(1)

    # Set appropriate default model based on provider
    if provider == "groq":
        model_name = config.get('system', {}).get('model_name', 'llama-3.1-8b-instant')
    else:
        model_name = config.get('system', {}).get('model_name', 'gemini-2.0-flash')

    processor = AIProcessor(api_key=api_key, model_name=model_name)
    
    return processor, "basic"


def run_pipeline():
    """Main pipeline execution"""
    logger.info("=" * 80)
    logger.info("On-Device AI Memory Intelligence Agent - Pipeline Starting")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Validate environment
        validate_environment()
        
        # Load configuration
        config = load_config()
        logger.info(f"Configuration: {config.get('system', {}).get('relevance_threshold', 60)} relevance threshold")
        
        # Initialize components
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: Initialization")
        logger.info("=" * 80)
        
        # Initialize AI processor
        processor, processor_type = initialize_processor(config)
        logger.info(f"Processor type: {processor_type}")
        
        # Initialize other components
        collector = Collector()
        formatter = ReportFormatter()
        history = HistoryManager()
        
        # Initialize mailer
        try:
            mailer = Mailer(config.get('email', {}))
        except Exception as e:
            logger.warning(f"Failed to initialize mailer: {e}")
            mailer = None
        
        # Collect articles
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: Article Collection")
        logger.info("=" * 80)
        
        raw_data = collector.fetch_all(config)
        logger.info(f"Collected {len(raw_data)} raw articles")
        
        if not raw_data:
            logger.warning("No articles collected. Exiting.")
            return
        
        # Deduplicate
        raw_data = deduplicate_articles(raw_data)
        logger.info(f"After deduplication: {len(raw_data)} unique articles")
        
        # Load historical context
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: Context Loading")
        logger.info("=" * 80)
        
        context_days = config.get('system', {}).get('context_days', 7)
        context_str = history.load_recent_context(days=context_days)
        
        if context_str:
            logger.info(f"Loaded {context_days}-day historical context")
        else:
            logger.info("No historical context available")
        
        # Analyze articles
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: AI Analysis")
        logger.info("=" * 80)
        
        relevance_threshold = config['system']['relevance_threshold']
        logger.info(f"Analyzing articles with {relevance_threshold}+ relevance threshold...")
        
        final_insights = []
        failed_analyses = []
        
        for idx, item in enumerate(raw_data, 1):
            logger.info(f"[{idx}/{len(raw_data)}] Analyzing: {item['title'][:60]}...")
            
            try:
                # Process with context
                analysis = processor.process_article(item, context_str=context_str)
                
                # Check if analysis was successful
                if analysis.get('status') == 'failed':
                    logger.warning(f"  [!] Analysis failed: {analysis.get('error_reason', 'Unknown')}")
                    failed_analyses.append(item)
                    continue
                
                # Check relevance threshold
                score = analysis.get('relevance_score', 0)
                if score >= relevance_threshold:
                    # Merge original data with AI analysis
                    merged = {**item, **analysis}
                    final_insights.append(merged)
                    
                    # Log provider used (for multi-model)
                    provider = analysis.get('provider_used', 'unknown')
                    logger.info(f"  [+] Score: {score} | Platform: {analysis.get('platform')} | Provider: {provider}")
                else:
                    logger.info(f"  [-] Score: {score} (below threshold)")
                    
            except Exception as e:
                logger.error(f"  [!] Exception during analysis: {e}")
                failed_analyses.append(item)
                continue
        
        # Log analysis statistics
        logger.info("\n" + "-" * 80)
        
        # Handle stats based on processor type
        if hasattr(processor, 'get_statistics'):
            stats = processor.get_statistics()
            if processor_type == "multi-model":
                logger.info(f"Analysis Statistics:")
                logger.info(f"  Total Processed: {stats['processor_stats']['total_processed']}")
                logger.info(f"  Successful: {stats['processor_stats']['successful']}")
                logger.info(f"  Above Threshold: {len(final_insights)}")
            else:
                logger.info(f"Analysis Statistics:")
                logger.info(f"  Total Processed: {stats.get('total_processed', 0)}")
                logger.info(f"  Successful: {stats.get('successful', 0)}")
        
        logger.info("-" * 80)
        
        # Save insights to history
        if final_insights:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 5: Saving to History")
            logger.info("=" * 80)
            
            history.save_insights(final_insights)
            logger.info(f"Saved {len(final_insights)} insights to history")
        
        # Generate and send report
        if final_insights:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 6: Report Generation & Distribution")
            logger.info("=" * 80)
            
            logger.info(f"Generating report for {len(final_insights)} relevant items...")
            html_report = formatter.build_html(final_insights)
            
            # Also generate text summary for logs
            text_summary = formatter.build_text_summary(final_insights)
            logger.info("\n" + text_summary)
            
            # Send email
            if mailer:
                logger.info("\nSending email report...")
                success = mailer.send(html_report)
                
                if success:
                    logger.info("[OK] Email sent successfully")
                else:
                    logger.error("[X] Failed to send email")
            else:
                logger.warning("Mailer not configured - skipping email dispatch")
                
                # Save report to file as fallback
                report_dir = "reports"
                os.makedirs(report_dir, exist_ok=True)
                report_file = os.path.join(
                    report_dir, 
                    f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                )
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(html_report)
                logger.info(f"Report saved to: {report_file}")
        
        else:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 6: No Relevant Research Found")
            logger.info("=" * 80)
            
            logger.info("No articles met the relevance threshold today.")
        
        # Calculate execution time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Complete")
        logger.info("=" * 80)
        logger.info(f"Execution time: {duration:.1f} seconds")
        logger.info(f"Relevant insights: {len(final_insights)}")
        logger.info(f"Processor mode: {processor_type}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("CRITICAL ERROR - Pipeline Failed")
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


def test_configuration():
    """Test configuration and connectivity"""
    logger.info("Testing configuration...")
    
    try:
        # Test config loading
        config = load_config()
        logger.info("[OK] Configuration file loaded")
        
        # Check multi-model settings
        multi_model_enabled = config.get('system', {}).get('multi_model', {}).get('enabled', False)
        
        if multi_model_enabled:
            logger.info("\nMulti-model mode detected - testing all providers:")
            
            # Test Groq
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                logger.info("[OK] GROQ_API_KEY found")
            else:
                logger.warning("[X] GROQ_API_KEY not found")
            
            # Test Ollama
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    logger.info("[OK] Ollama running")
                else:
                    logger.warning("[!] Ollama responding with error")
            except:
                logger.warning("[!] Ollama not reachable")
            
            # Test Gemini
            gemini_key = os.getenv("GOOGLE_API_KEY")
            if gemini_key:
                logger.info("[OK] GOOGLE_API_KEY found")
            else:
                logger.warning("[X] GOOGLE_API_KEY not found")
        
        else:
            # Test basic mode
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                logger.info("[OK] GOOGLE_API_KEY found")
            else:
                logger.error("[X] GOOGLE_API_KEY not found")
                return False
        
        # Test collector
        logger.info("\nTesting article collection...")
        collector = Collector()
        test_articles = collector.fetch_arxiv(["machine learning edge"])
        logger.info(f"[OK] Collected {len(test_articles)} test articles")
        
        logger.info("\n" + "=" * 80)
        logger.info("Configuration test complete")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "test":
            test_configuration()
        elif command == "help":
            print("Usage: python main.py [test|help]")
        else:
            print(f"Unknown command: {command}")
    else:
        run_pipeline()