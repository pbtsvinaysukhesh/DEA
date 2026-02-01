Your On-Device AI Intelligence (DEA)


┌──────────────────────────────────────────────────────────────┐
│     PUBLICATION-GRADE AI RESEARCH INTELLIGENCE SYSTEM        │
│                (For Users)                          │
└──────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼──────────┐                   ┌───────▼──────────┐
│  Data Ingestion  │                   │  Multi-Model AI  │
│                  │                   │                  │
│ • arXiv          │                   │ • Groq (fast)    │
│ • RSS Feeds      │                   │ • Ollama (local) │
│ • Web Search     │                   │ • Gemini (backup)│
└───────┬──────────┘                   └───────┬──────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            │
                ┌───────────▼───────────┐
                │  Knowledge Layer      │
                │                       │
                │ • Knowledge Graph     │
                │ • Vector Store        │
                │ • CoT Reasoner        │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │  Graph RAG Engine     │
                │                       │
                │ 1. Vector Search      │
                │ 2. Graph Traversal    │
                │ 3. Trend Analysis     │
                │ 4. Context Building   │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │  AI Analysis          │
                │  (Multi-stage CoT)    │
                │                       │
                │ • Evidence gathering  │
                │ • Reasoning chain     │
                │ • Citation tracking   │
                │ • Confidence scoring  │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │  Knowledge Update     │
                │                       │
                │ • Add to graph        │
                │ • Update vectors      │
                │ • Detect trends       │
                │ • Identify gaps       │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │  Output Generation    │
                │                       │
                │ • HTML Reports        │
                │ • Email Distribution  │
                │ • API Responses       │
                │ • Trend Dashboards    │
                └───────────────────────┘
#### 3. Collecting Articles

**enhanced but compatible:**
```python
from src.collector import Collector, deduplicate_articles

collector = Collector()

# Still works the same way
articles = collector.fetch_arxiv(queries)
articles += collector.fetch_rss(feeds)

# New: Deduplication utility
articles = deduplicate_articles(articles)

# New: Statistics
stats = collector.get_statistics()
print(f"Collected: {stats['total_fetched']}")
```

#### 4. Formatting Reports

**Old:**
```python
formatter = ReportFormatter()
html = formatter.build_html(insights)
```

**New (same interface, better output):**
```python
formatter = ReportFormatter()

# Same as before
html = formatter.build_html(insights)

# New: Text summary for logs
text = formatter.build_text_summary(insights)
```

## 🚨 Breaking Changes

**None!** The new version is 100% backward compatible with your old code.

However, we **recommend** these upgrades:

### 1. Add Status Checking

```python
Check status
result = processor.process_article(article)
if result.get('status') != 'failed' and result['relevance_score'] >= 60:
    # use result
```

### 2. Use Statistics

```python
# NEW: Track success rates
processor = AIProcessor(api_key=api_key)

# ... process articles ...

stats = processor.get_statistics()
logger.info(f"Success rate: {stats['success_rate']:.1f}%")
```

### 3. Enable Logging

```python
# Add to your code
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### 1. Review Logs

```bash
tail -f logs/pipeline_*.log
```

### 2. Check Statistics

After first run:
```python
from src.history import HistoryManager

history = HistoryManager()
stats = history.get_statistics(days=30)
print(stats)
```

### 3. Adjust Thresholds

If you're getting too many/too few results:

```yaml
# In config/config.yaml
system:
  relevance_threshold: 50  # Lower = more results
  # or
  relevance_threshold: 70  # Higher = fewer, higher quality
```

### 4. Monitor Email Delivery

```bash
# Check if emails are being sent
grep "Email sent" logs/pipeline_*.log
```

## 🔍 Troubleshooting Migration

### Issue: Import errors

**Problem:** `ModuleNotFoundError: No module named 'yaml'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Config not found

**Problem:** `Configuration file not found: config/config.yaml`

**Solution:**
```bash
# Ensure config directory exists
mkdir -p config
# Copy your old config or create new one
cp old-project/config/config.yaml config/
```

### Issue: History file format

**Problem:** Old history.json not working

**Solution:**
```bash
# Backup old history
cp data/history.json data/history_old.json

# The new system will read it correctly
# If there are issues, just start fresh:
rm data/history.json
# System will create new one automatically
```

### Issue: Email not sending

**Problem:** Emails worked before, not now

**Solution:**
```bash
# Test email configuration
python main.py test

# Check logs for specific error
grep -i "email\|smtp" logs/pipeline_*.log
```

## 📚 New Features to Try

### 1. Test Mode

```bash
python main.py test
```

### 2. Trend Detection

The system now detects:
- Popular quantization methods
- Model type trends
- Memory footprint averages
- High DRAM impact patterns

### 3. CSV Export

```python
from src.history import HistoryManager

history = HistoryManager()
history.export_csv("last_month.csv", days=30)
```

### 4. Search History

```python
history = HistoryManager()
results = history.search_history("INT4", days=30)
print(f"Found {len(results)} papers about INT4")
```

### 5. Progress Callbacks

```python
results = processor.process_batch(
    articles,
    progress_callback=lambda curr, total, title: 
        print(f"[{curr}/{total}] {title[:40]}...")
)


## 📞 Need Help?

1. Check logs: `tail -f logs/pipeline_*.log`
2. Run test mode: `python main.py test`
3. Review README.md for detailed docs
4. Check ENHANCEMENT_DOCS.md for technical details



**Recommended:** Keep both versions for a week to ensure smooth transition, then remove the backup once confident.
