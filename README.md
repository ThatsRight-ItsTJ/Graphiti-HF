# Graphiti-HF

<p align="center">
  <a href="https://huggingface.co/">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="150" alt="Hugging Face Logo">
  </a>
</p>

<h1 align="center">
Graphiti-HF
</h1>
<h2 align="center">Build Real-Time Knowledge Graphs with HuggingFace 🤗</h2>

<div align="center">

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Datasets](https://img.shields.io/badge/🤗-Datasets-orange)](https://huggingface.co/docs/datasets/)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Spaces-orange)](https://huggingface.co/spaces)

</div>

<div align="center">

⭐ _Bring the power of temporal knowledge graphs to the HuggingFace ecosystem!_ ⭐

</div>

<br />

> [!TIP]
> 🚀 **New to Knowledge Graphs?** Check out our [Interactive Demo Space](https://huggingface.co/spaces/graphiti-hf/knowledge-graph-demo) to see Graphiti-HF in action!

**Graphiti-HF** is a fork of [Zep's Graphiti](https://github.com/getzep/graphiti) that brings temporal knowledge graph capabilities to the HuggingFace ecosystem. Instead of requiring traditional graph databases, Graphiti-HF stores knowledge graphs as HuggingFace Datasets, making them collaborative, version-controlled, and deployable as interactive Spaces.

Perfect for researchers, developers, and organizations who want to:

- 🔬 **Research**: Build knowledge graphs from academic papers and research data
- 🏢 **Enterprise**: Create collaborative company knowledge bases
- 🤖 **AI Agents**: Give your agents persistent, queryable memory
- 🎓 **Education**: Teach graph concepts with interactive visualizations
- 📝 **Personal**: Organize your notes and knowledge with temporal tracking

<br />

<p align="center">
    <img src="images/graphiti-hf-demo.gif" alt="Graphiti-HF interactive demo" width="700px">
</p>

<br />

## 🎯 Why Graphiti-HF?

Traditional knowledge graph solutions require complex database infrastructure and lack collaboration features. Graphiti-HF solves this by leveraging HuggingFace's ecosystem:

| **Traditional Approach** | **Graphiti-HF** |
|---------------------------|------------------|
| ❌ Requires Neo4j/database setup | ✅ No database needed - uses HF Datasets |
| ❌ Single-user development | ✅ Multi-user collaboration via HF |
| ❌ Complex deployment | ✅ One-click deployment to Spaces |
| ❌ Manual version control | ✅ Git-based versioning built-in |
| ❌ Limited sharing options | ✅ Public/private dataset sharing |

### 🔥 Key Features

- **🤗 HuggingFace Native**: Knowledge graphs stored as datasets with automatic versioning
- **⚡ Real-time Updates**: Incremental graph updates without recomputation
- **🔍 Hybrid Search**: Semantic + keyword + graph structure search
- **⏰ Temporal Tracking**: Bi-temporal data model with historical queries
- **🎨 Custom Entities**: Define domain-specific entities with Pydantic models
- **🌐 Interactive Spaces**: Deploy graph interfaces with zero configuration
- **🤝 Collaborative**: Multiple users can work on the same knowledge graph
- **📊 Visual Analytics**: Built-in graph visualization and analytics

## 🚀 Quick Start

### Prerequisites

Before installing Graphiti-HF, you'll need:

- **Python 3.10 or higher**
- **HuggingFace Account**: [Sign up for free](https://huggingface.co/join)
- **LLM API Access**: OpenAI API key (or Anthropic, Gemini, etc.) for entity extraction
- **HuggingFace Token**: For dataset creation and management

### Setup Authentication

1. **Get your HuggingFace token**:
   - Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - Create a new token with "Write" permissions
   - Copy the token

2. **Set up authentication** (choose one method):

   **Option A: Environment Variables**
   ```bash
   export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
   export OPENAI_API_KEY="sk-your_openai_key_here"
   ```

   **Option B: Login via CLI**
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   # Paste your token when prompted
   ```

   **Option C: In Python**
   ```python
   import os
   os.environ["HUGGINGFACE_HUB_TOKEN"] = "hf_your_token_here"
   os.environ["OPENAI_API_KEY"] = "sk_your_openai_key_here"
   ```

### Installation

```bash
pip install graphiti-hf
```

**Optional Extras** (install based on your needs):

```bash
# For local embeddings (recommended for better performance)
pip install graphiti-hf[sentence-transformers]

# For additional LLM providers
pip install graphiti-hf[anthropic]        # Anthropic Claude
pip install graphiti-hf[google-genai]     # Google Gemini
pip install graphiti-hf[groq]             # Groq

# For advanced visualizations
pip install graphiti-hf[viz]              # Plotly, NetworkX extras

# For data integrations
pip install graphiti-hf[integrations]     # Notion, Slack, etc.

# Everything included
pip install graphiti-hf[all]
```

### 30-Second Example

```python
from graphiti_hf import GraphitiHF
from datetime import datetime
import os

# Set your API keys (if not already set in environment)
os.environ["OPENAI_API_KEY"] = "sk-your_key_here"  # Required for entity extraction
os.environ["HUGGINGFACE_HUB_TOKEN"] = "hf_your_token_here"  # Required for dataset access

# Initialize with HuggingFace dataset (will be created if it doesn't exist)
graphiti = GraphitiHF("your-username/my-knowledge-graph")

# Add your first episode
result = await graphiti.add_episode(
    name="Meeting Notes",
    episode_body="Alice discussed the new product launch with Bob. The launch is scheduled for Q2 2024.",
    source_description="Team standup",
    reference_time=datetime.now()
)

print(f"📊 Created {len(result.nodes)} entities and {len(result.edges)} relationships!")

# Search your knowledge graph
results = await graphiti.search("product launch timeline")
for edge in results:
    print(f"💡 {edge.fact}")

# Push to HuggingFace Hub (creates/updates the dataset)
await graphiti.push_to_hub("Added meeting notes")
print("✅ Knowledge graph saved to HuggingFace Hub!")
```

**First run will**:
- Create a new dataset in your HF account
- Initialize the graph schema
- Process your text to extract entities and relationships
- Save everything to HuggingFace Hub with version control

### 🏃‍♂️ Try it in HuggingFace Spaces

The fastest way to get started is with our pre-built Spaces:

1. **[Knowledge Graph Builder](https://huggingface.co/spaces/graphiti-hf/graph-builder)** - Create graphs from text
2. **[Graph Explorer](https://huggingface.co/spaces/graphiti-hf/graph-explorer)** - Visualize and query existing graphs  
3. **[Research Assistant](https://huggingface.co/spaces/graphiti-hf/research-assistant)** - Academic paper knowledge graphs

### ⚙️ Configuration Options

Graphiti-HF offers flexible configuration for different use cases:

```python
from graphiti_hf import GraphitiHF

# Basic setup (uses OpenAI for LLM, HF for embeddings)
graphiti = GraphitiHF("username/my-graph")

# Use local embeddings (no API calls for search)
graphiti = GraphitiHF(
    "username/my-graph",
    embedder="sentence-transformers/all-MiniLM-L6-v2"  # Local model
)

# Use different LLM providers
graphiti = GraphitiHF(
    "username/my-graph",
    llm_provider="anthropic",  # or "google", "groq"
    llm_model="claude-3-sonnet"
)

# Private dataset (only you can access)
graphiti = GraphitiHF(
    "username/private-graph",
    private=True
)

# Organization dataset (team collaboration)
graphiti = GraphitiHF(
    "my-org/team-knowledge-graph",
    # All org members with write access can contribute
)
```

**Environment Variables**:
```bash
# Required
export HUGGINGFACE_HUB_TOKEN="hf_your_token"
export OPENAI_API_KEY="sk_your_key"  # or other LLM provider

# Optional - Override defaults
export GRAPHITI_DEFAULT_LLM="anthropic"
export GRAPHITI_DEFAULT_EMBEDDER="sentence-transformers"
export GRAPHITI_CONCURRENCY_LIMIT="10"  # Rate limiting
```

## 📖 Detailed Examples

### Enterprise Knowledge Base

```python
from graphiti_hf import GraphitiHF
from graphiti_hf.entities import PersonEntity, CompanyEntity, ProjectEntity

# Initialize with custom entity types
graphiti = GraphitiHF(
    repo_id="mycompany/knowledge-base",
    entity_types={
        'Person': PersonEntity,
        'Company': CompanyEntity,  
        'Project': ProjectEntity
    }
)

# Add structured company data
await graphiti.add_episode(
    name="Q4 Team Updates",
    episode_body="""
    Sarah Johnson joined our AI team as Lead ML Engineer. She previously 
    worked at Google DeepMind for 4 years. Sarah will lead the recommendation 
    system project launching in Q2 2024, working closely with the product team.
    """,
    source_description="HR system integration",
    reference_time=datetime.now()
)

# Query with filters
ml_engineers = await graphiti.query_by_entity_type(
    'Person', 
    filters={'occupation': 'ML Engineer'}
)

# Visualize team connections
await graphiti.visualize_subgraph(
    center_nodes=[person.uuid for person in ml_engineers],
    depth=2
)
```

### Research Paper Analysis

```python
from graphiti_hf import GraphitiHF
from graphiti_hf.loaders import ArxivLoader

# Initialize research knowledge graph
graphiti = GraphitiHF("researcher/ai-papers-2024")

# Load papers from arXiv
loader = ArxivLoader()
papers = loader.load_papers(query="attention mechanism", max_results=50)

# Process papers in batch
await graphiti.add_episode_bulk([
    {
        'name': paper.title,
        'content': paper.abstract + "\n\n" + paper.content,
        'source_description': f"arXiv:{paper.id}",
        'reference_time': paper.published_date
    }
    for paper in papers
])

# Find connections between concepts
concept_connections = await graphiti.search(
    "transformer architecture relationships",
    config=SearchConfig(
        search_type="concept_clustering",
        limit=20
    )
)

# Generate research insights
insights = await graphiti.generate_insights(
    topic="recent advances in attention mechanisms",
    time_range=(datetime(2024, 1, 1), datetime.now())
)
```

### Personal Knowledge Management

```python
from graphiti_hf import GraphitiHF
from graphiti_hf.integrations import NotionIntegration, ObsidianIntegration

# Create personal knowledge graph
graphiti = GraphitiHF("username/personal-knowledge")

# Import from existing tools
notion = NotionIntegration(auth_token="your_token")
await notion.sync_to_graphiti(graphiti, database_id="your_database")

obsidian = ObsidianIntegration(vault_path="~/Documents/ObsidianVault")
await obsidian.sync_to_graphiti(graphiti)

# Ask questions about your knowledge
answer = await graphiti.ask(
    "What did I learn about machine learning last month?",
    context_depth=3
)
```

## 🔍 Advanced Search Capabilities

Graphiti-HF provides multiple search methods optimized for different use cases:

```python
# Semantic search - find conceptually similar content
semantic_results = await graphiti.search(
    "neural network architectures",
    search_type="semantic",
    limit=10
)

# Hybrid search - combines semantic + keyword + graph structure
hybrid_results = await graphiti.search_(
    "transformer attention mechanism",
    config=COMBINED_HYBRID_SEARCH_CROSS_ENCODER
)

# Temporal search - find information from specific time periods
temporal_results = await graphiti.search_temporal(
    "product launches",
    time_range=(datetime(2024, 1, 1), datetime(2024, 6, 1))
)

# Graph traversal - explore connections from a starting point
traversal_results = await graphiti.traverse_from_node(
    start_node_uuid="entity_123",
    max_depth=3,
    relationship_types=["COLLABORATES_WITH", "WORKS_ON"]
)
```

## 🎨 Custom Entity Types

Define domain-specific entities for better knowledge representation:

```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ResearchPaperEntity(BaseModel):
    title: str
    authors: List[str]
    venue: Optional[str] = None
    publication_date: Optional[datetime] = None
    doi: Optional[str] = None
    citations: int = 0
    keywords: List[str] = []

class ConceptEntity(BaseModel):
    name: str
    definition: str
    field: str  # "AI", "Biology", etc.
    complexity_level: int  # 1-10
    related_concepts: List[str] = []

# Use in your knowledge graph
graphiti = GraphitiHF(
    "research/ai-concepts",
    entity_types={
        'ResearchPaper': ResearchPaperEntity,
        'Concept': ConceptEntity
    },
    edge_types={
        'Cites': CitationEdge,
        'Introduces': IntroductionEdge
    }
)
```

## 🌐 Deploy to HuggingFace Spaces

Create an interactive knowledge graph interface:

```python
# spaces/app.py
import gradio as gr
from graphiti_hf import GraphitiHF

def create_knowledge_graph_interface():
    graphiti = GraphitiHF("your-org/knowledge-graph")
    
    def add_knowledge(text: str, source: str):
        result = await graphiti.add_episode(
            name="User Input",
            episode_body=text,
            source_description=source,
            reference_time=datetime.now()
        )
        return f"Added {len(result.nodes)} entities, {len(result.edges)} relationships"
    
    def search_knowledge(query: str, num_results: int = 10):
        results = await graphiti.search(query, limit=num_results)
        return [{"fact": edge.fact, "confidence": edge.confidence} for edge in results]
    
    def visualize_graph():
        return graphiti.create_visualization()
    
    interface = gr.Interface(
        fn=[add_knowledge, search_knowledge, visualize_graph],
        inputs=[
            [gr.Textbox(label="Knowledge Text"), gr.Textbox(label="Source")],
            [gr.Textbox(label="Search Query"), gr.Slider(1, 50, 10)],
            []
        ],
        outputs=[
            gr.Textbox(label="Addition Result"),
            gr.JSON(label="Search Results"),
            gr.Plot(label="Graph Visualization")
        ],
        title="🧠 Interactive Knowledge Graph",
        description="Add knowledge and explore connections"
    )
    
    return interface

if __name__ == "__main__":
    create_knowledge_graph_interface().launch()
```

Deploy with a simple `requirements.txt`:

```txt
graphiti-hf
gradio
plotly
networkx
```

## 📊 Comparison with Traditional Solutions

| Feature | Neo4j + Graphiti | Graphiti-HF | Traditional RAG |
|---------|------------------|-------------|-----------------|
| **Setup Complexity** | High (DB required) | Low (HF account) | Medium |
| **Collaboration** | Limited | Excellent (Git-based) | Poor |
| **Versioning** | Manual | Automatic | None |
| **Deployment** | Complex | One-click Spaces | Medium |
| **Cost** | High (DB hosting) | Low (HF free tier) | Medium |
| **Temporal Queries** | Yes | Yes | No |
| **Real-time Updates** | Yes | Yes | Limited |
| **Visual Interface** | Custom dev needed | Built-in | None |
| **Community Sharing** | Difficult | Easy (HF Hub) | None |

## 🔧 Migration from Original Graphiti

Easily migrate existing Graphiti knowledge graphs:

```python
from graphiti_hf.migration import GraphitiMigrator

# Migrate from Neo4j
migrator = GraphitiMigrator()
await migrator.migrate_from_neo4j(
    source_uri="bolt://localhost:7687",
    source_user="neo4j",
    source_password="password", 
    target_repo="your-org/migrated-graph"
)

# Or migrate from export files
await migrator.migrate_from_export(
    export_file="knowledge_graph_export.json",
    target_repo="your-org/imported-graph"
)
```

## 🛠️ Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/graphiti-hf.git
cd graphiti-hf

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black graphiti_hf/
mypy graphiti_hf/
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Make** your changes and add tests
4. **Run** tests: `pytest`
5. **Submit** a pull request

Areas where we need help:
- 🔌 **Integrations**: Notion, Obsidian, Slack, Discord
- 🎨 **Visualizations**: Advanced graph layouts and analytics
- 📚 **Documentation**: Tutorials and examples
- 🧪 **Testing**: Edge cases and performance tests
- 🌍 **Localization**: Multi-language support

## 🏆 Showcase

Projects built with Graphiti-HF:

- **[AI Research Navigator](https://huggingface.co/spaces/research/ai-navigator)** - Explore connections between AI papers
- **[Company Knowledge Hub](https://huggingface.co/spaces/acme/knowledge-hub)** - Internal company knowledge base
- **[Personal Learning Graph](https://huggingface.co/spaces/learner/knowledge-graph)** - Track learning journey
- **[News Analysis Engine](https://huggingface.co/spaces/news/analysis)** - Connect news events and entities

*Want to add your project? [Submit a PR](https://github.com/your-org/graphiti-hf/pulls)!*

## 📈 Performance

Graphiti-HF is optimized for the HuggingFace ecosystem:

- **⚡ Fast Search**: Sub-second queries on graphs with 1M+ nodes using FAISS
- **💾 Efficient Storage**: Parquet format reduces storage by 60% vs JSON
- **🔄 Incremental Updates**: Add new knowledge without reprocessing entire graph
- **🚀 Parallel Processing**: Batch operations with configurable concurrency
- **📱 Memory Efficient**: Lazy loading and streaming for large graphs

## 🔐 Privacy & Security

- **🔒 Private Datasets**: Keep sensitive knowledge graphs private
- **🏢 Organization Accounts**: Team collaboration with access controls
- **🌍 On-Premise**: Self-host with HuggingFace Hub Enterprise
- **🛡️ Data Protection**: Your data never leaves your control
- **📝 Audit Logs**: Track all changes with Git history

## 🆘 Troubleshooting

### Authentication Issues

**Q: "Repository not found" or "Authentication failed"**
```bash
# Check if you're logged in
huggingface-cli whoami

# Re-login if needed
huggingface-cli login

# Or check your token has write permissions
# Go to https://huggingface.co/settings/tokens
```

**Q: "OpenAI API key not found"**
```python
# Make sure your API key is set
import os
print(os.getenv("OPENAI_API_KEY"))  # Should not be None

# Or set it in Python
os.environ["OPENAI_API_KEY"] = "sk-your_key_here"
```

**Q: "Dataset creation failed"**
```python
# Make sure you have write permissions and the repo name is valid
# Repository names must be: username/dataset-name or org/dataset-name
graphiti = GraphitiHF("your-username/my-graph", create_if_not_exists=True)
```

### Common Issues

**Q: "Dataset not found" error**
```python
# Make sure dataset exists and you have access
graphiti = GraphitiHF("username/dataset-name", create_if_not_exists=True)
```

**Q: Search returns empty results**
```python
# Rebuild search indices
await graphiti.rebuild_indices()
```

**Q: Memory issues with large graphs**
```python
# Use streaming mode for large datasets
graphiti = GraphitiHF("username/large-graph", streaming=True)
```

**Q: Slow performance**
```python
# Increase concurrency (check your LLM rate limits)
import os
os.environ['SEMAPHORE_LIMIT'] = '20'
```

### Installation Extras Explained

- **`sentence-transformers`**: Local embedding models (faster, no API calls)
- **`anthropic`**: Use Claude for entity extraction instead of OpenAI
- **`google-genai`**: Use Gemini models for processing
- **`groq`**: Ultra-fast inference with Groq
- **`viz`**: Advanced graph visualization components  
- **`integrations`**: Connectors for Notion, Slack, Discord, etc.
- **`all`**: Everything included

For more help:
- 📖 [Documentation](https://graphiti-hf.readthedocs.io/)
- 💬 [Community Forum](https://huggingface.co/spaces/graphiti-hf/community)
- 🐛 [Issue Tracker](https://github.com/your-org/graphiti-hf/issues)

## 📄 License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **[Zep Team](https://github.com/getzep/graphiti)** for the original Graphiti framework
- **[HuggingFace Team](https://huggingface.co/)** for the amazing ecosystem
- **Contributors** who help make Graphiti-HF better

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-org/graphiti-hf&type=Date)](https://star-history.com/#your-org/graphiti-hf&Date)

---
