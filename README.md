# NexusBrain: A Research Collaboration Recommendation System
## Overview of the Project
NexusBrain is a smart recommendation system that leverages advanced Natural Language Processing (NLP) and Machine Learning (ML) techniques to promote impactful research collaborations. Designed initially for a university academic environment, it systematically analyzes faculty research interests, publications, and academic backgrounds to suggest optimal collaborations across departments and disciplines.

## Purpose & Motivation
In many academic institutions, interdisciplinary collaboration often happens by chance rather than by design. Researchers working on complementary or overlapping topics may remain unaware of each other’s work simply because there’s no structured system to connect them. This leads to:
- Missed opportunities for impactful research
- Duplication of efforts
- Underutilization of internal academic resources

NexusBrain was developed to solve this problem using a systematic, data-driven approach. Its core motivation is to bridge these collaboration gaps by analyzing publicly available faculty data — such as publications, research interests, and academic accolades — and using intelligent algorithms to recommend potential research partners.

Here’s how NexusBrain tackles the problem:
1. Creating Academic Synergy
NexusBrain uses machine learning to analyze faculty profiles and match researchers based on thematic and technical similarities. This enables faculty to discover collaborators they may never have encountered through traditional networking.
2. Promoting Interdisciplinary Research
By not being confined to departmental boundaries, the system identifies shared research themes across diverse fields — encouraging unexpected yet valuable partnerships between disciplines like Computer Science and Sociology, or Materials Science and Bioinformatics.
3. Enhancing Resource Utilization
Universities already house rich data about their researchers — but it’s often underused. NexusBrain taps into this existing data to build collaboration bridges without requiring additional surveys or manual input.
4. Supporting Long-Term Academic Impact
Better collaboration leads to more innovative research, stronger grant proposals, and a higher likelihood of real-world impact. NexusBrain helps lay the foundation for these high-impact outcomes by intelligently connecting the right minds at the right time.

## Scope of the Project
The scope of NexusBrain encompasses the end-to-end development and evaluation of a recommendation system aimed at enhancing academic collaboration. The project was designed with the following key objectives:
### 1. Develop a Research Collaboration Platform
The core aim of the project was to design and build a smart, end-to-end research collaboration platform that could analyze academic profiles and recommend potential collaborators. This involved collecting and preprocessing faculty data (such as research interests, publications, and accolades), applying natural language processing (NLP) techniques to extract key themes, and using machine learning algorithms to match researchers based on thematic similarities. The platform automates what would otherwise be a time-consuming and informal process, making collaboration discovery faster, smarter, and data-driven.
### 2. Evaluate the Efficacy of the Recommendation System
Beyond building the system, a critical part of the project was testing how accurately and effectively it could identify relevant collaboration opportunities. We evaluated multiple machine learning models, including K-Means clustering and Random Forest classification, using both real and synthetic datasets. Metrics such as accuracy, consistency of cluster predictions, and relevance of matches were used to assess performance. This phase ensured that the recommendations weren’t just technically functional, but practically meaningful and reliable for academic use.
### 3. Foster Interdisciplinary Partnerships
One of the most impactful goals of NexusBrain was to break academic silos and foster interdisciplinary research. The system was designed to go beyond traditional departmental boundaries and identify commonalities in research areas that might not be obvious at first glance. By analyzing research descriptions and publication data across diverse fields, it could recommend unexpected yet promising collaborations—such as linking a data scientist with a public health researcher—thus encouraging innovation through the blending of perspectives and expertise.

### 4. Create Scalable and Generalizable Architecture
To ensure long-term usability, the project was developed with scalability and generalizability in mind. The architecture was made modular and adaptable so that it could easily be extended to other departments, institutions, or even different countries. The system does not rely on hardcoded data structures and can accommodate varied datasets with minimal adjustments. This design philosophy opens the door for broader implementation, such as building a university-wide or global AI-powered research network to systematically drive collaboration across institutions.

## System Architecture
The system workflow includes:
### 1. Data Collection & Web Scraping
Used requests and BeautifulSoup libraries to scrape faculty data:
- Name
- University
- Publications
- Research Overview
- Awards
- Total Citations
### 2. Data Preprocessing & NLP Pipeline
- Tokenization
- PoS Tagging
- Named Entity Recognition (NER)
- Lemmatization
- TF-IDF vectorization
- Word2Vec and Document Embedding with Attention Weights
### 3. Feature Extraction
- Extracted key research terms and contextual meaning using NLP
- Analyzed term relevance using TF-IDF
- Calculated importance of accolades and publication count
### 4. Clustering & Classification
- K-Means Clustering: Unsupervised clustering of researchers based on research profiles
- Random Forest Classifier: Supervised learning based on citation/award metrics to predict researcher clusters

## Results & Analysis
- Achieved 85% accuracy on real datasets, and 90% accuracy with synthetic data
- Demonstrated that term frequency impacts recommendation quality {Example: "Swarm" had a higher TF-IDF score (0.8417) than "Robot" (0.5399)}
- Initially tested on the McCormick dataset (smaller scope), followed by the entire faculty dataset
- With more data, predictions became more stable and accurate (e.g., consistent cluster predictions from both K-Means and Random Forest)

## Libraries & Tools Used
- requests: For sending HTTP requests and fetching faculty profile web pages.
- BeautifulSoup: For parsing HTML and extracting structured data from messy web content.
- NLTK: Used for text preprocessing tasks like tokenization, lemmatization, and PoS tagging.
- spaCy: For efficient NLP tasks including Named Entity Recognition and syntactic parsing.
- scikit-learn: Powered TF-IDF vectorization, K-Means clustering, and Random Forest classification.
- gensim: Used to create Word2Vec embeddings for semantic similarity analysis.
- matplotlib: For plotting graphs and visualizing model performance and cluster outputs.
- seaborn: For generating aesthetically pleasing statistical data visualizations.

## Key Features
### 1. End-to-End Data Pipeline: From Web Scraping to Recommendation Output
NexusBrain offers a fully automated, end-to-end system that handles every step of the research collaboration recommendation process. It starts by scraping publicly available faculty data from university websites using requests and BeautifulSoup. This includes research interests, publications, awards, and citation counts. The raw data is then cleaned and processed through an NLP pipeline, where meaningful keywords and themes are extracted using techniques like lemmatization, tokenization, TF-IDF, and Word2Vec embeddings. Finally, this enriched textual data is passed through machine learning models (K-Means and Random Forest) to cluster and classify researchers, generating personalized collaboration recommendations. The entire process from unstructured data to actionable output is integrated, efficient, and reproducible.

### 2. Domain-Agnostic NLP Preprocessing Pipeline
One of NexusBrain’s standout features is its domain-agnostic Natural Language Processing pipeline, which means it doesn’t rely on domain-specific templates or manually curated taxonomies. Instead, it uses scalable NLP techniques like Named Entity Recognition (NER), PoS tagging, lemmatization, and TF-IDF scoring to clean and extract research themes across a wide range of academic disciplines. This allows the system to function just as well for a Computer Science professor working on “deep learning” as it does for a Humanities professor researching “postcolonial literature.” The flexibility of this pipeline makes NexusBrain easily expandable to different departments and even entirely different universities without needing custom code for each field.

### 3. Real-Time Researcher Profile Matching
NexusBrain can dynamically match researchers based on the latest available data, enabling real-time discovery of collaboration opportunities. Once a faculty member’s profile is scraped and processed, the system can immediately position them within a cluster of similar researchers and identify potential collaborators based on semantic similarity, publication alignment, and research overlap. This feature is particularly useful in fast-evolving research environments, where new faculty members join, new papers are published, or research focuses shift. Rather than relying on outdated databases or manual search, NexusBrain provides on-the-fly, data-driven matching that evolves with the academic ecosystem.

### 4. Prioritization Based on Awards and Publication Counts
Not all researchers are equally positioned in terms of experience, recognition, or academic contribution. To account for this, NexusBrain integrates quantitative metrics like number of publications, total citation count, and academic awards into its recommendation logic. Researchers with a stronger track record are given higher priority in the recommendation list, ensuring that the suggestions are not just thematically relevant but also academically impactful. This weighting mechanism helps identify not only who shares similar interests, but also who is likely to be the most influential or productive partner in a potential collaboration, a crucial factor in real-world research dynamics and funding proposals.

## Future Scope

### 1. Expand to Multiple Universities and Academic Networks
Currently designed for use within a single academic institution, NexusBrain has the potential to scale and serve as a multi-institutional platform. By adapting its data pipeline to work across different university websites and formats, the system can facilitate cross-university research collaborations, breaking institutional silos and fostering a broader academic ecosystem. This would allow faculty and researchers from different colleges, departments, or even countries to find partners beyond their local networks.

### 2. Integrate Transformer-Based NLP Models for Richer Context Understanding
While the current system uses TF-IDF and Word2Vec for text representation, future iterations could benefit greatly from state-of-the-art Transformer-based models like BERT, RoBERTa, or SciBERT. These models offer contextualized embeddings that understand the meaning of words in relation to surrounding text, enabling deeper semantic analysis of research content. This would improve matching accuracy, especially in complex, jargon-heavy academic writing, and allow the system to better grasp nuanced differences between similar research fields.

### 3. Build Industry-Academia Collaboration Modules
Beyond academic circles, NexusBrain can be extended to connect researchers with industry experts, R&D labs, think tanks, and innovation hubs. By incorporating data from corporate research teams and industrial publications, the platform could recommend mutually beneficial partnerships between academia and industry facilitating real-world impact, joint publications, internships, funding opportunities, and technology transfer.

### 4. Create a Global Research Network Powered by AI
The long-term vision is to transform NexusBrain into a global AI-powered research collaboration network, capable of processing vast datasets across institutions and disciplines. This network would allow any researcher regardless of geography to find ideal collaborators, track emerging research trends, and access data-driven insights about the global academic landscape. In essence, NexusBrain could become the LinkedIn of academic research, driven not by social connections, but by research relevance and scholarly alignment.

## Team & Contribution
This project was a team-based academic initiative, collaboratively designed and implemented over several months. Each member contributed to different modules: data scraping, NLP pipeline, machine learning, and documentation.
