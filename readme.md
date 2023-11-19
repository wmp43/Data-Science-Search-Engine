# WikiSearch ReadMe
### Motivation: Build large scale NLP/IR system using wikipedia as a case study
## Core Functionality by Feature
1. Search 
    - Develop an IR system to return relevant documents based on tags, input query, etc.  
    - Text queries, but also date, author, tags, metadata...
    - Tech Used: VecDB, VecEmbeddings, Similarity Search
2. Text Generation 
   - Answer questions about retrieved or uploaded documents  
   - Summarize retrieved or uploaded documents
   - Tech Used: Fine-tuned LLM
3. Version Control
   - txt/pdf flavor of classical version control -- remaking git for text documents (literally Google Drive)
   - I will probably learn more about programming/swe from this than any of the other topics combined
4. Text Editing Module
   - Being able to edit these documents live with someone else
   - Annotation ability too
5. Data Visualization
   - Documents connections and tags, but also search trend analysis
   - That search trend analysis would require another db
6. 
 ## Core Functionality by Tech
I feel like many of these technologies could overlap onto each other to offer even more. 
Kind of a crazy example is a single query bar for all functionality based on query the proper function is applied to query
1. Classical ML (Most Obvious Case is that this will aid vector search and query options)
   - Document Classification (Person, Technology, etc.)
     - Added Functionality is highlighting most important tokens.
   - named Entity Recognition as a part of the processing phase
2. Generative Language Models (GLM)
   - Question Answering
   - Wikipedia page drafting
3. Vector Search and Storage (VecS&S)
   - Doc Retrieval
   - Precedent Search
4. Combinations of above
   - VecSearch and Storage & Generative LM: Risk Assessments
   - Classical ML & VectorSS: ML will help query specificity for VecS&S



#### Other Ideas that may become viable with greater effort or a prod Licensed Release:
##### This should be the goal would be so sick
1. Version Controlling Files
   - This is like building a whole new Git for text documents. Would be a great chance to learn Rust or C++ or something else cool
   - Fucking hard; it seems hard/challenging meaning it will be twice as hard/challenging in practice
   - Introduces a large amount of front-end work (ew) work as well.

2. Custom Implementation for a data ingestion & processing
   - Meaning how does a new client stream their current data/txt/documents into their vector db for search?
   - This needs to be handled for backend implementation


#### ChatGPT Suggestions
1. Detailed Features and User Stories
   - To ensure building a useful tool that is used and needed: add detailed user stories for each core functionality. This keeps development user-centric
2. Data Privacy and Security
   - Legal docs are sensitive, don't fuck it up
3. Scalability and Performance
   - How will the system scale? Larger Instances to make up for un-optimized code lmao
4. Integration with existing tools
   - How does this integrate with existing tools, how does data get ported into our system
   - What happens if user cannot find desired file via text? Needs to still have some visual directory
   - Could also Visualize Database in 3D to find relevant documents by type, themes, etc.
5. Analytics and reporting
   - Data Viz for Vec DB


### Items to Look At:
1. https://docs.trychroma.com/
2. Multi-Dim Vec Clustering - https://arxiv.org/pdf/2012.08466.pdf
3. Retreival Augmented Generation - https://arxiv.org/pdf/2005.11401v4.pdf
   - Useful for question answering
