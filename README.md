# AI_FOR_BLOCKCHAIN
Jupyter notebooks for ai fraud detection and tagging

# Hackhaton Encode 2024

## Empowering the Solana Blockchain with AI-driven content moderation and fraud detection to create a safer, more trustworthy digital ecosystem.




### Problem definition:



Manipulation of Decentralized Exchanges (DEXs): Decentralized exchanges like Mango Market on Solana are designed to provide trustless and transparent trading. However, they can still be vulnerable to manipulation and exploits. In the case of Avraham Eisenberg's manipulation, it appears that he exploited vulnerabilities in the smart contracts or trading mechanisms of Mango Market to acquire and liquidate assets worth $116 million. Such incidents underscore the importance of robust security measures, code audits, and community vigilance in ensuring the integrity of decentralized exchanges. [1] [2]

Fake Solana Giveaways: Scammers often exploit the popularity and hype surrounding blockchain platforms like Solana to trick users into participating in fake giveaways or airdrops. These scams typically involve social engineering tactics, such as impersonating prominent figures or projects associated with Solana, and enticing users to send cryptocurrency in exchange for promised rewards. Such scams highlight the importance of user education, skepticism towards unsolicited offers, and e verifying the authenticity of information before taking any action. [3]

Fraudulent Transactions in DeFi: DeFi platforms, including those built on Solana, are susceptible to various types of fraudulent activities, including rug pulls, fake token offerings, honeypot contracts, and Ponzi schemes. These schemes exploit vulnerabilities in smart contracts, liquidity pools, or token mechanisms to deceive users and siphon funds from unsuspecting investors. [4]

### Proposed Solution:


We'll start by developing an ML model to detect fraudulent transactions. Since labeled data for Solana is unavailable, we'll train the model on the Ethereum-Fraud-Dataset. Principal Component Analysis (PCA) will extract key fraud-related features, which we'll fine-tune using transfer learning on similar Solana data.

Next, we're crafting a chatbot with Large Language Models (LLM) to aid Solana users with transactions and blockchain use. This LLM model will be trained to recognize and warn against potential scams, like Ponzi Schemes, by analyzing text patterns. 

Previous models, such as LSTM networks, have been effective in identifying and addressing DeFi fraud [4].
 We're exploring advanced techniques like Transformers to enhance detection accuracy for complex fraud patterns.

Ultimately, both models will merge into a single platform, offering real-time assistance to Solana users, effectively combating fraudulent activities.



###Work done so far:
The initial step has achieved a remarkable accuracy of 98.9%. Key features highly correlated with fraudulent transactions include 'Time_Diff_between_first_and_last_(Mins)', 'Avg_min_between_received_tnx', 'total_transactions_(including_tnx_to_create_contract', 'Received_Tnx', 'Sent_tnx', 'avg_val_sent', and 'Unique_Sent_To_Addresses', among others.

Additionally, progress has been made on developing the LLM chatbot, currently in the midst of testing.

###Work to be done:
Fine-tune the existing model using transfer learning on Solona's unlabelled transaction dataset.
Implement an advanced LLM-based chatbot to detect and advise on sophisticated DeFi fraud, including Rug pull, Fake token offering, Honeypot contract, and Ponzi scheme.

Finally, both models need to be combined in a singualar platform.

### Benefits of our approach against existing solutions:

Competitor 1: Blockchain Watchdog

Reasons: Blockchain Watchdog relies on manual reporting mechanisms, which limits its ability to detect fraudulent activities in real-time. Additionally, it lacks incentives for community engagement, hindering the effectiveness of flagging illicit activities.

Competitor 2: Solsniffer

Description: Solsniffer analyzes previous Solana tokens and estimates whether they are potentially scams or not based on historical data. Hence it cannot detect for newer addresses.[5]



### Work done so far:

The first step mentioned above has been successfully implemented with an accuracy of 98.9%. The highly correlated features in regard with fraudulent transactions are 'Time_Diff_between_first_and_last_(Mins)',  'Avg_min_between_received_tnx', 'total_transactions_(including_tnx_to_create_contract', 'Received_Tnx', 'Sent_tnx', 'avg_val_sent', and 'Unique_Sent_To_Addresses', among others.

Moreover, the intial LLM chatbox is still in development and is halfway through testing phase.

### Work to be done:

Using transfer learning, the current model should be fined tuned on the unlabelled dataset available for Solona Transaction on Google Cloud. [6]

Implementation of advanced LLM based chat bot to advice and detect sophisticated DeFi fraud such as  Rug pull, Fake token offering, Honeypot contract and Ponzi scheme.


# AI-Enhanced Security for Solana Blockchain

## Introduction

This project aims to leverage artificial intelligence (AI) to enhance the security within the Solana blockchain network. Using advanced AI techniques, we analyze transaction data to detect potential fraud and ensure content integrity, providing a safer blockchain ecosystem for users and developers.

Crypto Transactions Fraud Detection
This project aims to detect fraudulent activities in cryptocurrency transactions using machine learning techniques. The dataset used in this project is obtained from an open-source platform (Kaggle) and contains information about various crypto transactions.
Data Extraction
Frequency of Transactions
•	Analyze Sent tnx and Received Tnx distribution; spikes or drops may indicate unusual activity.
Timing Patterns
•	Investigate Avg min between sent tnx and received tnx for transaction regularity; abrupt changes suggest fraud.
Transaction Amounts
•	Examine min/max/avg values received and sent; compare with contract interactions for anomalies.
ERC20 Token Transactions
•	Review ERC20-related columns for token transactions; watch for irregularities compared to ETH transactions.
Address Patterns
•	Analyze Unique Received from and Sent To Addresses for involvement; watch for sudden address increases.
Total Ether Balance
•	Detect discrepancies in total ether balance for potentially fraudulent activities.
Contract Creation
•	Investigate Created Contracts for sudden increases; may indicate fraudulent smart contracts.
Time Span
•	Analyze Time Diff between first and last (Mins) for transaction duration anomalies.
ERC20 Token Types
•	Monitor most sent/received ERC20 token types for sudden changes indicating fraud.
Transaction Patterns Over Time
•	Visualize transaction-related features over time to spot irregularities.
Pre-Processing
1.	Data Preprocessing:
•	Removing special characters, making the data more uniform, handling missing values and imputation, Feature extraction, Feature engineering, PCA analysis, and finding most important features to build a model.
•	Tokenization of categorical columns using MultiLabelBinarizer.
•	Standardization of numeric columns using StandardScaler.
•	Principal Component Analysis (PCA) for dimensionality reduction.
•	Handling missing values and renaming columns.
2.	Exploratory Data Analysis (EDA):
•	Visualization of missing values heatmap.
•	Correlation analysis using a heatmap to understand feature relationships.
•	Identification of top features based on correlation with the target variable.
3.	Model Training and Evaluation:
•	Utilization of a RandomForestClassifier as the base model.
•	Splitting the data into train, validation, and test sets.
•	Evaluation of the model on the validation set for ROC AUC and accuracy.
•	Cross-validation to assess model performance on training data.
•	Final evaluation on the test set for generalization performance.
4.	Results:
•	Assessment of ROC AUC and accuracy on validation and test sets.
•	Calculation of mean ROC AUC and accuracy through cross-validation.
5.	Insights:
•	The model demonstrates robustness in terms of performance across different datasets.
•	It provides insights into feature importance through correlation analysis.
•	Visualization techniques aid in understanding data characteristics and model performance.
6.	Further Improvements:
•	Experimentation with different machine learning algorithms to explore performance variations.
•	Feature engineering to enhance model interpretability and predictive power.
•	Fine-tuning hyperparameters for optimizing model performance.
Overall, the model presents a structured approach to fraud detection, encompassing data preprocessing, exploratory analysis, model training, and evaluation, with the potential for further enhancements and optimizations, achieving the Mean ROC AUC: 0.9977501839597485 and Mean Accuracy: 0.989996343785065.



## Methodology FOR LLMS to Use ai for user security and assistance in Defi marketplace

The project utilizes a multi-step process involving data loading, processing, AI model interaction, and indexing via Pinecone, followed by AI analysis for content and transaction security. Here is an overview of the methodology used:

1. Library Installation: Essential libraries are installed to support AI model functions, data processing, and vector storage.

2. Data Loading: Data is loaded from a PDF, likely containing Solana whitepapers or relevant blockchain transaction data.

3. Text Splitting: Loaded data is split into manageable chunks for easier processing.

4. Embeddings Generation: Each text chunk is processed to create embeddings that capture the semantic meaning of the text, which are essential for similarity searches.

5. Vector Storage with Pinecone: The generated embeddings are stored using Pinecone, a vector database optimized for similarity search.

6. Similarity Search Setup: The infrastructure for performing similarity searches on the indexed data is initialized.

7. AI Model Interaction: Utilizing a pre-trained model, likely a version of GPT (such as a Llama model), the system is set up to process queries and detect patterns indicative of fraudulent activity.

8. Index Creation and Management: An index is created in Pinecone to manage the embeddings, allowing for efficient retrieval during similarity searches.

9. Model Conversion and Loading: The AI model is converted to a format suitable for the environment, and loaded for execution.



## Execution Flow

The code in `LLM_SOL.ipynb` appears to follow these steps for execution:

1. Install necessary Python packages using `pip`.
2. Load the dataset from a PDF file, which in this context is the Solana whitepaper.
3. Split the loaded document into smaller chunks using a text splitter.
4. Generate embeddings for each text chunk using a pre-trained model from Hugging Face.
5. Initialize the Pinecone environment and create an index for storing embeddings.
6. Execute similarity search queries to find relevant information within the indexed data.
7. Pass queries through an LLM (large language model) for processing and analysis.
8. Manage the Pinecone index, checking its existence and stats, and performing cleanup if necessary.

## How to Run

Ensure you have Python 3.x installed and then follow these steps to set up the environment:

```bash
pip install -r requirements.txt


Use JupyterLab




Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change. Please ensure to update tests as appropriate.

License
This project is open-sourced under the MIT license.





#####References
[1]  Know all About the $116 Million Mango Markets Fraud and the Man Who Pulled it Off (marketrealist.com)
[2] https://fortune.com/crypto/2023/01/23/the-solana-hacker-who-took-100m-is-a-bad-man-but-is-he-a-criminal/
[3] https://malwaretips.com/blogs/solana-giveaway-scam/
[4] https://arxiv.org/pdf/2308.15992v1.pdf
[5] https://www.solsniffer.com/
[6] https://solana.com/news/solana-data-live-on-google-cloud-bigquery

