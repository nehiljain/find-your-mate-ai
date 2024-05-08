# find-your-mate-ai


## Overview


The Find Your Mate AI project is designed to scrape and process cofounder matching profiles from startupschool.org. It involves scraping the web pages, extracting relevant information, and storing it in a structured format for further analysis.


## Setup and Installation


### Prerequisites


- Python 3.8 or higher
- macOS operating system


### Installation

1. **Clone the repository:**
```
   git clone https://github.com/your-repository/find_your_mate_ai.git
   cd find_your_mate_ai
```


2. **Install PDM (Python Dependency Manager):**

```
  brew install pdm
```

3. **Install the project dependencies:**
```
     pdm install
```
```
pdm run playwright install
```

## Environment Variables


- create a .env file in the root directory of the project and add the following environment variables:

```
OPENAI_API_KEY=<your-openai-api-key>
```



## Running the scripts

### Scraper

To scrape the cofounder matching profiles, run the following command:
```
pdm run python src/find_your_mate_ai/scraper.py --cookie-path "./custom-path/startup-school-cookies.json" --urls "https://www.startupschool.org/cofounder-matching/candidate/123" "https://www.startupschool.org/cofounder-matching/candidate/456"
```


### Data Ingestion

To ingest the data from the specified directory path and index it using LlamaIndex, run the following command:
```
pdm run python src/find_your_mate_ai/data_ingestion.py
```

