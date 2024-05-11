# find-your-mate-ai


## Overview


The Find Your Mate AI project is designed to scrape and process cofounder matching profiles from [startupschool.org](https://www.ycombinator.com/cofounder-matching). It involves scraping the web pages, extracting relevant information, and storing it in a structured format for Querying them.

This is a project born out of my own itch and helps me learn how to use LLM's to build a search based apps.


## Setup and Installation


### Prerequisites


- Python 3.8 or higher
- macOS operating system
- MongoDB Account [Register here](https://www.mongodb.com/docs/guides/atlas/account/)
- YC Account Profile [Signup here](https://www.ycombinator.com/cofounder-matching)
- EditThisCookie Chrome Extension [Install here](https://chromewebstore.google.com/detail/editthiscookie/fngmhnnpilhplaeedifhccceomclgfbg)

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

3. Export Cookies by following the instructions at
[EditThisCookie Chrome Extension](https://docs.apify.com/academy/tools/edit-this-cookie#export-cookies)

## Environment Variables


- Create settings.toml and .secrets.toml in the root directory of the project and add the following environment variables:




## Running the scripts

### Scraper

To scrape the cofounder matching profiles, run the following command:
```
pdm run python src/find_your_mate_ai/scraper.py --cookie-path ".startupschool-cookie.json"
```


### Data Ingestion

To ingest the data from the specified directory path and index it using LlamaIndex, run the following command:
```
pdm run python src/find_your_mate_ai/data_ingestion.py
```
