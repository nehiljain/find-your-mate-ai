"""
This script is used to scrape the cofounder matching profiles from startupschool.org
and save them the screenshot and page content as html and markdown formats
"""
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Optional, List

import html2text
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from playwright.sync_api import sync_playwright

from pydantic import BaseModel
from unstructured.partition.html import partition_html
from unstructured.staging.base import convert_to_dataframe

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  handlers=[
    logging.FileHandler('scraper.log'),
    logging.StreamHandler()
  ]
)
# TODO: Migrate to Dynaconf
load_dotenv(find_dotenv())
class FounderProfile(BaseModel):
    """
    CleanProfile is a Pydantic model that represents a cofounder matching profile.
    """
    filepath: Path
    profile_url: str
    linkedin_url: Optional[str]
    md_content: str
    last_seen: str
    location: str
    age: int


def check_if_file_is_html(filepath: str) -> bool:
    """
    Check if the file is html.
    Args:
        filepath (str): The path to the file.
    Returns:
        bool: True if the file is html, False otherwise.
    """
    with open(filepath, "r") as file:
        text = file.read()
        html_content = "\n".join(text.split("\n")[1:])
        # using beautifulsoup to parse the html
        soup = BeautifulSoup(html_content, "html.parser")
        # fetch the content of the div with title "Last seen on co-founder matching"
        last_seen = soup.find("div", {"title": "Last seen on co-founder matching"})
        # check if html_content is html not plain text
        return True if last_seen else False


def extract_clean_md_urls(filepath: Path) -> FounderProfile:
    with filepath.open("r") as file:
        text = file.read()

        profile_url = re.search(
            r"https://www.startupschool.org/cofounder-matching/candidate/\w+", text
        )
        profile_url = profile_url.group(0)
        linkedin_url = re.search(r"https://linkedin.com/in/[\w-]+", text)
        linkedin_url = linkedin_url.group(0) if linkedin_url else None
        # get all lines except first
        html_content = "\n".join(text.split("\n")[1:])
        elements = partition_html(text=html_content)

        content_dataframe = convert_to_dataframe(elements)
        linkedin_index = content_dataframe[
            content_dataframe["text"] == "View on LinkedIn"
        ].index
        if len(linkedin_index) > 0:
            linkedin_index = linkedin_index[0]
            # delete all rows after linkedin_index
            content_dataframe = content_dataframe.drop(
                content_dataframe.index[linkedin_index + 1 :]
            )

        # find index of the first match of text = "Save to favorites"
        save_to_fav_index = content_dataframe[
            content_dataframe["text"] == "Save to favorites"
        ].index
        if len(save_to_fav_index) > 0:
            save_to_fav_index = save_to_fav_index[0]
            # delete all rows before save_to_fav_index
            content_dataframe = content_dataframe.drop(
                content_dataframe.index[: save_to_fav_index + 1]
            )
        html_parser = html2text.HTML2Text()
        markdown_output = ""
        content_dataframe["text_as_html"] = content_dataframe["text_as_html"].fillna("")
        for _, row in content_dataframe.iterrows():
            if row["type"] == "Title":
                markdown_output += f"## {row['text']} \n\n"
            elif len(row["text_as_html"]) > 0:
                markdown_output += html_parser.handle(row["text_as_html"])
            else:
                markdown_output += f"{row['text']} \n\n"

        # fetch the content from html content for last seen, location and age
        soup = BeautifulSoup(html_content, "html.parser")
        last_seen = soup.find("div", {"title": "Last seen on co-founder matching"})
        location = soup.find("div", {"title": "Location"})
        age = soup.find("div", {"title": "Age"})

        return FounderProfile(
                filepath=filepath,
                profile_url=profile_url,
                linkedin_url=linkedin_url,
                md_content=markdown_output,
                last_seen=last_seen.text,
                age=int(age.text) if age and age.text.strip() else -1,
                location=location.text,
            )



def get_all_cofounder_text_files():
    # using rglob find all the files in data folder and pull into a list
    return list(Path("data").rglob("*.txt"))


def load_cookies(cookie_path: Optional[str] = None):
    """
    Load cookies from a specified path or default path.
    """
    if cookie_path is None:
        cookie_path = "./startup-school-cookies.json"
    with open(cookie_path, "r") as f:
        return json.load(f)


def slugify(text):
  """
  Slugify the text for filename
  """
  pattern = r'[^\w+]'
  return re.sub(pattern, '-', text.lower().strip())


def get_output_path(filename):
  # write a function to get path from .env or create a data folder to store the output
  # add the filename to the path
  output_path = Path(os.getenv('OUTPUT_PATH', 'data'))
  output_path.mkdir(exist_ok=True)
  return output_path / filename


def get_url_from_file_text(filepath):
    with open(filepath, "r") as file:
        text = file.read()
        # get first line
        profile_url = re.search(
            r"https://www.startupschool.org/cofounder-matching/candidate/\w+", text
        )
        return profile_url.group(0)

# TODO: Add a step to validate the full page was stored in the output file else retry

import typer

app = typer.Typer()

@app.command()
def main(urls: List[str] = typer.Option(None, help="List of URLs to scrape."),
         num_profiles: int = typer.Option(None, help="Number of profiles to scrape dynamically. Cannot be used with URLs."),
         cookie_path: Optional[str] = typer.Option(None, help="Path to the cookie file. If not provided, defaults to './startup-school-cookies.json'")):
    """
    Main function to scrape the cofounder matching profiles from startupschool.org.
    This can either scrape profiles from a list of URLs provided, scrape a specified number of profiles dynamically, or continue until no more profiles are available.
    """
    if urls and num_profiles is not None:
        raise ValueError("Cannot use both 'urls' and 'num_profiles' options at the same time.")

    logging.info("Starting the scraper...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        cookies = load_cookies(cookie_path)
        for cookie in cookies:
            if "sameSite" not in cookie or cookie["sameSite"] not in ["Strict", "Lax", "None"]:
                cookie["sameSite"] = "None"
        context.add_cookies(cookies)
        logging.info("Cookies added to the browser context.")

        if urls:
            for url in urls:
                scrape_profile(context, url)
        elif num_profiles:
            for _ in range(num_profiles):
                scrape_profile(context, "https://www.startupschool.org/cofounder-matching/candidate/next")
        else:
            while True:
                if not scrape_profile(context, "https://www.startupschool.org/cofounder-matching/candidate/next"):
                    break

        context.close()
        browser.close()
        logging.info("Scraper finished successfully.")

def scrape_profile(context, url):
    logging.info("Initiating scraping for URL: %s", url)
    page = context.new_page()
    page.goto(url)
    sleep_time = random.uniform(5, 8)
    logging.info("Sleeping for %.2f seconds to mimic human interaction", sleep_time)
    time.sleep(sleep_time)
    full_content = page.content()
    headings = page.get_by_role("heading").all_text_contents()
    if not headings:
        logging.warning("No headings found for URL: %s. Exiting the scraper for this URL.", url)
        return False
    output_txt_path = get_output_path(f"{slugify(headings[0])}.txt")
    output_screenshot_path = get_output_path(f"{slugify(headings[0])}.png")
    logging.info("Writing full page html content to %s", output_txt_path)
    with open(output_txt_path, "w") as f:
        f.write(f"URL: {page.url}\n")
        f.write(full_content)
    logging.info("Taking screenshot and saving to %s", output_screenshot_path)
    page.screenshot(path=output_screenshot_path, full_page=True)

    # Generate markdown file from the HTML content
    logging.info("Extracting structured data from %s", output_txt_path)
    founder_profile = extract_clean_md_urls(output_txt_path)
    output_md_path = get_output_path(f"{slugify(headings[0])}.md")
    logging.info("Writing markdown output to %s", output_md_path)
    with open(output_md_path, "w") as md_file:
        md_file.write(founder_profile.md_content)
        md_file.write("\n\nOther Metadata\n\n")
        md_file.write(f"- Location: {founder_profile.location}\n\n")
        md_file.write(f"- Age: {founder_profile.age}\n\n")
        md_file.write(f"- Last seen: {founder_profile.last_seen}\n\n")
        md_file.write(f"LinkedIn URL: {founder_profile.linkedin_url}\n")
        md_file.write(f"Profile URL: {founder_profile.profile_url}\n")

    logging.info("Completed processing for URL: %s", url)
    return True

if __name__ == "__main__":
    app()

