import requests
from typing import Dict, Optional, Any
from io import BytesIO
from zipfile import ZipFile
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import logging


class WebUtils:

    @staticmethod
    def download_from_url(url: str) -> Optional[requests.Response]:
        """Downloads content from a URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading from URL {url}: {e}")
            return None

    @staticmethod
    def download_zip_from_url(url: str) -> Optional[ZipFile]:
        """Downloads a ZIP file from a URL."""
        response = WebUtils.download_from_url(url)
        if response:
            try:
                return ZipFile(BytesIO(response.content))
            except Exception as e:
                logging.error(f"Error extracting ZIP file from URL {url}: {e}")
                return None
        return None

    @staticmethod
    def get_page_content_selenium(url: str) -> Dict[str, Optional[str]]:
        """Retrieves page content using Selenium for JavaScript rendering."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)

        driver.implicitly_wait(10)

        raw_html = driver.page_source
        driver.quit()

        soup = BeautifulSoup(raw_html, "html.parser")

        title = soup.title.string if soup.title else None

        updated_time_meta = soup.find("meta", property="og:updated_time")
        updated_date = updated_time_meta["content"] if updated_time_meta else None

        content_html = WebUtils.extract_main_content(soup)

        return {
            "title": title,
            "updated_date": updated_date,
            "content_html": content_html,
        }

    @staticmethod
    def extract_main_content(soup: BeautifulSoup) -> Optional[str]:
        """Extracts the main content from a BeautifulSoup object."""
        content_selectors = [
            "main[role='main']",
            "main",
            "article",
            "div#content",
            "div.content",
            "div#main",
            "div.main",
            "div#main-content",
            "div.main-content",
            "body",
        ]

        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                return str(main_content)

        logging.warning("Main content not found using any of the specified selectors.")
        return None
