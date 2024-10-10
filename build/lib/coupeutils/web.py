import requests
from typing import Dict, Optional, Any
from io import BytesIO
from zipfile import ZipFile
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging


class WebUtils:

    @staticmethod
    def download_from_url(url: str) -> Optional[requests.Response]:
        """Downloads content from a URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
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
    def get_page_content_selenium(url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
        """Retrieves page content using Selenium for JavaScript rendering."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=chrome_options)

        # Set custom headers if provided
        if headers:
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": headers.get("User-Agent", "")})
            driver.execute_cdp_cmd('Network.enable', {})
            for key, value in headers.items():
                driver.execute_cdp_cmd('Network.setExtraHTTPHeaders', {"headers": {key: value}})

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

    def get_text_and_links_selenium(self, url):
        driver = webdriver.Chrome()  # Or whichever driver you're using
        driver.get(url)

        # Wait for the body to be present
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Extract all text
        text_content = driver.find_element(By.TAG_NAME, "body").text

        # Extract all hyperlinks
        links = driver.find_elements(By.TAG_NAME, "a")
        hyperlinks = [
            {"text": link.text, "href": link.get_attribute("href")}
            for link in links
            if link.get_attribute("href")
        ]

        driver.quit()

        return text_content, hyperlinks

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
            "div.main-content",
            "body",
        ]

        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                # Remove script and style elements
                for element in main_content(["script", "style"]):
                    element.decompose()

                # Remove empty elements
                for element in main_content.find_all():
                    if len(element.get_text(strip=True)) == 0:
                        element.decompose()

                return str(main_content)

        logging.warning("Main content not found using any of the specified selectors.")
        return None
