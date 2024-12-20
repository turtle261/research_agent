from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.tools import BraveSearch, WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Union
import time
import json
import logging
import sys
import os
from urllib.parse import urlparse
from datetime import datetime, timezone
import pytz
import platform
import locale
from dataclasses import dataclass
from typing import Optional
import base64
import tempfile
import webbrowser
import subprocess
from PIL import Image
import io
import math

# Selenium imports
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load keys from environment variables or have them empty if not provided
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")

class ModelProvider:
    OLLAMA = "ollama"
    GROQ = "groq"

    @staticmethod
    def get_provider_choice() -> str:
        return ModelProvider.GROQ

class HostTracker:
    def __init__(self, filename="HOSTS.txt"):
        self.filename = filename
        self.failed_hosts = set()
        self.load_failed_hosts()
    
    def load_failed_hosts(self):
        """Load failed hosts from file."""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    self.failed_hosts = set(line.strip() for line in f if line.strip())
                logger.info(f"Loaded {len(self.failed_hosts)} problematic hosts from {self.filename}")
        except Exception as e:
            logger.error(f"Error loading failed hosts: {str(e)}")
            self.failed_hosts = set()
    
    def add_failed_host(self, url: str):
        """Add a failed host to the tracking list."""
        try:
            host = urlparse(url).netloc
            if host and host not in self.failed_hosts:
                self.failed_hosts.add(host)
                with open(self.filename, 'a') as f:
                    f.write(f"{host}\n")
                logger.info(f"Added {host} to problematic hosts list")
        except Exception as e:
            logger.error(f"Error adding failed host: {str(e)}")
    
    def is_problematic_host(self, url: str) -> bool:
        """Check if a URL's host is in the problematic list."""
        try:
            host = urlparse(url).netloc
            return host in self.failed_hosts
        except Exception:
            return False

# Initialize host tracker at module level
host_tracker = HostTracker()

@dataclass
class UserSettings:
    country: str
    timezone: str
    locale: str
    current_time: datetime
    currency: str
    
    @classmethod
    def auto_detect(cls) -> 'UserSettings':
        """Automatically detect user settings from system."""
        try:
            country = "Canada"
            local_tz = pytz.timezone('America/Toronto')
            current_time = datetime.now(local_tz)
            locale_str = 'en_CA.UTF-8'
            locale.setlocale(locale.LC_ALL, locale_str)
            currency = 'CAD'
            logger.info(f"Detected user settings: Country={country}, Timezone={local_tz}, Current time={current_time}")
            return cls(
                country=country,
                timezone=str(local_tz),
                locale=locale_str,
                current_time=current_time,
                currency=currency
            )
        except Exception as e:
            logger.error(f"Error detecting user settings: {str(e)}")
            return cls(
                country="Canada",
                timezone="America/Toronto",
                locale="en_CA.UTF-8",
                current_time=datetime.now(pytz.UTC),
                currency="CAD"
            )

# Initialize user settings at module level
user_settings = UserSettings.auto_detect()

def ensure_size_within_limits(width: int, height: int, max_pixels: int = 33177600) -> tuple:
    """Ensure dimensions are within the pixel limit while maintaining aspect ratio."""
    total_pixels = width * height
    
    # Add a 10% safety margin to max_pixels to ensure we stay well under the limit
    max_pixels = int(max_pixels * 0.9)  # 10% safety margin
    
    if total_pixels <= max_pixels:
        return width, height
    
    # Calculate scaling factor to fit within limit
    scale = math.sqrt(max_pixels / total_pixels)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Double-check the new dimensions
    if new_width * new_height > max_pixels:
        # Apply additional scaling if needed
        scale *= 0.95
        new_width = int(width * scale)
        new_height = int(height * scale)
    
    return new_width, new_height

def capture_full_page_screenshot(driver, url: str) -> bytes:
    """Capture a full page screenshot by scrolling and stitching."""
    try:
        # Get initial dimensions
        total_height = driver.execute_script("return Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);")
        total_width = driver.execute_script("return Math.max(document.documentElement.scrollWidth, document.body.scrollWidth);")
        
        # Calculate viewport height
        viewport_height = driver.execute_script("return window.innerHeight;")
        
        # Pre-calculate final dimensions to ensure they're within limits
        MAX_PIXELS = 33177600 * 0.9  # 10% safety margin
        
        # If the page is very long, we'll split it into sections
        if total_height > 15000 or (total_width * total_height) > MAX_PIXELS:
            # Calculate maximum height that would fit within pixel limit
            max_safe_height = int(MAX_PIXELS / total_width)
            
            # Adjust section size based on max safe height
            section_height = min(viewport_height, max_safe_height // 4)  # Use quarter of max safe height per section
            
            sections = []
            offset = 0
            while offset < total_height:
                # Scroll to position
                driver.execute_script(f"window.scrollTo(0, {offset});")
                time.sleep(0.5)  # Wait for scroll and content to load
                
                # Capture viewport
                section_png = driver.get_screenshot_as_png()
                section = Image.open(io.BytesIO(section_png))
                
                # Ensure section is within limits
                if section.height > section_height:
                    section = section.crop((0, 0, section.width, section_height))
                
                sections.append(section)
                offset += section_height
            
            # Calculate final dimensions ensuring they're within limits
            final_width = min(total_width, 1920)  # Cap width at 1920px
            final_height = min(total_height, int(MAX_PIXELS / final_width))
            
            # Create new image with calculated dimensions
            final_image = Image.new('RGB', (final_width, final_height))
            y_offset = 0
            
            for section in sections:
                if y_offset + section.height > final_height:
                    # Crop section if it would exceed final height
                    remaining_height = final_height - y_offset
                    if remaining_height <= 0:
                        break
                    section = section.crop((0, 0, section.width, remaining_height))
                
                final_image.paste(section, (0, y_offset))
                y_offset += section.height
                if y_offset >= final_height:
                    break
            
            # Verify final size
            if final_image.width * final_image.height > MAX_PIXELS:
                # Resize if somehow still too large
                scale = math.sqrt(MAX_PIXELS / (final_image.width * final_image.height))
                new_width = int(final_image.width * scale)
                new_height = int(final_image.height * scale)
                final_image = final_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PNG
            output = io.BytesIO()
            final_image.save(output, format='PNG', optimize=True)
            return output.getvalue()
        else:
            # For shorter pages, still ensure we're within limits
            final_width, final_height = ensure_size_within_limits(total_width, total_height)
            driver.set_window_size(final_width, final_height)
            time.sleep(0.5)
            return driver.get_screenshot_as_png()
            
    except Exception as e:
        logger.error(f"Error in full page capture: {str(e)}")
        # Fallback to a safe capture
        safe_width, safe_height = ensure_size_within_limits(1920, 1080)
        driver.set_window_size(safe_width, safe_height)
        return driver.get_screenshot_as_png()

def test_selenium() -> bool:
    """Test if Selenium can run and capture a screenshot of a test page."""
    try:
        firefox_options = Options()
        firefox_options.add_argument('--headless')
        firefox_options.add_argument('--window-size=1920,1080')
        firefox_options.set_preference('layout.css.devPixelsPerPx', '2.0')  # Increased from 1.5
        
        driver = webdriver.Firefox(
            service=Service(GeckoDriverManager().install()),
            options=firefox_options
        )
        
        driver.set_page_load_timeout(20)
        driver.get("https://example.com")
        
        # Set zoom level for better text legibility
        driver.execute_script("document.body.style.zoom = '200%'")  # Increased from 150%
        
        # Ensure text is readable
        driver.execute_script("""
            document.querySelectorAll('*').forEach(function(el) {
                let style = window.getComputedStyle(el);
                if (parseInt(style.fontSize) < 16) {  // Increased minimum font size
                    el.style.fontSize = '16px';
                }
                // Improve contrast
                if (style.color && style.backgroundColor) {
                    let textColor = style.color;
                    let bgColor = style.backgroundColor;
                    if (textColor === bgColor || textColor === 'rgba(0, 0, 0, 0)') {
                        el.style.color = '#000000';
                    }
                }
            });
        """)
        
        # Additional wait for text scaling
        time.sleep(1)
        
        # Get page dimensions with padding for better quality
        total_height = driver.execute_script("return Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);")
        total_width = driver.execute_script("return Math.max(document.documentElement.scrollWidth, document.body.scrollWidth);")
        
        # Add padding and ensure minimum dimensions
        total_width = max(total_width, 1920)
        total_height = int(total_height * 1.1)
        
        # Ensure dimensions are within pixel limit
        final_width, final_height = ensure_size_within_limits(total_width, total_height)
        
        # Set window size with the adjusted dimensions
        driver.set_window_size(final_width, final_height)
        
        # Wait for any dynamic content to load
        time.sleep(1)
        
        # Capture full screenshot in memory with high quality
        screenshot_png = driver.get_screenshot_as_png()
        driver.quit()

        # Decode and verify image
        img = Image.open(io.BytesIO(screenshot_png))
        img.verify()
        logger.info(f"✅ Selenium is running and captured screenshot ({img.size[0]}x{img.size[1]} px)")
        return True
    except Exception as e:
        logger.error(f"❌ Selenium test failed: {str(e)}")
        return False

def test_ollama() -> bool:
    """Test if Ollama is running and accessible."""
    try:
        test_llm = ChatOllama(
            model="llama3.2:3b-instruct-q8_0",
            base_url="http://localhost:11434",
            temperature=0,
            num_gpu=1,
            num_thread=8
        )
        resp = test_llm([HumanMessage(content="Hello")])
        if isinstance(resp, AIMessage) and len(resp.content) > 0:
            logger.info("✅ Ollama is accessible")
            return True
        else:
            logger.error("❌ Ollama did not return a valid response")
            return False
    except Exception as e:
        logger.error(f"❌ Ollama test failed: {str(e)}")
        return False

def test_model_provider(provider: str) -> bool:
    """Test if the selected model provider is accessible."""
    try:
        if provider == ModelProvider.OLLAMA:
            return test_ollama()
        else:
            if not GROQ_API_KEY:
                logger.error("GROQ_API_KEY not set.")
                return False
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
            response.raise_for_status()
            logger.info("✅ Groq API is accessible")
            return True
    except Exception as e:
        logger.error(f"❌ {provider.capitalize()} test failed: {str(e)}")
        return False

def configure_llm(provider: str) -> Union[ChatOllama, ChatGroq]:
    """Configure LLM based on selected provider."""
    if provider == ModelProvider.OLLAMA:
        return ChatOllama(
            model="llama3.2:3b-instruct-q8_0",
            base_url="http://localhost:11434",
            temperature=0,
            num_gpu=1,
            num_thread=8
        )
    else:
        return ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
            groq_api_key=GROQ_API_KEY
        )

def configure_vision_model(provider: str) -> Union[ChatOllama, ChatGroq]:
    """Configure vision model based on selected provider."""
    if provider == ModelProvider.OLLAMA:
        return ChatOllama(
            model="llama3.2-vision:11b",
            base_url="http://localhost:11434",
            temperature=0,
            num_gpu=1,
            num_thread=8,
            madvise=True,
            f16=True
        )
    else:
        return ChatGroq(
            model="llama-3.2-90b-vision-preview",
            temperature=0,
            groq_api_key=GROQ_API_KEY
        )

def configure_llama():
    """Configure model and prompt based on user's choice."""
    provider = ModelProvider.get_provider_choice()
    if not test_model_provider(provider):
        logger.error(f"Failed to initialize {provider} models")
        sys.exit(1)
    llm = configure_llm(provider)
    prompt = PromptTemplate(
        template="""You are an assistant for research tasks. Use the following documents to provide a comprehensive and concise report on the topic. Ensure the report is self-contained with all necessary information.

        Topic: {topic}
        Documents: {documents}
        Report: """,
        input_variables=["topic", "documents"],
    )
    return llm, prompt, provider

def invoke_model(llm, prompt: str) -> AIMessage:
    """Helper to invoke LLM with a single prompt."""
    response = llm([HumanMessage(content=prompt)])
    return response

def safe_json_loads(json_str: str, fallback: Dict, content: str = "") -> Dict:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        fixed = json_str.replace('""', '"')
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            logger.error("Still can't parse JSON after fix.")
            return fallback

def view_image(base64_str: str, url: str):
    try:
        image_data = base64.b64decode(base64_str)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name
        
        if platform.system() == 'Windows':
            os.startfile(tmp_path)
        else:
            webbrowser.open('file://' + tmp_path)
            
        logger.info(f"Opened screenshot from {url}")
        return tmp_path
    except Exception as e:
        logger.error(f"Error viewing image: {str(e)}")
        return None

def generate_vision_query(llm, original_query: str) -> str:
    """Generate a focused vision query based on the original research question."""
    prompt = f"""Given a research question, create a natural and focused query for a vision model to extract information from a webpage screenshot.
    The query should be direct and specific, but sound natural. Always start with "Describe the image in detail, focusing on".
    Avoid using quotes or mechanical phrases like "extract X from the image."
    Keep it under 15 words and focused on the key information needed.
    
    Examples:
    Research question: What is the current Tesla stock price?
    Vision query: Describe the image in detail, focusing on the specific Tesla stock price.
    
    Research question: What are the iPhone 15 specs?
    Vision query: Describe the image in detail, focusing on the iPhone 15 specifications and features.
    
    Research question: {original_query}
    
    Vision query:"""
    
    try:
        response = invoke_model(llm, prompt)
        vision_query = response.content.strip()
        
        # Clean up and standardize the query
        vision_query = vision_query.replace('"', '').replace("'", '')
        if not vision_query.lower().startswith("describe the image"):
            vision_query = f"Describe the image in detail, focusing on {vision_query}"
        
        # Remove mechanical phrases
        vision_query = vision_query.replace("extract from the image", "")
        vision_query = vision_query.replace("from the image", "")
        vision_query = re.sub(r'\s+', ' ', vision_query).strip()
        
        # Ensure it ends properly
        if vision_query.endswith("focusing on"):
            vision_query = vision_query[:-11].strip()
        
        return vision_query
    except Exception as e:
        logger.error(f"Error generating vision query: {str(e)}")
        return "Describe the image in detail, focusing on the main content and key information."

def fetch_webpage_content(url: str, provider: str, original_query: str) -> str:
    """Fetch webpage content by capturing a screenshot via Selenium and processing it with a vision model."""
    if host_tracker.is_problematic_host(url):
        logger.info(f"Skipping known problematic host: {urlparse(url).netloc}")
        return f"Skipped: Known problematic host"

    try:
        # Set up Selenium (headless Firefox)
        firefox_options = Options()
        firefox_options.add_argument('--headless')
        firefox_options.add_argument('--window-size=1920,1080')
        firefox_options.add_argument('--disable-blink-features=AutomationControlled')  # Hide automation
        firefox_options.add_argument('--disable-notifications')
        firefox_options.set_preference('layout.css.devPixelsPerPx', '2.0')  # Increased from 1.25
        
        # Add headers to appear more like a real browser
        firefox_options.set_preference('general.useragent.override', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36')
        
        driver = webdriver.Firefox(
            service=Service(GeckoDriverManager().install()),
            options=firefox_options
        )
        driver.set_page_load_timeout(60)
        
        # Set cookies and localStorage to bypass some anti-bot measures
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Attempt to load the page
        driver.get(url)
        time.sleep(2)  # Give time for dynamic content to load
        
        # Check for and handle CAPTCHA/cookie popups
        try:
            driver.execute_script("""
                // Remove common overlay elements
                document.querySelectorAll('[class*="cookie"], [class*="popup"], [class*="modal"], [id*="cookie"], [id*="popup"], [id*="modal"]')
                    .forEach(el => el.remove());
                // Remove fixed position elements that might overlay content
                document.querySelectorAll('*').forEach(el => {
                    const style = window.getComputedStyle(el);
                    if (style.position === 'fixed' || style.position === 'sticky') {
                        el.remove();
                    }
                });
            """)
        except Exception as e:
            logger.warning(f"Error handling overlays: {str(e)}")
        
        # Set text size and ensure readability with special handling for financial data
        driver.execute_script("""
            // Set base zoom
            document.body.style.zoom = '200%';  // Increased from 125%
            
            // Function to check if text might be financial data
            function isFinancialData(text) {
                return /\\$|\\d+\\.\\d+|\\d+%|price|stock|market|share/i.test(text);
            }
            
            // Ensure text is readable with special handling for financial data
            document.querySelectorAll('*').forEach(function(el) {
                let style = window.getComputedStyle(el);
                let text = el.textContent || '';
                
                // Special handling for financial data
                if (isFinancialData(text)) {
                    el.style.fontSize = '24px';  // Larger size for financial data
                    el.style.fontWeight = 'bold';
                    el.style.color = '#000000';  // Ensure high contrast
                } else if (parseInt(style.fontSize) < 16) {  // Increased minimum font size
                    el.style.fontSize = '16px';
                }
                
                // Improve contrast
                if (style.color && style.backgroundColor) {
                    let textColor = style.color;
                    let bgColor = style.backgroundColor;
                    if (textColor === bgColor || textColor === 'rgba(0, 0, 0, 0)' || 
                        textColor === 'rgb(255, 255, 255)' || textColor === '#ffffff') {
                        el.style.color = '#000000';
                    }
                }
                
                // Improve visibility of links
                if (el.tagName.toLowerCase() === 'a') {
                    el.style.textDecoration = 'underline';
                }
            });
            
            // Additional handling for table cells (common in financial data)
            document.querySelectorAll('td, th').forEach(function(el) {
                let text = el.textContent || '';
                if (isFinancialData(text)) {
                    el.style.padding = '10px';
                    el.style.fontSize = '24px';
                    el.style.fontWeight = 'bold';
                }
            });
        """)
        
        # Additional wait for text adjustments
        time.sleep(2)  # Increased wait time
        
        # Get dimensions and ensure they're within limits
        total_height = driver.execute_script("return Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);")
        total_width = driver.execute_script("return Math.max(document.documentElement.scrollWidth, document.body.scrollWidth);")
        
        final_width, final_height = ensure_size_within_limits(total_width, total_height)
        logger.info(f"Adjusted dimensions to {final_width}x{final_height} to stay within pixel limit")
        
        # Set final window size
        driver.set_window_size(final_width, final_height)
        time.sleep(1)
        
        # Capture the screenshot using our improved method
        screenshot_png = capture_full_page_screenshot(driver, url)
        driver.quit()

        # Process the image
        img = Image.open(io.BytesIO(screenshot_png))
        
        # Convert to RGB and enhance readability
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Enhance image quality with specified values
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.25)  # Modified sharpness value
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.25)  # Modified contrast value
        
        # Save with high quality
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=100, optimize=True)  # Maximum quality
        screenshot_data = output.getvalue()
        
        # Convert to base64
        base64_image = base64.b64encode(screenshot_data).decode('utf-8')
        
        vision_llm = configure_vision_model(provider)
        text_llm = configure_llm(provider)
        
        vision_query = generate_vision_query(text_llm, original_query)
        logger.info(f"Using vision query: {vision_query}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        logger.info(f"Processing screenshot from {url} with vision model ({provider})")
        vision_response = vision_llm.invoke(messages)
        
        extracted_text = vision_response.content.strip()
        
        print("\n" + "="*80)
        print(f"Vision Model Description for {url}:")
        print("-"*80)
        print(extracted_text)
        print("="*80 + "\n")
        
        logger.info(f"Successfully processed content from {url}")
        return extracted_text
        
    except Exception as e:
        host_tracker.add_failed_host(url)
        logger.error(f"Error processing {url}: {str(e)}")
        return f"Error processing {url}: {str(e)}"

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    return text_splitter.split_documents(docs)

def create_vectorstore(docs_splits):
    embeddings = OllamaEmbeddings(
        model="all-minilm",
        base_url="http://localhost:11434/v1"
    )
    vectorstore = SKLearnVectorStore.from_documents(
        documents=docs_splits,
        embedding=embeddings
    )
    return vectorstore.as_retriever(k=4)

def decompose_topic_into_subtopics(llm, topic):
    decomposition_prompt = f"""You are a research assistant.
You will be given a research topic. If the topic is broad or complex, break it down into a list of more specific subtopics or sub-questions that would help in researching it thoroughly.
If the topic is simple or already focused, just return it as is.
Format your response as a simple list with one subtopic per line.

Topic: {topic}

Subtopics:"""
    response = invoke_model(llm, decomposition_prompt)
    response_text = response.content
    subtopics = [line.strip("- ").strip() for line in response_text.split("\n") if line.strip()]
    subtopics = [s for s in subtopics if s and not s.lower().startswith(("subtopic", "topic"))]
    return subtopics if subtopics else [topic]

def extract_urls_from_search_results(search_text: str) -> List[str]:
    urls = re.findall(r'(https?://[^\s\'"]+)', search_text)
    valid_urls = []
    for url in urls:
        url = re.sub(r'[.,)\]]+$', '', url)
        if url.startswith(('http://', 'https://')):
            if not host_tracker.is_problematic_host(url):
                valid_urls.append(url)
            else:
                logger.info(f"Filtered out problematic host: {urlparse(url).netloc}")
    return list(set(valid_urls))

@dataclass
class SourceReliability:
    domain: str
    query_types: Dict[str, float]
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    total_attempts: int
    successful_attempts: int
    average_response_time: float
    notes: List[str]

class ResearchMemory:
    def __init__(self, memory_file="research_memory.json"):
        self.memory_file = memory_file
        self.source_reliability = {}
        self.query_patterns = {}
        self.feedback_history = {}
        self.load_memory()
    
    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    
                    for domain, info in data.get('sources', {}).items():
                        self.source_reliability[domain] = SourceReliability(
                            domain=domain,
                            query_types=info.get('query_types', {}),
                            last_success=datetime.fromisoformat(info['last_success']) if info.get('last_success') else None,
                            last_failure=datetime.fromisoformat(info['last_failure']) if info.get('last_failure') else None,
                            total_attempts=info.get('total_attempts', 0),
                            successful_attempts=info.get('successful_attempts', 0),
                            average_response_time=info.get('average_response_time', 0.0),
                            notes=info.get('notes', [])
                        )
                    
                    self.query_patterns = data.get('query_patterns', {})
                    self.feedback_history = data.get('feedback_history', {})
                    
                logger.info(f"Loaded research memory with {len(self.source_reliability)} sources and {len(self.feedback_history)} feedback entries")
        except Exception as e:
            logger.error(f"Error loading research memory: {str(e)}")
            self.source_reliability = {}
            self.query_patterns = {}
            self.feedback_history = {}
    
    def save_memory(self):
        try:
            data = {
                'sources': {
                    domain: {
                        'query_types': info.query_types,
                        'last_success': info.last_success.isoformat() if info.last_success else None,
                        'last_failure': info.last_failure.isoformat() if info.last_failure else None,
                        'total_attempts': info.total_attempts,
                        'successful_attempts': info.successful_attempts,
                        'average_response_time': info.average_response_time,
                        'notes': info.notes
                    }
                    for domain, info in self.source_reliability.items()
                },
                'query_patterns': self.query_patterns,
                'feedback_history': self.feedback_history
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info("Successfully saved research memory")
        except Exception as e:
            logger.error(f"Error saving research memory: {str(e)}")
    
    def categorize_query(self, query: str) -> str:
        categories = {
            'stock_price': r'(?i)(stock|share)\s+price|price\s+of\s+stock',
            'financial_data': r'(?i)financial|revenue|earnings|profit|market\s+cap',
            'company_info': r'(?i)headquarters|ceo|founded|employees|about',
            'news': r'(?i)news|latest|recent|update|announce',
            'technical': r'(?i)technology|software|product|service|api',
            'general': r'.*'
        }
        
        for category, pattern in categories.items():
            if re.search(pattern, query):
                return category
        return 'general'
    
    def update_source_reliability(self, domain: str, query_type: str, success: bool, response_time: float, content_quality: float):
        if domain not in self.source_reliability:
            self.source_reliability[domain] = SourceReliability(
                domain=domain,
                query_types={},
                last_success=None,
                last_failure=None,
                total_attempts=0,
                successful_attempts=0,
                average_response_time=0.0,
                notes=[]
            )
        
        source = self.source_reliability[domain]
        current_time = datetime.now(timezone.utc)
        
        if query_type not in source.query_types:
            source.query_types[query_type] = 0.0
        
        source.total_attempts += 1
        if success:
            source.successful_attempts += 1
            source.last_success = current_time
            source.query_types[query_type] = (
                source.query_types[query_type] * 0.9 +
                content_quality * 0.1
            )
        else:
            source.last_failure = current_time
            source.query_types[query_type] *= 0.9
        
        source.average_response_time = (
            source.average_response_time * 0.9 +
            response_time * 0.1
        )
        
        self.save_memory()
    
    def get_best_sources(self, query_type: str, min_reliability: float = 0.3) -> List[str]:
        relevant_sources = []
        
        for domain, source in self.source_reliability.items():
            reliability = source.query_types.get(query_type, 0.0)
            if reliability >= min_reliability:
                relevant_sources.append((domain, reliability))
        
        relevant_sources.sort(key=lambda x: x[1], reverse=True)
        return [domain for domain, _ in relevant_sources]
    
    def prioritize_urls(self, urls: List[str], query: str) -> List[str]:
        query_type = self.categorize_query(query)
        self.get_best_sources(query_type)
        
        scored_urls = []
        for url in urls:
            domain = urlparse(url).netloc
            source = self.source_reliability.get(domain)
            
            if source:
                reliability = source.query_types.get(query_type, 0.0)
                success_rate = source.successful_attempts / max(1, source.total_attempts)
                response_speed = 1.0 / (1.0 + source.average_response_time)
                score = (reliability * 0.5 +
                        success_rate * 0.3 +
                        response_speed * 0.2)
            else:
                score = 0.1
            
            scored_urls.append((url, score))
        
        scored_urls.sort(key=lambda x: x[1], reverse=True)
        return [url for url, _ in scored_urls]
    
    def record_feedback(self, topic: str, sources: List[str], agent_assessment: Dict, human_feedback: bool, notes: str = None):
        current_time = datetime.now(timezone.utc)
        query_type = self.categorize_query(topic)
        
        feedback_entry = {
            'timestamp': current_time.isoformat(),
            'topic': topic,
            'sources': sources,
            'agent_assessment': agent_assessment,
            'human_feedback': human_feedback,
            'query_type': query_type,
            'notes': notes
        }
        
        if topic not in self.feedback_history:
            self.feedback_history[topic] = []
        self.feedback_history[topic].append(feedback_entry)
        
        agent_confidence = agent_assessment.get('confidence', 0.0)
        agent_correct = agent_assessment.get('is_accurate', False)
        
        for source in sources:
            domain = urlparse(source).netloc
            if domain not in self.source_reliability:
                continue
                
            source_info = self.source_reliability[domain]
            
            if human_feedback:
                if agent_correct == human_feedback:
                    self._update_source_confidence(domain, query_type, True, 1.0)
                    source_info.notes.append(f"[{current_time.isoformat()}] Accurate assessment confirmed by human feedback")
                else:
                    self._update_source_confidence(domain, query_type, False, 1.0)
                    source_info.notes.append(f"[{current_time.isoformat()}] Assessment contradicted by human feedback")
            else:
                self._update_source_confidence(domain, query_type, agent_correct, agent_confidence)
        
        self.save_memory()
        
    def _update_source_confidence(self, domain: str, query_type: str, success: bool, confidence: float):
        source = self.source_reliability[domain]
        
        if query_type not in source.query_types:
            source.query_types[query_type] = 0.0
            
        current_reliability = source.query_types[query_type]
        
        if success:
            new_reliability = current_reliability + (1 - current_reliability) * confidence * 0.1
        else:
            new_reliability = current_reliability * 0.8
            
        source.query_types[query_type] = max(0.0, min(1.0, new_reliability))
    
    def get_feedback_stats(self, domain: str = None, query_type: str = None) -> Dict:
        stats = {
            'total_entries': 0,
            'agent_accuracy': 0.0,
            'human_agreement': 0.0,
            'query_type_performance': {},
            'recent_trends': []
        }
        
        relevant_entries = []
        
        for topic_entries in self.feedback_history.values():
            for entry in topic_entries:
                if domain and not any(domain in s for s in entry['sources']):
                    continue
                if query_type and entry['query_type'] != query_type:
                    continue
                relevant_entries.append(entry)
        
        if not relevant_entries:
            return stats
            
        stats['total_entries'] = len(relevant_entries)
        
        correct_assessments = sum(1 for e in relevant_entries 
                                if e['agent_assessment'].get('is_accurate') == e['human_feedback'])
        human_agreements = sum(1 for e in relevant_entries if e['human_feedback'])
        
        stats['agent_accuracy'] = correct_assessments / len(relevant_entries)
        stats['human_agreement'] = human_agreements / len(relevant_entries)
        
        query_types = {}
        for entry in relevant_entries:
            qt = entry['query_type']
            if qt not in query_types:
                query_types[qt] = {'total': 0, 'successful': 0}
            query_types[qt]['total'] += 1
            if entry['human_feedback']:
                query_types[qt]['successful'] += 1
        
        stats['query_type_performance'] = {
            qt: {'success_rate': data['successful'] / data['total']}
            for qt, data in query_types.items()
        }
        
        recent = relevant_entries[-10:]
        stats['recent_trends'] = [
            {
                'timestamp': e['timestamp'],
                'query_type': e['query_type'],
                'success': e['human_feedback']
            }
            for e in recent
        ]
        
        return stats

class ResearchAgent:
    def __init__(self, retriever, llm, prompt, brave_search, wikipedia, provider):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.brave_search = brave_search
        self.wikipedia = wikipedia
        self.provider = provider
        self.max_retries = 3
        self.retry_delay = 2
        self.research_memory = {}
        self.confidence_threshold = 0.5
        self.host_tracker = host_tracker
        self.current_topic = None
        self.memory = ResearchMemory()
        self.current_assessment = None

    def assess_content_relevance(self, content: str, topic: str) -> Dict:
        assessment_prompt = f"""You are a content assessment expert. Analyze this content's relevance and completeness for the given topic.
        Consider:
        1. How directly it answers the topic/question
        2. The specificity and accuracy of information
        3. Whether it provides context and supporting details
        4. The currentness and reliability of the information
        
        Topic: {topic}
        Content length: {len(content)} characters
        First 1000 chars: {content[:1000]}
        
        You must respond with ONLY a JSON object in this exact format:
        {{
            "relevance": <number between 0-1>,
            "is_complete": <true or false>,
            "found_data": "<key information found>",
            "needs_verification": <true or false>,
            "needs_context": <true or false>,
            "confidence": <number between 0-1>
        }}"""
        
        try:
            response = invoke_model(self.llm, assessment_prompt)
            response_text = response.content.strip()
            json_match = re.search(r'\{[\s\S]*?\}', response_text)
            if not json_match:
                return {
                    'relevance': 0.5 if len(content) > 100 else 0.0,
                    'is_complete': False,
                    'found_data': content[:200] if len(content) > 0 else '',
                    'needs_verification': True,
                    'needs_context': True,
                    'confidence': 0.3
                }
            json_str = json_match.group(0)
            fallback = {
                'relevance': 0.5 if len(content) > 100 else 0.0,
                'is_complete': False,
                'found_data': content[:200] if len(content) > 0 else '',
                'needs_verification': True,
                'needs_context': True,
                'confidence': 0.3
            }
            result = safe_json_loads(json_str, fallback, content)
            return {
                'relevance': float(result.get('relevance', 0)),
                'is_complete': bool(result.get('is_complete', False)),
                'found_data': str(result.get('found_data', '')),
                'needs_verification': bool(result.get('needs_verification', True)),
                'needs_context': bool(result.get('needs_context', True)),
                'confidence': float(result.get('confidence', 0))
            }
        except Exception as e:
            logger.error(f"Error in content assessment: {str(e)}")
            return {
                'relevance': 0.0,
                'is_complete': False,
                'found_data': '',
                'needs_verification': True,
                'needs_context': True,
                'confidence': 0.0
            }

    def extract_key_information(self, content: str, topic: str) -> Dict:
        extraction_prompt = f"""You are a precise information extractor. Extract key information from the content that is relevant to the topic.
        You must respond in valid JSON format with exactly these fields:
        {{
            "main_facts": [list of key facts as strings],
            "confidence": number between 0.0-1.0,
            "timestamp": string or null,
            "source_quality": number between 0.0-1.0
        }}

        Topic: {topic}
        Content: {content}

        Respond ONLY with the JSON object, no other text:"""

        try:
            response = invoke_model(self.llm, extraction_prompt)
            response_text = response.content.strip()
            json_match = re.search(r'\{[\s\S]*?\}', response_text)
            if not json_match:
                return {
                    "main_facts": ["Unable to extract structured information from source"],
                    "confidence": 0.0,
                    "timestamp": None,
                    "source_quality": 0.0
                }
            json_str = json_match.group(0)
            fallback = {
                "main_facts": ["Unable to extract structured information from source"],
                "confidence": 0.0,
                "timestamp": None,
                "source_quality": 0.0
            }
            info = safe_json_loads(json_str, fallback, content)
            
            if not isinstance(info.get('main_facts', []), list):
                info['main_facts'] = [str(info.get('main_facts', ''))]
            
            return {
                'main_facts': info.get('main_facts', []),
                'confidence': min(max(info.get('confidence', 0.0), 0.0), 1.0),
                'timestamp': info.get('timestamp'),
                'source_quality': min(max(info.get('source_quality', 0.0), 0.0), 1.0)
            }
        except Exception as e:
            logger.error(f"Error extracting information: {str(e)}")
            return {
                "main_facts": ["Unable to extract structured information from source"],
                "confidence": 0.0,
                "timestamp": None,
                "source_quality": 0.0
            }

    def assess_question_complexity(self, topic: str) -> float:
        complexity_prompt = f"""
        Analyze the complexity of this research topic/question.
        Rate from 0.0 to 1.0, where:
        - 0.0: Very simple
        - 0.3: Basic fact-finding
        - 0.6: Moderate complexity
        - 1.0: Complex analysis
        
        Topic: {topic}
        
        Respond with only a number between 0.0 and 1.0:"""
        
        try:
            response = invoke_model(self.llm, complexity_prompt)
            matches = re.findall(r"0?\.[0-9]+", response.content)
            if matches:
                rating = float(matches[0])
            else:
                rating = 0.5
            return min(max(rating, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Error assessing question complexity: {str(e)}")
            return 0.5

    def _check_information_consistency(self, facts: List[str]) -> bool:
        return True

    def should_continue_research(self, topic: str, current_source: Dict) -> Dict:
        if topic not in self.research_memory:
            return {"continue": True, "reason": "No research started yet"}

        findings = self.research_memory[topic]
        sources_count = len(findings['sources'])
        complexity = self.assess_question_complexity(topic)
        
        min_sources = max(2, int(complexity * 5))
        quality_threshold = 0.7 + (complexity * 0.2)
        high_quality_sources = sum(1 for s in findings['sources'] 
                                   if s.get('relevance', 0) > quality_threshold 
                                   and s.get('confidence', 0) > quality_threshold)
        
        if high_quality_sources >= min_sources:
            return {"continue": False, "reason": "Sufficient high-quality sources found"}
        
        max_sources = min_sources * 2
        if sources_count >= max_sources:
            return {"continue": False, "reason": "Maximum sources reached"}
        
        if sources_count > 1:
            info_consistent = self._check_information_consistency(findings['main_facts'])
            if not info_consistent:
                return {"continue": True, "reason": "Found inconsistent information", "priority": "verification"}
        
        if sources_count > 0:
            latest_source = findings['sources'][-1]
            if latest_source.get('needs_verification', True):
                return {"continue": True, "reason": "Need verification", "priority": "verification"}
            if latest_source.get('needs_context', True) and complexity > 0.5:
                return {"continue": True, "reason": "Need context", "priority": "context"}
        
        return {"continue": True, "reason": "Need more information"}

    def brave_search_run(self, query: str, retries: int = 3) -> str:
        if not BRAVE_API_KEY:
            logger.error("Brave Search API key not set. Unable to perform search.")
            return ""
        for i in range(retries):
            try:
                return self.brave_search.run(query)
            except requests.HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning("Hit rate limit. Waiting before retry...")
                    time.sleep((i+1)*2)
                else:
                    logger.error(f"HTTP Error during Brave search: {str(e)}")
                    time.sleep(2)
            except Exception as ex:
                logger.error(f"Error in Brave search: {str(ex)}")
                time.sleep(2)
        return ""

    def fetch_additional_info(self, topic: str) -> str:
        self.current_topic = topic
        query_type = self.memory.categorize_query(topic)
        
        if topic not in self.research_memory:
            self.research_memory[topic] = {
                'sources': [],
                'main_facts': [],
                'last_update': time.time(),
                'visited_urls': set()
            }

        all_research = []
        research_status = {"continue": True, "reason": "Initial research"}
        
        if query_type == 'stock_price':
            priority_domains = [
                'marketwatch.com',
                'finance.yahoo.com',
                'bloomberg.com',
                'reuters.com'
            ]
            
            for domain in priority_domains:
                if any(domain in s.get('url', '') for s in self.research_memory[topic]['sources']):
                    continue
                    
                search_query = f"site:{domain} {topic}"
                search_results = self.brave_search_run(search_query)
                urls = extract_urls_from_search_results(search_results)
                
                if urls:
                    url = urls[0]
                    if url not in self.research_memory[topic]['visited_urls']:
                        self.research_memory[topic]['visited_urls'].add(url)
                        content = fetch_webpage_content(url, self.provider, topic)
                        
                        assessment = self.assess_content_relevance(content, topic)
                        if assessment['relevance'] > 0.7:
                            info = self.extract_key_information(content, topic)
                            current_source = {**assessment, **info}
                            
                            self.research_memory[topic]['sources'].append({
                                'url': url,
                                'content': content,
                                **current_source
                            })
                            self.research_memory[topic]['main_facts'].extend(info['main_facts'])
                            
                            if assessment['relevance'] > 0.8 and assessment['confidence'] > 0.8:
                                research_status = {"continue": False, "reason": "Found reliable stock price"}
                                break

        search_attempts = 0
        max_search_attempts = 3
        
        while research_status["continue"] and search_attempts < max_search_attempts:
            try:
                if search_attempts == 0:
                    search_query = topic
                elif search_attempts == 1:
                    search_query = f"{topic} latest information"
                else:
                    search_query = f"{topic} current data {datetime.now().strftime('%Y')}"

                if research_status.get("priority") == "verification":
                    search_query += " facts verify source"
                elif research_status.get("priority") == "context":
                    search_query += " background context"
                
                logger.info(f"Searching with query: {search_query}")
                results = self.brave_search_run(search_query)
                urls = extract_urls_from_search_results(results)
                
                urls = [url for url in urls if url not in self.research_memory[topic]['visited_urls']]
                
                if not urls:
                    search_attempts += 1
                    continue
                
                urls = self.memory.prioritize_urls(urls, topic)
                
                for url in urls[:2]:
                    if url in self.research_memory[topic]['visited_urls']:
                        continue
                        
                    self.research_memory[topic]['visited_urls'].add(url)
                    start_time = time.time()
                    content = fetch_webpage_content(url, self.provider, topic)
                    response_time = time.time() - start_time
                    
                    assessment = self.assess_content_relevance(content, topic)
                    domain = urlparse(url).netloc
                    
                    success = assessment['relevance'] > 0.5
                    self.memory.update_source_reliability(
                        domain=domain,
                        query_type=query_type,
                        success=success,
                        response_time=response_time,
                        content_quality=assessment['relevance']
                    )
                    
                    if success:
                        info = self.extract_key_information(content, topic)
                        current_source = {**assessment, **info}
                        
                        self.research_memory[topic]['sources'].append({
                            'url': url,
                            'content': content,
                            **current_source
                        })
                        self.research_memory[topic]['main_facts'].extend(info['main_facts'])
                        
                        research_status = self.should_continue_research(topic, current_source)
                        logger.info(f"Research status: {research_status['reason']}")
                        
                        if not research_status["continue"]:
                            break
                
                if not research_status["continue"]:
                    break
                    
                search_attempts += 1
                
            except Exception as e:
                logger.error(f"Error in research iteration: {str(e)}")
                search_attempts += 1

        all_research.append(f"""
        === Research Summary ===
        Query Type: {query_type}
        Total Sources: {len(self.research_memory[topic]['sources'])}
        Key Facts Found: {json.dumps(self.research_memory[topic]['main_facts'], indent=2)}
        Sources: {json.dumps([{
            'url': s['url'],
            'relevance': s.get('relevance', 0),
            'confidence': s.get('confidence', 0),
            'found_data': s.get('found_data', '')
        } for s in self.research_memory[topic]['sources']], indent=2)}
        """)

        return "\n\n".join(all_research)

    def generate_report(self, topic: str) -> str:
        additional_info = self.fetch_additional_info(topic)
        
        enhanced_prompt = f"""
        Generate a comprehensive report based on the research findings.
        Focus on the most relevant and current information.
        
        Topic: {topic}
        Research Findings: {additional_info}
        
        Guidelines:
        1. Prioritize information from high-quality sources
        2. Include specific, factual information
        3. Note any significant gaps or uncertainties
        4. Cite sources where appropriate
        
        Report:"""
        
        for attempt in range(self.max_retries):
            try:
                report = invoke_model(self.llm, enhanced_prompt)
                return report.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"Error generating report: {str(e)}"
                time.sleep(self.retry_delay)

    def assess_research_accuracy(self, topic: str, research_data: Dict) -> Dict:
        assessment_prompt = f"""Analyze the research results for accuracy and completeness.
        Consider:
        1. Consistency across sources
        2. Data freshness and relevance
        3. Source reliability
        4. Information completeness
        
        Topic: {topic}
        Research Data: {json.dumps(research_data, indent=2)}
        
        Respond with JSON:
        {{
            "is_accurate": boolean,
            "confidence": float,
            "completeness": float,
            "concerns": [string],
            "verification_needed": boolean
        }}"""
        
        try:
            response = invoke_model(self.llm, assessment_prompt)
            json_match = re.search(r'\{[\s\S]*?\}', response.content)
            if json_match:
                assessment = safe_json_loads(json_match.group(0), {
                    "is_accurate": False,
                    "confidence": 0.0,
                    "completeness": 0.0,
                    "concerns": ["Assessment failed"],
                    "verification_needed": True
                })
            else:
                assessment = {
                    "is_accurate": False,
                    "confidence": 0.0,
                    "completeness": 0.0,
                    "concerns": ["No JSON returned"],
                    "verification_needed": True
                }
            self.current_assessment = assessment
            return assessment
        except Exception as e:
            logger.error(f"Error in research assessment: {str(e)}")
            return {
                "is_accurate": False,
                "confidence": 0.0,
                "completeness": 0.0,
                "concerns": ["Assessment failed"],
                "verification_needed": True
            }
    
    def record_human_feedback(self, topic: str, is_accurate: bool, notes: str = None):
        if not self.current_assessment:
            logger.error("No current research assessment available")
            return
            
        sources = [s['url'] for s in self.research_memory.get(topic, {}).get('sources', [])]
        self.memory.record_feedback(
            topic=topic,
            sources=sources,
            agent_assessment=self.current_assessment,
            human_feedback=is_accurate,
            notes=notes
        )
        
        self.current_assessment = None

def main():
    if not test_selenium():
        logger.error("Selenium service check failed")
        sys.exit(1)
    
    provider = input("Choose model provider (ollama/groq): ").lower().strip()
    if provider not in [ModelProvider.OLLAMA, ModelProvider.GROQ]:
        logger.error("Invalid provider choice")
        sys.exit(1)
    
    if not test_model_provider(provider):
        logger.error(f"Model provider {provider} check failed")
        sys.exit(1)
    
    llm, prompt, provider = configure_llama()
    
    if not BRAVE_API_KEY:
        logger.warning("Brave Search API key not set. Searches will not return results.")
    brave_search = BraveSearch.from_api_key(
        api_key=BRAVE_API_KEY,
        search_kwargs={"count": 5}
    )
    
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    retriever = None
    
    agent = ResearchAgent(retriever, llm, prompt, brave_search, wikipedia, provider)
    
    while True:
        try:
            topic = input("\nEnter research topic (or 'quit' to exit): ").strip()
            if topic.lower() == 'quit':
                break
                
            if not topic:
                print("Please enter a valid topic")
                continue
            
            logger.info(f"Starting research for topic: {topic}")
            report = agent.generate_report(topic)
            print("\nFinal Report:")
            print("=" * 80)
            print(report)
            print("=" * 80)
            
            feedback = input("\nWas this information accurate? (y/n): ").lower().strip()
            if feedback in ['y', 'n']:
                is_accurate = feedback == 'y'
                notes = input("Any additional notes? (Enter to skip): ").strip()
                agent.record_human_feedback(topic, is_accurate, notes if notes else None)
                print("Thank you for your feedback!")
            
        except KeyboardInterrupt:
            print("\nResearch interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error during research: {str(e)}")
            print("An error occurred. Please try again.")
            continue

if __name__ == "__main__":
    main()
