#!/usr/bin/env python3
"""
BSP Speech Scraper - Final Production Version  
Scrapes ALL speeches by trying all possible ItemIds (1-800) and filtering by date
"""

import csv
import time
import re
import json
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper_log.txt'),
        logging.StreamHandler()
    ]
)

class BSPSpeechScraperFinal:
    def __init__(self, output_file='bsp_speeches.csv'):
        self.base_url = "https://www.bsp.gov.ph"
        self.output_file = output_file
        self.progress_file = 'scraper_progress.json'
        
        self.speeches = []
        self.scraped_ids = set()
        self.start_date = datetime(2017, 7, 3)
        self.end_date = datetime(2026, 2, 3)
        
        # Load progress if exists
        self.load_progress()
        
    def load_progress(self):
        """Load previously scraped IDs to allow resuming"""
        if Path(self.progress_file).exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.scraped_ids = set(data.get('scraped_ids', []))
                    # Also load already found speeches
                    if Path(self.output_file).exists():
                        df = pd.read_csv(self.output_file)
                        self.speeches = df.to_dict('records')
                logging.info(f"Loaded progress: {len(self.scraped_ids)} IDs processed, {len(self.speeches)} speeches saved")
            except Exception as e:
                logging.warning(f"Could not load progress: {e}")
    
    def save_progress(self):
        """Save progress to allow resuming"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'scraped_ids': list(self.scraped_ids),
                    'last_updated': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logging.warning(f"Could not save progress: {e}")
    
    def parse_date(self, date_str):
        """Parse date string to datetime object"""
        if not date_str:
            return None
        
        date_str = date_str.strip()
        formats = [
            '%B %d, %Y', '%b %d, %Y', '%Y-%m-%d', '%m/%d/%Y',
            '%d %B %Y', '%d %b %Y', '%B %d %Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ''
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\r', ' ').replace('\n', ' ')
        return text.strip()
    
    def scrape_speech(self, page, item_id):
        """Scrape individual speech page using Playwright"""
        url = f"{self.base_url}/SitePages/MediaAndResearch/SpeechesDisp.aspx?ItemId={item_id}"
        
        try:
            # Initialize speech data
            speech_data = {
                'Title': '',
                'Date': '',
                'Place': '',
                'Occasion': '',
                'Speaker': '',
                'Content': '',
                'URL': url,
                'ItemId': str(item_id)
            }
            
            # Navigate to the page
            page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for Angular to render
            time.sleep(2)
            
            # Extract Title
            try:
                title = page.locator('h3.ng-binding').first.inner_text(timeout=5000)
                speech_data['Title'] = self.clean_text(title)
            except:
                # No title found - page doesn't exist or is invalid
                return None
            
            # Skip if title is just "Legal" or empty
            if not speech_data['Title'] or speech_data['Title'] in ['Legal', 'Quick Links']:
                return None
            
            # Extract metadata from divs with ng-bind-html
            try:
                divs = page.locator('div[ng-bind-html]').all()
                
                for div in divs:
                    try:
                        text = div.inner_text()
                        ng_bind = div.get_attribute('ng-bind-html') or ""
                        
                        if 'Date' in ng_bind:
                            speech_data['Date'] = self.clean_text(text.replace('Date:', ''))
                        elif 'Place' in ng_bind:
                            speech_data['Place'] = self.clean_text(text.replace('Place:', ''))
                        elif 'Occasion' in ng_bind:
                            speech_data['Occasion'] = self.clean_text(text.replace('Occasion:', ''))
                        elif 'Speaker' in ng_bind:
                            speech_data['Speaker'] = self.clean_text(text.replace('Speaker:', ''))
                    except:
                        pass
            except:
                pass
            
            # Extract Content
            try:
                content_elem = page.locator('span[ng-bind-html*="Transcription"]').first
                content = content_elem.inner_text(timeout=5000)
                speech_data['Content'] = self.clean_text(content)[:10000]
            except:
                # Fallback: get body text
                try:
                    all_text = page.inner_text('body')
                    if speech_data['Speaker']:
                        start_idx = all_text.find(speech_data['Speaker']) + len(speech_data['Speaker'])
                        content = all_text[start_idx:start_idx+10000]
                        content = re.sub(r'Share.*?Download PDF.*?$', '', content, flags=re.DOTALL)
                        speech_data['Content'] = self.clean_text(content)
                except:
                    pass
            
            # Filter by date range
            if speech_data['Date']:
                speech_date = self.parse_date(speech_data['Date'])
                if speech_date:
                    if speech_date < self.start_date or speech_date > self.end_date:
                        logging.debug(f"ItemId={item_id} - Date {speech_date.strftime('%Y-%m-%d')} outside range")
                        return None
            
            return speech_data
            
        except PlaywrightTimeout:
            return None
        except Exception as e:
            logging.debug(f"ItemId={item_id} - Error: {e}")
            return None
    
    def run(self, start_id=1, end_id=1114):
        """Run the scraper by trying all ItemIds in range"""
        logging.info("=" * 70)
        logging.info("BSP Speech Scraper - Final Version")
        logging.info(f"Scraping ItemIds {start_id} to {end_id}")
        logging.info(f"Date Range: {self.start_date.strftime('%B %d, %Y')} - {self.end_date.strftime('%B %d, %Y')}")
        logging.info("=" * 70)
        
        start_time = time.time()
        total_items = end_id - start_id + 1
        items_to_process = [id for id in range(start_id, end_id + 1) if str(id) not in self.scraped_ids]
        items_processed = len(self.scraped_ids)
        
        logging.info(f"Total items to check: {total_items}")
        logging.info(f"Already processed: {items_processed} | Remaining: {len(items_to_process)}")
        logging.info("")
        
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            try:
                for idx, item_id in enumerate(range(start_id, end_id + 1), 1):
                    # Skip if already scraped
                    if str(item_id) in self.scraped_ids:
                        continue
                    
                    # Calculate progress
                    progress_pct = (idx / total_items) * 100
                    elapsed = time.time() - start_time
                    
                    # Estimate time remaining
                    if idx > items_processed + 1:
                        rate = (idx - items_processed) / elapsed
                        remaining_items = total_items - idx
                        eta_seconds = remaining_items / rate if rate > 0 else 0
                        eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    else:
                        eta_str = "calculating..."
                    
                    # Print progress every 10 items
                    if idx % 10 == 0 or idx == 1:
                        logging.info(f"[{progress_pct:5.1f}%] ItemId {item_id:4d}/{end_id} | Speeches: {len(self.speeches):3d} | ETA: {eta_str}")
                    
                    speech_data = self.scrape_speech(page, item_id)
                    
                    # Mark as scraped regardless of outcome
                    self.scraped_ids.add(str(item_id))
                    
                    if speech_data:
                        self.speeches.append(speech_data)
                        logging.info(f"  ✓ [{len(self.speeches):3d}] ItemId={item_id:4d}: {speech_data['Title'][:60]}")
                    
                    # Save progress every 25 speeches
                    if len(self.speeches) > 0 and len(self.speeches) % 25 == 0:
                        self.save_to_csv()
                        self.save_progress()
                        logging.info(f"  💾 Progress saved - {len(self.speeches)} speeches")
                    
                    # Very small delay
                    time.sleep(0.3)
                
            finally:
                browser.close()
        
        # Final save
        if self.speeches:
            self.save_to_csv()
            self.save_to_excel()
            self.save_progress()
            
            logging.info("")
            logging.info("=" * 70)
            logging.info("SCRAPING SUMMARY")
            logging.info("=" * 70)
            logging.info(f"Total speeches found: {len(self.speeches)}")
            logging.info(f"Date range: {self.start_date.strftime('%B %d, %Y')} - {self.end_date.strftime('%B %d, %Y')}")
            logging.info(f"Output files: {self.output_file}, {self.output_file.replace('.csv', '.xlsx')}")
            logging.info("=" * 70)
        else:
            logging.warning("No speeches found in the specified date range!")
    
    def save_to_csv(self):
        """Save to CSV"""
        if not self.speeches:
            return
        
        fieldnames = ['Title', 'Date', 'Place', 'Occasion', 'Speaker', 'Content', 'URL', 'ItemId']
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.speeches)
        logging.info(f"Saved {len(self.speeches)} speeches to {self.output_file}")
    
    def save_to_excel(self):
        """Save to Excel"""
        try:
            df = pd.DataFrame(self.speeches)
            excel_file = self.output_file.replace('.csv', '.xlsx')
            df.to_excel(excel_file, index=False, engine='openpyxl')
            logging.info(f"Saved to {excel_file}")
        except Exception as e:
            logging.warning(f"Could not save Excel: {e}")

if __name__ == "__main__":
    scraper = BSPSpeechScraperFinal()
    # Try all possible ItemIds from 1 to 1114 (latest ItemId)
    scraper.run(start_id=1, end_id=1114)
