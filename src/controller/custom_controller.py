import pyperclip, csv, asyncio, logging
from pathlib import Path
from typing import Optional, Type
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller
from main_content_extractor import MainContentExtractor
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
#Import Job Model
from src.extensions.job_search.models import Job
#Import Pdf Reader
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


class CustomController(Controller):
    def __init__(self, cv_path:  Optional[Path] = None, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self.cv_path = cv_path  # Store cv_path
        self._register_custom_actions()

    def is_linkedin_application_page(self, url: str) -> bool:
        """Checks if the current URL is a LinkedIn application page."""
        return "linkedin.com/jobs/collections/recommended/" in url or "linkedin.com/jobs/view/" in url

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action("Copy text to clipboard")
        def copy_to_clipboard(text: str):
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)

        @self.registry.action("Paste text from clipboard", requires_browser=True)
        async def paste_from_clipboard(browser: BrowserContext):
            text = pyperclip.paste()
            # send text to browser
            page = await browser.get_current_page()
            await page.keyboard.type(text)

            return ActionResult(extracted_content=text)

        @self.registry.action(
            'Extract page content to get the pure text or markdown with links if include_links is set to true',
            param_model=ExtractPageContentAction,
            requires_browser=True,
        )
        async def extract_content(params: ExtractPageContentAction, browser: BrowserContext):
            page = await browser.get_current_page()
            # use jina reader
            url = page.url
            jina_url = f"https://r.jina.ai/{url}"
            await page.goto(jina_url)
            output_format = 'markdown' if params.include_links else 'text'
            content = MainContentExtractor.extract(  # type: ignore
                html=await page.content(),
                output_format=output_format,
            )
            # go back to org url
            await page.go_back()
            msg = f'📄  Extracted page content as {output_format}\n: {content}\n'
            logger.info(msg)
            return ActionResult(extracted_content=msg)

        @self.registry.action("Read my cv for context to fill forms")
        def read_cv() -> ActionResult:
            # --- Access self.cv_path directly, and check for None ---
            if not self.cv_path or not self.cv_path.exists():
                msg = f"CV file not found at {self.cv_path}"
                logger.error(msg)
                return ActionResult(error=msg)
            # --- End of corrected section ---
            try:
                reader = PdfReader(str(self.cv_path))
                text = "".join(page.extract_text() or "" for page in reader.pages)
                logger.info(f"CV read with {len(text)} characters")
                return ActionResult(extracted_content=text, include_in_memory=True)
            except Exception as e:
                msg = f"Error reading CV: {e}"
                logger.error(msg)
                return ActionResult(error=msg)

        @self.registry.action("Upload cv to element - try alternate indexes if needed", requires_browser=True)
        async def upload_cv(index: int, browser: BrowserContext) -> ActionResult:
            # --- Access self.cv_path directly, and check for None ---
            if not self.cv_path or not self.cv_path.exists():
                msg = f"CV file not found at {self.cv_path}"
                logger.error(msg)
                return ActionResult(error=msg)
            # --- End of corrected section ---
            path = str(self.cv_path.absolute())
            dom_el = await browser.get_dom_element_by_index(index)
            if dom_el is None:
                return ActionResult(error=f"No element at index {index}")
            file_upload_dom_el = dom_el.get_file_upload_element()
            if not file_upload_dom_el:
                return ActionResult(error=f"No file upload element at index {index}")
            file_upload_el = await browser.get_locate_element(file_upload_dom_el)
            if not file_upload_el:
                return ActionResult(error=f"Could not locate upload element at index {index}")
            try:
                await file_upload_el.set_input_files(path)
                msg = f"Uploaded CV to element at index {index}"
                logger.info(msg)
                return ActionResult(extracted_content=msg)
            except Exception as e:
                msg = f"Failed to upload file at index {index}: {e}"
                logger.debug(msg)
                return ActionResult(error=msg)

        @self.registry.action("Save jobs to file - with a score how well it fits to my profile", param_model=Job)
        def save_jobs(job: Job) -> str:
            with open("jobs.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    job.title, job.company, job.link,
                    job.salary, job.location,
                    job.fit_score, job.fit_score_explanation
                ])
            return "Saved job to file"

        @self.registry.action("Read jobs from file")
        def read_jobs() -> str:
            try:
                with open("jobs.csv", "r") as f:
                    return f.read()
            except FileNotFoundError:
                return "No jobs.csv file found."

        @self.registry.action("Click LinkedIn Easy Apply Button", requires_browser=True)
        async def click_linkedin_easy_apply(browser: BrowserContext) -> ActionResult:
            try:
                page = await browser.get_current_page()
                button = page.get_by_role("button", name="Easy Apply")
                await button.click()
                return ActionResult(extracted_content="Clicked Easy Apply")
            except Exception as e:
                return ActionResult(error=f"Easy Apply click failed: {e}")

        @self.registry.action("Click LinkedIn Next Button", requires_browser=True)
        async def click_linkedin_next_button(browser: BrowserContext) -> ActionResult:
            selectors = [
                'button[aria-label="Continue to next step"].artdeco-button--primary',
                'button[aria-label="Continue to next step"]',
                'button.artdeco-button--primary span:contains("Next")',
                'button.artdeco-button--primary'
            ]
            page = await browser.get_current_page()
            for sel in selectors:
                try:
                    btn = await page.locator(f'css={sel}').element_handle(timeout=2000)
                    if btn:
                        await page.evaluate('(e) => e.scrollIntoView({block:"center", inline:"center"})', btn)
                        await asyncio.sleep(0.5)
                        await btn.click()
                        return ActionResult(extracted_content="Clicked Next")
                except Exception as e:
                    logger.debug(f"Next selector {sel} failed: {e}")
            return ActionResult(error="Could not click Next button.")

        @self.registry.action("Click LinkedIn Review Button", requires_browser=True)
        async def click_linkedin_review_button(browser: BrowserContext) -> ActionResult:
            selectors = [
                'button[aria-label="Review your application"].artdeco-button--primary',
                'button[aria-label="Review your application"]',
                'button.artdeco-button--primary span:contains("Review")',
                'button.artdeco-button--primary'
            ]
            page = await browser.get_current_page()
            for sel in selectors:
                try:
                    btn = await page.locator(f'css={sel}').element_handle(timeout=5000)
                    if btn:
                        await page.evaluate('(e) => e.scrollIntoView({block:"center", inline:"center"})', btn)
                        await asyncio.sleep(0.5)
                        await btn.click()
                        return ActionResult(extracted_content="Clicked Review")
                except Exception as e:
                    logger.debug(f"Review selector {sel} failed: {e}")
            return ActionResult(error="Could not click Review button.")

        @self.registry.action("Click LinkedIn Submit Button", requires_browser=True)
        async def click_linkedin_submit_button(browser: BrowserContext) -> ActionResult:
            selectors = [
                'button[aria-label="Submit application"].artdeco-button--primary',
                'button[aria-label="Submit application"]',
                'button.artdeco-button--primary span:contains("Submit application")',
                'button.artdeco-button--primary'
            ]
            page = await browser.get_current_page()
            for sel in selectors:
                try:
                    btn = await page.locator(f'css={sel}').element_handle(timeout=5000)
                    if btn:
                        await page.evaluate('(e) => e.scrollIntoView({block:"center", inline:"center"})', btn)
                        await asyncio.sleep(0.5)
                        await btn.click()
                        return ActionResult(extracted_content="Clicked Submit")
                except Exception as e:
                    logger.debug(f"Submit selector {sel} failed: {e}")
            return ActionResult(error="Could not click Submit button.")

        @self.registry.action("Extract job description", requires_browser=True)
        async def extract_job_description(browser: BrowserContext) -> ActionResult:
            try:
                page = await browser.get_current_page()
                js = """
                    () => {
                        const container = document.querySelector('.jobs-description__container');
                        return container ? container.textContent.trim() : null;
                    }
                """
                content = await page.evaluate(js)
                if content:
                    return ActionResult(extracted_content=content, include_in_memory=True)
                return ActionResult(error="Job description container not found.")
            except Exception as e:
                msg = f"Job description extraction error: {e}"
                logger.error(msg)
                return ActionResult(error=msg)

        @self.registry.action("Click Generic Save Button", requires_browser=True)
        async def click_generic_save_button(browser: BrowserContext) -> ActionResult:
            selectors = ['.jobs-save-button.artdeco-button--3', '.jobs-save-button']
            page = await browser.get_current_page()
            for sel in selectors:
                try:
                    btn = await page.locator(f'css={sel}').element_handle(timeout=2000)
                    if btn:
                        await btn.click()
                        return ActionResult(extracted_content="Clicked generic save button.")
                except Exception as e:
                    logger.debug(f"Generic save selector {sel} failed: {e}")
            return ActionResult(error="Could not click generic save button.")

        @self.registry.action("Click Save Button for Job", requires_browser=True)
        async def click_save_button_for_job(browser: BrowserContext, job_title: str) -> ActionResult:
            selector = f'button.jobs-save-button[aria-label*="Save {job_title}"]'
            page = await browser.get_current_page()
            try:
                btn = await page.locator(f'css={selector}').element_handle(timeout=5000)
                if not btn:
                    return ActionResult(error=f"Save button not found for '{job_title}'")
                await btn.click()
                saved_text_sel = f'{selector} span.jobs-save-button__text'
                saved_el = await page.locator(f'css={saved_text_sel}').element_handle(timeout=5000)
                if saved_el:
                    text = (await saved_el.inner_text()).strip().lower()
                    if text == "saved":
                        return ActionResult(extracted_content=f"Saved job '{job_title}' successfully.")
                    return ActionResult(error=f"Save verification failed. Button text: '{text}'")
                return ActionResult(error="Save verification failed: text element not found.")
            except Exception as e:
                return ActionResult(error=f"Error saving job '{job_title}': {e}")

        @self.registry.action("Click Dismiss Button for Job", requires_browser=True)
        async def click_dismiss_button_for_job(browser: BrowserContext, job_name: str) -> ActionResult:
            selector = f'button[aria-label="Dismiss {job_name} job"]'
            page = await browser.get_current_page()
            try:
                btn = await page.locator(f'css={selector}').element_handle(timeout=5000)
                if not btn:
                    return ActionResult(error=f"Dismiss button not found for '{job_name}'")
                await btn.click()
                # Verification: check if aria-label has updated
                updated_selector = selector.replace("Dismiss", f"{job_name} job is dismissed, undo")
                updated_btn = await page.locator(f'css={updated_selector}').element_handle(timeout=5000)
                if updated_btn:
                    aria = await updated_btn.get_attribute("aria-label")
                    if aria and aria.startswith(f"{job_name} job is dismissed, undo"):
                        return ActionResult(extracted_content=f"Dismissed job '{job_name}' successfully.")
                    return ActionResult(error=f"Dismiss verification failed. Aria-label: {aria}")
                return ActionResult(error="Dismiss verification failed: Updated button not found.")
            except Exception as e:
                return ActionResult(error=f"Error dismissing job '{job_name}': {e}")