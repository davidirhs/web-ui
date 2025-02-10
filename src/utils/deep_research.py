import asyncio
import json
import logging
import os
from uuid import uuid4
import time
import random

import rookiepy
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage
from json_repair import repair_json
from google.api_core.exceptions import ResourceExhausted  # Import the specific exception

from src.agent.custom_agent import CustomAgent
from browser_use.browser.browser import BrowserConfig, Browser
from browser_use.browser.context import BrowserContextConfig
from src.controller.custom_controller import CustomController
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from urllib.parse import urlparse, urlunparse


load_dotenv()
logger = logging.getLogger(__name__)


def normalize_url(url: str) -> str:
    """Normalizes a URL by removing query parameters, fragments, and trailing slashes."""
    try:
        parsed_url = urlparse(url)
        # Check if this is actually a URL before processing
        if not parsed_url.scheme or not parsed_url.netloc:
            return url  # Return original string if it's not a URL
        # Reconstruct URL without query parameters and fragment
        normalized_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, '', '', ''))
        # Remove trailing slash, but only if the path isn't just "/"
        if normalized_url.endswith('/') and parsed_url.path != '/':
            normalized_url = normalized_url[:-1]
        return normalized_url
    except Exception:
        return url  # Return original string if parsing fails


async def retry_for_rate_limit_per_minute(func, *args, max_retries=5, initial_wait=1, target_wait_time=59, **kwargs):
    """
    Retries a function with linearly increasing wait times, optimized for per-minute rate limits.
    Corrected to handle synchronous functions as well.

    Args:
        func: The function to retry (can be synchronous or asynchronous).
        *args: Positional arguments to pass to the function.
        max_retries: The maximum number of retries.
        initial_wait: The initial wait time in seconds for the first retry.
        target_wait_time: The target wait time in seconds to approach by the last retry (e.g., 59 for per-minute limit).
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function if successful, or raises ResourceExhausted if all retries fail.
    """
    retries = 0
    while retries < max_retries:
        try:
            return func(*args, **kwargs)  # Removed 'await' - assuming func might be synchronous
        except ResourceExhausted as e:
            retries += 1
            if retries >= max_retries:
                logger.error(f"Max retries reached for rate-limited function. Failing.")
                raise  # Re-raise the exception to signal failure

            # Linearly increase wait time, approaching target_wait_time
            if max_retries <= 1: #Handle edge case when max_retries is 1
                wait_time = target_wait_time
            else:
                wait_time = initial_wait + (target_wait_time - initial_wait) * (retries - 1) / (max_retries - 1)

            wait_time = min(wait_time, target_wait_time) #Ensure it never exceeds target_wait_time
            wait_time = max(wait_time, 0) # Ensure wait_time is not negative
            wait_time = wait_time * (1 + random.uniform(-0.1, 0.1))  # Jitter (reduced jitter)

            logger.warning(
                f"Rate limit hit (attempt {retries}/{max_retries}). Retrying in {wait_time:.2f} seconds..."
            )
            await asyncio.sleep(wait_time)  # Await the sleep (delay is still async)
        except Exception as e:
            logger.exception("An unexpected error occurred during retry:")
            raise  # Re-raise other exceptions

    logger.error("Max retries reached without success for rate-limited function.")
    raise ResourceExhausted("Max retries reached for rate-limited function.") #Explicitly raise error.

async def deep_research(task, llm, **kwargs):
    task_id = str(uuid4())
    save_dir = kwargs.get("save_dir", os.path.join(f"./tmp/deep_research/{task_id}"))
    logger.info(f"Save Deep Research at: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    max_query_num = kwargs.get("max_query_num", 3)
    visited_links = set()  # Initialize the set *here*

    search_system_prompt = kwargs.get("search_system_prompt", f"""
           You are a **Deep Researcher**, an AI agent specializing in in-depth information gathering and research using a web browser with **automated execution capabilities**. Your expertise lies in formulating comprehensive research plans and executing them meticulously to fulfill complex user requests. You will analyze user instructions, devise a detailed research plan, and determine the necessary search queries to gather the required information.

           **Your Task:**

           Given a user's research topic, you will:

           1.  **Develop a Research Plan:** Outline the key aspects and subtopics that need to be investigated to thoroughly address the user's request. This plan should be a high-level overview of the research direction.
           2.  **Generate Search Queries:** Based on your research plan, generate a list of specific search queries to be executed in a web browser. These queries should be designed to efficiently gather relevant information for each aspect of your plan.

           **Output Format:**

           Your output will be a JSON object with the following structure:

           ```json
           {{
           "plan": "A concise, high-level research plan outlining the key areas to investigate.",
             "queries": [
               "search query 1",
               "search query 2",
               //... up to a maximum of {max_query_num} search queries
             ]
           }}
           ```

           **Important:**

           *   Limit your output to a **maximum of {max_query_num}** search queries.
           *   Make the search queries to help the automated agent find the needed information. Consider what keywords are most likely to lead to useful results.
           *   If you have gathered for all the information you want and no further search queries are required, output queries with an empty list: `[]`
           *   Make sure output search queries are different from the history queries.

           **Inputs:**

           1.  **User Instruction:** The original instruction given by the user.
           2.  **Previous Queries:** History Queries.
           3.  **Previous Search Results:** Textual data gathered from prior search queries. If there are no previous search results this string will be empty.
       """)

    record_system_prompt = kwargs.get("record_system_prompt", """
           You are an expert information recorder. Your role is to process user instructions, current search results, and previously recorded information to extract, summarize, and record new, useful information that helps fulfill the user's request. Your output will be a JSON formatted list, where each element represents a piece of extracted information and follows the structure: `{"url": "source_url", "title": "source_title", "summary_content": "concise_summary", "thinking": "reasoning"}`.

       **Important Considerations:**

       1. **Minimize Information Loss:** While concise, prioritize retaining important details and nuances from the sources. Aim for a summary that captures the essence of the information without over-simplification. **Crucially, ensure to preserve key data and figures within the `summary_content`. This is essential for later stages, such as generating tables and reports.**

       2. **Avoid Redundancy:** Do not record information that is already present in the Previous Recorded Information. Check for semantic similarity, not just exact matches. However, if the same information is expressed differently in a new source and this variation adds valuable context or clarity, it should be included.

       3. **Source Information:** Extract and include the source title and URL for each piece of information summarized. This is crucial for verification and context. **The Current Search Results are provided in a specific format, where each item starts with "Title:", followed by the title, then "URL Source:", followed by the URL, and finally "Markdown Content:", followed by the content. Please extract the title and URL from this structure.** If a piece of information cannot be attributed to a specific source from the provided search results, use `"url": "unknown"` and `"title": "unknown"`.

       4. **Thinking and Report Structure:**  For each extracted piece of information, add a `"thinking"` key. This field should contain your assessment of how this information could be used in a report, which section it might belong to (e.g., introduction, background, analysis, conclusion, specific subtopics), and any other relevant thoughts about its significance or connection to other information.

       **Output Format:**

       Provide your output as a JSON formatted list. Each item in the list must adhere to the following format:

       ```json
       [
         {
           "url": "source_url_1",
           "title": "source_title_1",
           "summary_content": "Concise summary of content. Remember to include key data and figures here.",
           "thinking": "This could be used in the introduction to set the context. It also relates to the section on the history of the topic."
         },
         // ... more entries
         {
           "url": "unknown",
           "title": "unknown",
           "summary_content": "concise_summary_of_content_without_clear_source",
           "thinking": "This might be useful background information, but I need to verify its accuracy. Could be used in the methodology section to explain how data was collected."
         }
       ]
       ```

       **Inputs:**

       1.  **User Instruction:** The original instruction given by the user. This helps you determine what kind of information will be useful and how to structure your thinking.
       2.  **Previous Recorded Information:** Textual data gathered and recorded from previous searches and processing, represented as a single text string.
       3.  **Current Search Results:** Textual data gathered from the most recent search query.
           """)

    search_messages = [SystemMessage(content=search_system_prompt)]
    record_messages = [SystemMessage(content=record_system_prompt)]

    # --- COOKIE INJECTION (using rookiepy) ---
    use_own_cookies = kwargs.get("use_own_cookies", False)

    if use_own_cookies:
        try:
            # Use rookiepy to get Chrome cookies.
            #  -  We're passing an empty list for domains, meaning "get all cookies."
            #     If you only need cookies from specific sites (e.g., linkedin.com),
            #     you could improve efficiency by specifying them:  ["linkedin.com"]
            cookies = rookiepy.chrome([])  # Get cookies for all domains

            # Convert rookiepy cookies to the format expected by Playwright
            cookies_to_inject = [
                {
                    'name': c['name'],
                    'value': c['value'],
                    'domain': c['domain'],
                    'path': c['path'],
                    'expires': c['expires'] if c['expires'] != -1 else None, # -1 means session cookie
                    'httpOnly': c['httponly'],
                    'secure': c['secure'],
                    'sameSite': c['samesite'].capitalize() if c['samesite'] else 'Lax'  # Handle potential None and capitalize
                } for c in cookies
            ]
            logger.info(f"Successfully loaded {len(cookies_to_inject)} cookies using rookiepy.")

        except Exception as e:
            logger.error(f"Error loading cookies with rookiepy: {e}")
            cookies_to_inject = []  # Fallback: empty list
    else:
        cookies_to_inject = []
    # --- END COOKIE INJECTION ---


    # --- BROWSER INITIALIZATION (Simplified) ---
    headless = kwargs.get("headless", False)
    window_w = kwargs.get("window_w", 1280)  # default
    window_h = kwargs.get("window_h", 1100)  # default

    browser = Browser(
        config=BrowserConfig(
            disable_security=True,  # Important for interacting with many sites
            headless=headless,
            extra_chromium_args=[f"--window-size={window_w},{window_h}"],
        )
    )
    # --- END BROWSER INITIALIZATION ---

    controller = CustomController()
    search_iteration = 0
    max_search_iterations = kwargs.get("max_search_iterations", 10)
    use_vision = kwargs.get("use_vision", False)

    history_query = []
    history_infos = []
    try:
        while search_iteration < max_search_iterations:
            search_iteration += 1
            logger.info(f"Start {search_iteration}th Search...")

            history_query_ = json.dumps(history_query, indent=4)
            history_infos_ = json.dumps(history_infos, indent=4)

            query_prompt = (
                f"This is search {search_iteration} of {max_search_iterations} maximum searches allowed.\n"
                f"User Instruction:{task} \n Previous Queries:\n {history_query_} \n"
                f"Previous Search Results:\n {history_infos_}\n"
            )
            search_messages.append(HumanMessage(content=query_prompt))

            # Use the retry function here!
            ai_query_msg = await retry_for_rate_limit_per_minute(llm.invoke, search_messages[:1] + search_messages[1:][-1:]) #Corrected line
            if ai_query_msg is None:  # Retry failed
                break

            search_messages.append(ai_query_msg)


            if hasattr(ai_query_msg, "reasoning_content"):
                logger.info("🤯 Start Search Deep Thinking: ")
                logger.info(ai_query_msg.reasoning_content)
                logger.info("🤯 End Search Deep Thinking")

            ai_query_content = ai_query_msg.content.replace("```json", "").replace("```", "")
            ai_query_content = repair_json(ai_query_content)
            try:
                ai_query_content = json.loads(ai_query_content)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON: {ai_query_content}")
                break

            query_plan = ai_query_content["plan"]
            logger.info(f"Current Iteration {search_iteration} Planing:")
            logger.info(query_plan)
            query_tasks = ai_query_content["queries"]

            if not query_tasks:
                break
            else:
                history_query.extend(query_tasks)
                logger.info("Query tasks:")
                logger.info(query_tasks)

            add_infos = (
                "1. Please click on the most relevant link to get information and go deeper, instead of just staying on the search page. \n"
                "2. When opening a PDF file, please remember to extract the content using extract_content instead of simply opening it for the user to view."
                "3. Please do not revisit any URL's that have already been visited in this or any previous search iterations."

            )
            contexts = []
            for _ in range(len(query_tasks)):
                context = await browser.new_context(config=BrowserContextConfig())
                if cookies_to_inject:
                    await context.context.add_cookies(cookies_to_inject)
                contexts.append(context)

            # *** CRITICAL CHANGE: Filter query_tasks BEFORE creating agents ***
           
            filtered_query_tasks = []
            for task in query_tasks:
                try:
                    parsed = urlparse(task)
                    if parsed.scheme and parsed.netloc:
                        norm = normalize_url(task)
                        if norm in visited_links:
                            logger.info(f"Skipping already visited URL: {task}")
                            continue
                        # Mark the URL as visited immediately
                        visited_links.add(norm)
                except Exception:
                    pass
                filtered_query_tasks.append(task)


            if not filtered_query_tasks:
                logger.info("All URLs already visited. Stopping.")
                break
            agents = [
                CustomAgent(
                    task=task,
                    llm=llm,
                    add_infos=add_infos,
                    browser_context=contexts[i],
                    use_vision=use_vision,
                    system_prompt_class=CustomSystemPrompt,
                    agent_prompt_class=CustomAgentMessagePrompt,    
                    max_actions_per_step=5,
                    controller=controller
                ) for i, task in enumerate(filtered_query_tasks) # Use filtered tasks
            ]



            query_results = await asyncio.gather(*[agent.run(max_steps=kwargs.get("max_steps", 10)) for agent in agents])



            for context in contexts:
                await context.close()

            query_result_dir = os.path.join(save_dir, "query_results")
            os.makedirs(query_result_dir, exist_ok=True)
            for i in range(len(filtered_query_tasks)):
                if query_results[i] is None:
                    continue
                query_result = query_results[i].final_result()
                querr_save_path = os.path.join(query_result_dir, f"{search_iteration}-{i}.md")
                logger.info(f"save query: {filtered_query_tasks[i]} at {querr_save_path}")
                with open(querr_save_path, "w", encoding="utf-8") as fw:
                    fw.write(f"Query: {filtered_query_tasks[i]}\n")
                    if query_result is not None:
                        fw.write(query_result)
                    else:
                        fw.write("No final result was returned for this query.\n")
                        logger.warning(f"Query '{filtered_query_tasks[i]}' returned None.")

            history_infos_ = json.dumps(history_infos, indent=4)
            record_prompt = (
                f"User Instruction:{task}. \nPrevious Recorded Information:\n {json.dumps(history_infos_)} \n"
                f"Current Search Results: {query_result}\n"
            )
            record_messages.append(HumanMessage(content=record_prompt))

            ai_record_msg = await retry_for_rate_limit_per_minute(llm.invoke, record_messages[:1] + record_messages[-1:])

            if ai_record_msg is None:
                break

            record_messages.append(ai_record_msg)


            if hasattr(ai_record_msg, "reasoning_content"):
                logger.info("🤯 Start Record Deep Thinking: ")
                logger.info(ai_record_msg.reasoning_content)
                logger.info("🤯 End Record Deep Thinking")

            record_content = ai_record_msg.content
            record_content = repair_json(record_content)
            try:
                new_record_infos = json.loads(record_content)
                for info in new_record_infos:
                    if "url" in info and info["url"] != "unknown":
                        norm_url = normalize_url(info["url"])
                        visited_links.add(norm_url)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON: {record_content}")
                new_record_infos = []

            history_infos.extend(new_record_infos)

        # 5. Report Generation in Markdown (or JSON if you prefer)
        writer_system_prompt = """
            You are a **Deep Researcher** and a professional report writer tasked with creating polished, high-quality reports that fully meet the user's needs, based on the user's instructions and the relevant information provided. You will write the report using Markdown format, ensuring it is both informative and visually appealing.

    **Specific Instructions:**

    *   **Structure for Impact:** The report must have a clear, logical, and impactful structure. Begin with a compelling introduction that immediately grabs the reader's attention. Develop well-structured body paragraphs that flow smoothly and logically, and conclude with a concise and memorable conclusion that summarizes key takeaways and leaves a lasting impression.
    *   **Engaging and Vivid Language:** Employ precise, vivid, and descriptive language to make the report captivating and enjoyable to read. Use stylistic techniques to enhance engagement. Tailor your tone, vocabulary, and writing style to perfectly suit the subject matter and the intended audience to maximize impact and readability.
    *   **Accuracy, Credibility, and Citations:** Ensure that all information presented is meticulously accurate, rigorously truthful, and robustly supported by the available data. **Cite sources exclusively using bracketed sequential numbers within the text (e.g., [1], [2], etc.). If no references are used, omit citations entirely.** These numbers must correspond to a numbered list of references at the end of the report.
    *   **Publication-Ready Formatting:** Adhere strictly to Markdown formatting for excellent readability and a clean, highly professional visual appearance. Pay close attention to formatting details like headings, lists, emphasis, and spacing to optimize the visual presentation and reader experience. The report should be ready for immediate publication upon completion, requiring minimal to no further editing for style or format.
    *   **Conciseness and Clarity (Unless Specified Otherwise):** When the user does not provide a specific length, prioritize concise and to-the-point writing, maximizing information density while maintaining clarity.
    *   **Data-Driven Comparisons with Tables:**  **When appropriate and beneficial for enhancing clarity and impact, present data comparisons in well-structured Markdown tables. This is especially encouraged when dealing with numerical data or when a visual comparison can significantly improve the reader's understanding.**
    *   **Length Adherence:** When the user specifies a length constraint, meticulously stay within reasonable bounds of that specification, ensuring the content is appropriately scaled without sacrificing quality or completeness.
    *   **Comprehensive Instruction Following:** Pay meticulous attention to all details and nuances provided in the user instructions. Strive to fulfill every aspect of the user's request with the highest degree of accuracy and attention to detail, creating a report that not only meets but exceeds expectations for quality and professionalism.
    *   **Reference List Formatting:** The reference list at the end must be formatted as follows:
        `[1] Title (URL, if available)`
        **Each reference must be separated by a blank line to ensure proper spacing.** For example:

        ```
        [1] Title 1 (URL1, if available)

        [2] Title 2 (URL2, if available)
        ```
        **Furthermore, ensure that the reference list is free of duplicates. Each unique source should be listed only once, regardless of how many times it is cited in the text.**
    *   **ABSOLUTE FINAL OUTPUT RESTRICTION:**  **Your output must contain ONLY the finished, publication-ready Markdown report. Do not include ANY extraneous text, phrases, preambles, meta-commentary, or markdown code indicators (e.g., "```markdown```"). The report should begin directly with the title and introductory paragraph, and end directly after the conclusion and the reference list (if applicable).**  **Your response will be deemed a failure if this instruction is not followed precisely.**

    **Inputs:**

    1.  **User Instruction:** The original instruction given by the user. This helps you determine what kind of information will be useful and how to structure your thinking.
    2.  **Search Information:** Information gathered from the search queries.
            """

        history_infos_ = json.dumps(history_infos, indent=4)
        record_json_path = os.path.join(save_dir, "record_infos.json")
        logger.info(f"save All recorded information at {record_json_path}")
        with open(record_json_path, "w", encoding="utf-8") as fw:
            json.dump(history_infos, fw, indent=4, ensure_ascii=False)

        report_prompt = f"User Instruction:{task} \n Search Information:\n {history_infos_}"
        report_messages = [SystemMessage(content=writer_system_prompt), HumanMessage(content=report_prompt)]

        # Retry the report generation too!
        ai_report_msg = await retry_for_rate_limit_per_minute(llm.invoke, report_messages) #Corrected Line
        if ai_report_msg is None:
            return "", None

        if hasattr(ai_report_msg, "reasoning_content"):
            logger.info("🤯 Start Report Deep Thinking: ")
            logger.info(ai_report_msg.reasoning_content)
            logger.info("🤯 End Report Deep Thinking")

        report_content = ai_report_msg.content
        report_file_path = os.path.join(save_dir, "final_report.md")
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info(f"Save Report at: {report_file_path}")
        return report_content, report_file_path

    except Exception as e:
        logger.exception("An error occurred during deep research:")
        return "", None  # Return empty string and None on error

    finally:
        if browser:
            await browser.close()
        logger.info("Browser closed.")