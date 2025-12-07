# from llamaindex.agents import AgentExecutor, ToolAgent
from dotenv import load_dotenv
import asyncio
import os
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import PromptTemplate
from serpapi.google_search import GoogleSearch
#from llama_index.core.agent.function_agent import FunctionAgent
os.environ["LLAMA_INDEX_DISABLE_PG"] = "true"

# Load environmental variables (API keys)
load_dotenv('config.env')

SERPAPI_KEY = os.getenv("SERP_API_KEY")
OPENAI_API_KEY = os.getenv('OPENAI_KEY')

# Define SerpAPI tool for web search
def serp_search(query: str):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 5
    }
    results = GoogleSearch(params).get_dict()
    out = []
    for r in results.get("organic_results", []):
        out.append(f"{r.get('title')}\n{r.get('snippet')}\n{r.get('link')}")
    return "\n\n".join(out)

serp_tool = FunctionTool.from_defaults(
    fn=serp_search,
    name="serp_search",
    description="Search the internet for recent information using SerpAPI."
)

#### AGENT PROMPTS ####

system_prompt = """System/Role: You are an AI financial analysis engine specializing in mid-cap stocks on the NASDAQ exchange. 
Your task is to evaluate the given ticker using INTUITION, data, financial metrics, and industry fundamentals and to produce a structured, analytical, explainable report.

Instructions:
For the provided ticker, ask and answer the following questions. For every question, cite the data used (e.g., revenue, margins, user metrics, analyst notes) and state any assumptions. If a value is unavailable or cannot be determined from public data, state "NA" and explain why.

SECTION 1 — INDUSTRY & MARKET (ask then answer)
1.1 What industry and sub-industry is the company in?
1.2 What are the industry-specific questions relevant to this company? (List at least 6 — e.g., for EdTech: user retention, certification partnerships, regulatory exposure, localization, content cost, B2B vs B2C mix.)
1.3 Market size & growth:
    - Current global market size (USD) for the company’s core market(s) and 3-5yr CAGR estimates (cite sources).
    - Estimate the company’s realistic Serviceable Addressable Market (SAM) and near-term obtainable share (quantify assumptions).
1.4 What macro, regulatory, and structural factors will affect future stability & growth? (List positives and negatives.)
1.5 Bull case tailwinds (3 bullet points).
1.6 Major challenges / downside risks (3 bullet points).

SECTION 2 — MANAGEMENT & GOVERNANCE (ask then answer)
2.1 Who are the CEO, CTO, CFO, and at least two other senior leaders? Provide short bios (education, prior roles).
2.2 Describe leadership & culture style (data: interviews, proxy statements, public statements). What evidence supports your view?
2.3 Board composition: independence, relevant expertise, rotation, recent additions/removals.
2.4 Significant management strengths (3) and weaknesses (3).
2.5 Governance risks (insider selling, compensation misalignment, shareholder activism, regulatory exposures).

SECTION 3 — PRODUCT & TECHNOLOGY (ask then answer)
3.1 What are the company’s primary products / services and revenue streams? (Percent splits if known.)
3.2 How effective is the product? Use these subquestions:
    - Core engagement metrics (MAU/DAU, retention, cohort retention if available).
    - Measured efficacy (outcomes, third-party validations, certifications).
    - Feature differentiation (AI, IP, patents, unique data).
    - Scalability and marginal cost per user/customer.
3.3 Product roadmap & R&D: recent launches, AI adoption, time-to-market risks.
3.4 UX / customer satisfaction signals (NPS, reviews, churn drivers).
3.5 Product weaknesses and potential technical debt.

SECTION 4 — MARKET FINANCIALS & MODELING (ask then answer)
4.1 Key historical financials (last 3 years and most recent quarter): revenue, YoY growth, gross margin, operating margin, net income, FCF, cash balance, debt. Cite sources.
4.2 Unit economics (if applicable): LTV, CAC, LTV/CAC, gross retention, net retention, CAC payback.
4.3 Valuation snapshot: current market cap, P/E, EV/EBITDA, EV/Sales, Price/FCF; compare to sector and midcap medians.
4.4 Analyst consensus: consensus rating distribution, average 12-month target, and recent upgrades/downgrades (cite sources).
4.5 Scenario modeling (3-year and 5-year): produce three scenarios (Bear, Base, Bull). For each scenario provide:
    - Assumptions: revenue CAGRs, margin progression, multiple assumptions.
    - Projected revenue and EPS at horizon.
    - Implied per-share price range (show math: revenue × multiple, or DCF inputs if used).
    - Expected % return per $1 invested (no advice).
4.6 Risks that would invalidate each scenario.

SECTION 5 — FINAL SYNTHESIS & RATINGS (ask then answer)
5.1 Punchline paragraph: one concise paragraph answering “Does this stock have sizeable gains potential in 3–5 years?” (No advice — factual probability / scenario language only.)
5.2 Rate each of the four categories on a 1–10 scale (Industry, Management, Product, Market Financials). For each rating give 2–3 bullet points justification and a single KPI that most influenced the score.
5.3 Composite score: give overall **Expected 3–5yr Growth Rating (1–10)** and a **Confidence Score (0–100%)**. Explain how the confidence is computed (data coverage, model variance, regulatory uncertainty).
5.4 Top 5 catalysts that would materially increase the rating.
5.5 Top 5 red flags that would materially decrease the rating.

SECTION 6 — EXPLAINABILITY & DATA SOURCES (mandatory)
6.1 List all sources used (news, filings, APIs) with timestamps.
6.2 Provide the most influential quantitative drivers (top 8) that moved the model/reasoning toward the verdict and give their approximate effect (e.g., “+1.2 points from 40% YoY revenue growth”).
6.3 If a model (e.g., XGBoost) was used, include the model version, training data date range, top 10 features by importance, and a SHAP-style explanation for the current prediction.

SECTION 7 — OUTPUT FORMAT (machine-friendly)
Return a JSON object with the following top-level keys:
{
  "ticker": string,
  "timestamp": ISO8601,
  "industry": {...},
  "management": {...},
  "product": {...},
  "market_financials": {...},
  "scenarios": { "bear": {...}, "base": {...}, "bull": {...} },
  "ratings": { "industry": number, "management": number, "product": number, "financials": number, "composite": number, "confidence": number },
  "punchline": string,
  "catalysts_up": [string],
  "red_flags": [string],
  "sources": [ {"name":string, "url":string, "date":ISO8601} ],
  "explainability": {...}
}

**Return the results in a JSON format. Use each subquestion as a key.**

CONSTRAINTS & SAFETY:
- Provide direct investment advice, trading instructions, or personalized recommendations. State your confidence interval, and explain why you are that confident.
- When you estimate prices/returns, always show the underlying assumptions and math.  
- If any regulatory/legal advice is relevant, add the disclaimer and recommend consulting counsel.  
- Keep the answer length to a maximum of ~1500 words for the human summary, but include detailed numeric tables in the JSON.

End.
"""

industry_prompt = """
You are the Industry Analysis Agent.
Your job: analyze industry structure, market size, growth, and macro factors.
INDUSTRY & MARKET (ask then answer)
1.1 What industry and sub-industry is the company in?
1.2 What are the industry-specific questions relevant to this company? (List at least 6 — e.g., for EdTech: user retention, certification partnerships, regulatory exposure, localization, content cost, B2B vs B2C mix.)
1.3 Market size & growth:
    - Current global market size (USD) for the company’s core market(s) and 3-5yr CAGR estimates (cite sources).
    - Estimate the company’s realistic Serviceable Addressable Market (SAM) and near-term obtainable share (quantify assumptions).
1.4 What macro, regulatory, and structural factors will affect future stability & growth? (List positives and negatives.)
1.5 Bull case tailwinds (3 bullet points).
1.6 Major challenges / downside risks (3 bullet points).
Return a JSON string with only the industry section completed.
"""

product_prompt = """
You are the Product Analysis Agent.
Your job: analyze the company's products, technology, and user metrics.
PRODUCT & TECHNOLOGY (ask then answer)
3.1 What are the company’s primary products / services and revenue streams? (Percent splits if known.)
3.2 How effective is the product? Use these subquestions:
    - Core engagement metrics (MAU/DAU, retention, cohort retention if available).
    - Measured efficacy (outcomes, third-party validations, certifications).
    - Feature differentiation (AI, IP, patents, unique data).
    - Scalability and marginal cost per user/customer.
3.3 Product roadmap & R&D: recent launches, AI adoption, time-to-market risks.
3.4 UX / customer satisfaction signals (NPS, reviews, churn drivers).
3.5 Product weaknesses and potential technical debt.
Return a JSON string with only the product section completed."""

management_prompt = """
You are the Management Analysis Agent.
Your job: analyze the company's leadership, governance, and board structure.
MANAGEMENT & GOVERNANCE (ask then answer)
2.1 Who are the CEO, CTO, CFO, and at least two other senior leaders? Provide short bios (education, prior roles).
2.2 Describe leadership & culture style (data: interviews, proxy statements, public statements). What evidence supports your view?
2.3 Board composition: independence, relevant expertise, rotation, recent additions/removals.
2.4 Significant management strengths (3) and weaknesses (3).
2.5 Governance risks (insider selling, compensation misalignment, shareholder activism, regulatory exposures).
Return a JSON string.
"""
financials_prompt = """
You are the Financials Analysis Agent.
Your job: analyze the company's financial statements, valuation, and scenario modeling.
MARKET FINANCIALS & MODELING (ask then answer)
4.1 Key historical financials (last 3 years and most recent quarter): revenue, YoY growth, gross margin, operating margin, net income, FCF, cash balance, debt. Cite sources.
4.2 Unit economics (if applicable): LTV, CAC, LTV/CAC, gross retention, net retention, CAC payback.
4.3 Valuation snapshot: current market cap, P/E, EV/EBITDA, EV/Sales, Price/FCF; compare to sector and midcap medians.
4.4 Analyst consensus: consensus rating distribution, average 12-month target, and recent upgrades/downgrades (cite sources).
4.5 Scenario modeling (3-year and 5-year): produce three scenarios (Bear, Base, Bull). For each scenario provide:
    - Assumptions: revenue CAGRs, margin progression, multiple assumptions.
    - Projected revenue and EPS at horizon.
    - Implied per-share price range (show math: revenue × multiple, or DCF inputs if used).
    - Expected % return per $1 invested (no advice).
4.6 Risks that would invalidate each scenario.
Return a JSON string with only the market_financials and scenarios sections completed.
"""
supervisor_prompt = """
You are the Supervisor Agent overseeing four specialized agents: Industry, Management, Product, and Financials Analysis Agents.
Your job: coordinate these agents to produce a comprehensive financial analysis report for the given ticker.

FINAL SYNTHESIS & RATINGS (ask then answer)
5.1 Punchline paragraph: one concise paragraph answering “Does this stock have sizeable gains potential in 3–5 years?” (No advice — factual probability / scenario language only.)
5.2 Rate each of the four categories on a 1–10 scale (Industry, Management, Product, Market Financials). For each rating give 2–3 bullet points justification and a single KPI that most influenced the score.
5.3 Composite score: give overall **Expected 3–5yr Growth Rating (1–10)** and a **Confidence Score (0–100%)**. Explain how the confidence is computed (data coverage, model variance, regulatory uncertainty).
5.4 Top 5 catalysts that would materially increase the rating.
5.5 Top 5 red flags that would materially decrease the rating.

"""
## Define agents ##

industry_agent = ReActAgent.from_tools(
    tools=[serp_tool],  # optional if you want web scraping
    llm=OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    system_prompt=industry_prompt,
    verbose=True,
)

management_agent = ReActAgent.from_tools(
    tools=[serp_tool],
    llm=OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    system_prompt=management_prompt,
    verbose=True,
)
product_agent = ReActAgent.from_tools(
    tools=[serp_tool],
    llm=OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    system_prompt=product_prompt,
    verbose=True,
)

financials_agent = ReActAgent.from_tools(
    tools=[serp_tool],
    llm=OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    system_prompt=financials_prompt,
    verbose=True,
)

def call_industry_agent(query: str):
    return industry_agent.query(f"Given {industry_prompt}, Analyze this ticker: {query}").response

def call_management_agent(query: str):
    return management_agent.query(f"Given {management_prompt} instruction, Analyze this ticker: {query}").response

def call_product_agent(query: str):
    return product_agent.query(f"Given {product_prompt}, Analyze this ticker: {query}").response

def call_financials_agent(query: str):
    return financials_agent.query(f"Given {financials_prompt}, Analyze this ticker: {query}").response

industry_tool = FunctionTool.from_defaults(
    fn=call_industry_agent,
    name="industry_agent",
    description="Runs an industry analysis agent and returns JSON."
)

management_tool = FunctionTool.from_defaults(
    fn=call_management_agent,
    name="management_agent",
    description="Runs a management analysis agent and returns JSON."
)

product_tool = FunctionTool.from_defaults(
    fn=call_product_agent,
    name="product_agent",
    description="Runs a product analysis agent and returns JSON."
)

financials_tool = FunctionTool.from_defaults(
    fn=call_financials_agent,
    name="financials_agent",
    description="Runs a financials analysis agent and returns JSON."
)



supervisor_agent = ReActAgent.from_tools(
    tools=[industry_tool, management_tool, product_tool, financials_tool, serp_tool],
    llm=OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    system_prompt=supervisor_prompt,
    verbose=True,
)

llm = OpenAI(model = "gpt-4", 
             api_key=OPENAI_API_KEY,
            system_prompt= system_prompt,)

# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
# new_react_header_template = PromptTemplate(
#     input_variables=["tool_names", "tool_descriptions", "input"],
#     template="""You are a helpful assistant that can use the following tools:
    
#     {tool_names_and_descriptions}
    
#     When given a user input, decide which tool to use and provide the appropriate input to that tool.
    
#     User Input: {input}
    
#     Remember to think step-by-step and use the tools wisely.""",
# )

# Create an agent workflow with our calculator tool
# agent = ReActAgent.from_tools(
#     tools=[serp_tool],
#     llm=llm,
# )
def run_pipeline(ticker: str):
    result = supervisor_agent.query(f"Given {system_prompt} and {supervisor_prompt}, Analyze this ticker: {ticker}")
    return result.response

async def main():
    # prompt_dict = agent.get_prompts()
    # for k, v in prompt_dict.items():
    #     print(f"Prompt: {k}\n\nValue: {v.template}")
    # Run the agent
    # print(OPENAI_API_KEY)
    response = run_pipeline("AAL")
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())